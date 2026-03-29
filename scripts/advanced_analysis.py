
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy import stats
import os
import json

from .paths import (
    ADVANCED_ANALYSIS_JSON,
    FIG_CONS,
    FIG_LAB,
    OUT_ADVANCED,
    OUTPUTS_DIR,
)

import statsmodels.api as sm
import statsmodels.formula.api as smf

# GLMM (generalized linear mixed-effects models)
def run_glmm(df, choice_col='sequenceChoiceLeft', subject_col='subject'):
    """
    Choice ~ Evidence_Strength * Complexity + (1|Subject)
    GEE (Generalized Estimating Equations) - accounts for hierarchy: rounds nested in subjects.
    """

    # evidence_strength = |EV_diff| (strength of evidence for left vs right)
    # complexity - already in df
    df = df.copy()
    df['Evidence_Strength'] = np.abs(df['EV_diff'])
    df['Choice'] = df[choice_col]
    df['Subject'] = df[subject_col].astype(str)

    # interaction
    df['EvStr_x_Complexity'] = df['Evidence_Strength'] * df['complexity']

    try:
        mod = smf.gee(
            "Choice ~ Evidence_Strength + complexity + EvStr_x_Complexity",
            "Subject",
            data=df,
            cov_struct=sm.cov_struct.Exchangeable(),
            family=sm.families.Binomial()
        )
        result = mod.fit()
        return result
    except Exception as e:
        print(f"GEE/GLMM error: {e}")
        return None

# psychometric function: sigmoid P(x) = 1 / (1 + exp(-k*(x - x0)))
def sigmoid(x, k, x0):
    """P(choice) = 1 / (1 + exp(-k*(x - x0)))"""
    return 1.0 / (1.0 + np.exp(-np.clip(k * (x - x0), -500, 500)))


def fit_psychometric(df, x_col='EV_diff', y_col='sequenceChoiceLeft'):
    """fit sigmoid. x0 = point of indifference, k = slope (sensitivity)."""
    x = df[x_col].values.astype(float)
    y = df[y_col].values.astype(float)
    # exclude degenerate
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 10:
        return None, None, None

    try:
        p0 = [0.1, 0.0]
        popt, pcov = curve_fit(sigmoid, x, y, p0=p0, maxfev=5000)
        k, x0 = popt
        return k, x0, lambda xx: sigmoid(np.array(xx), k, x0)
    except Exception:
        return None, None, None


def plot_psychometric_easy_hard(df, complexity_col='complexity', choice_col='sequenceChoiceLeft',
                                ev_col='EV_diff', save_path=None):
    """two curves: Easy trials vs Hard trials (by median complexity)."""
    median_complexity = df[complexity_col].median()
    easy = df[df[complexity_col] <= median_complexity]
    hard = df[df[complexity_col] > median_complexity]

    fig, ax = plt.subplots(figsize=(9, 6))
    x_plot = np.linspace(df[ev_col].min(), df[ev_col].max(), 200)

    for subset, label, color in [(easy, 'Easy trials', 'green'), (hard, 'Hard trials', 'coral')]:
        if len(subset) < 20:
            continue
        k, x0, curve = fit_psychometric(subset, x_col=ev_col, y_col=choice_col)
        if k is not None:
            ax.plot(x_plot, curve(x_plot), '-', label=f'{label}: k={k:.3f}, x0={x0:.2f}', color=color, lw=2)
        # binned points
        bins = pd.qcut(subset[ev_col], q=10, duplicates='drop')
        agg = subset.groupby(bins).agg({choice_col: 'mean', ev_col: 'mean'})
        ax.scatter(agg[ev_col], agg[choice_col], color=color, s=50, alpha=0.7, edgecolors='black')

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Evidence Strength (EV_diff)', fontsize=12)
    ax.set_ylabel('P(choice = Left)', fontsize=12)
    ax.set_title('Psychometric function: Easy vs Hard trials\nk = sensitivity, x0 = bias (point of indifference)', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig

# leaky integrator (evidence accumulation)
def leaky_integrator_trial(left_seq, right_seq, leak=1.0):
    """accumulation_t = leak * accumulation_{t-1} + (left_t - right_t). leak < 1 = forgetting."""
    acc = 0.0
    for lt, rt in zip(left_seq, right_seq):
        acc = leak * acc + (lt - rt)
    return acc


def predict_choice_leaky(left_seq, right_seq, leak):
    """choice: sign(accumulation). 1 = left, 0 = right."""
    acc = leaky_integrator_trial(left_seq, right_seq, leak)
    return 1 if acc > 0 else 0


def fit_leaky_integrator(df, cols_left, cols_right, choice_col='sequenceChoiceLeft'):
    """
    Fit leak (λ) for Leaky Integrator model.
    λ < 1: forgetting early sequence items (Leakage).
    """
    left_arr = df[cols_left].values
    right_arr = df[cols_right].values
    y_true = df[choice_col].values

    def neg_acc(leak):
        leak = np.clip(leak, 0.01, 1.5)
        preds = np.array([predict_choice_leaky(L, R, leak) for L, R in zip(left_arr, right_arr)])
        return -np.mean(preds == y_true)

    res = minimize(neg_acc, x0=[0.95], bounds=[(0.01, 1.5)], method='L-BFGS-B')
    best_leak = float(np.clip(res.x[0], 0.01, 1.5))
    acc = -res.fun
    return best_leak, acc


# cluster interpretation (LLM-expertise)
def interpret_clusters(df, labels, residuals, ev_col='EV_diff', complexity_col='complexity',
                      correct_col='is_correct', n_clusters=4):
    """
    Cluster characteristics + theoretical interpretation.
    Mapping to known biases: Recency, Primacy, Conservatism, Extreme weighting.
    """
    df = df.copy()
    df['cluster'] = labels
    df['residual'] = residuals

    interpretations = []
    for c in range(n_clusters):
        sub = df[df['cluster'] == c]
        if len(sub) < 10:
            interpretations.append({'cluster': c, 'n': len(sub), 'interpretation': 'Insufficient data'})
            continue

        mean_ev = sub[ev_col].mean()
        mean_complexity = sub[complexity_col].mean()
        mean_resid = sub['residual'].mean()
        err_rate = 1 - sub[correct_col].mean()
        # proportion of "left overestimation": residual > 0 when EV-optimal is right, etc.
        # |residual| — отклонение вероятности «лево» от EV-оптимального выбора
        avg_abs_resid = np.abs(sub['residual']).mean()

        # heuristic for interpretation
        if mean_resid > 0.1 and mean_ev > 0:
            bias_type = "Left overestimation / Anchoring to left option"
        elif mean_resid < -0.1 and mean_ev < 0:
            bias_type = "Right overestimation / Anchoring to right option"
        elif err_rate > 0.35 and mean_complexity > df[complexity_col].median():
            bias_type = "Conservatism — ignoring strong evidence at high complexity"
        elif mean_ev > 0 and mean_resid < -0.05:
            bias_type = "Evidence undervaluation / Counter-intuitive choice"
        else:
            bias_type = "Mixed pattern / Moderate deviation from EV-optimal"

        interpretations.append({
            'cluster': int(c),
            'n': int(len(sub)),
            'mean_EV_diff': float(mean_ev),
            'mean_complexity': float(mean_complexity),
            'mean_residual': float(mean_resid),
            'error_rate': float(err_rate),
            'avg_abs_residual': float(avg_abs_resid),
            'interpretation': bias_type,
        })

    return interpretations


# calibration: confidence vs accuracy (metacognitive bias)

def run_calibration_analysis(df, complexity_col='complexity', correct_col='is_correct',
                             confidence_col='confidence', n_bins=5):
    """
    Calibration: how confidence relates to accuracy across complexity.
    Metacognitive bias: at high complexity, accuracy drops but confidence remains high.
    """
    df = df.copy()
    # normalize confidence to 0–1 (original scale typically 1–6)
    conf_min, conf_max = df[confidence_col].min(), df[confidence_col].max()
    if conf_max > conf_min:
        df['confidence_norm'] = (df[confidence_col] - conf_min) / (conf_max - conf_min)
    else:
        df['confidence_norm'] = 0.5

    bins = pd.qcut(df[complexity_col], q=n_bins, duplicates='drop')
    agg = df.groupby(bins).agg({
        correct_col: ['mean', 'count'],
        'confidence_norm': 'mean',
        complexity_col: 'mean',
    }).reset_index()
    agg.columns = ['bin', 'accuracy', 'n', 'confidence', 'complexity_mean']
    agg = agg.drop(columns=['bin'], errors='ignore')

    # correlations
    r_acc_comp = stats.pearsonr(df[complexity_col], df[correct_col])[0]
    r_conf_comp = stats.pearsonr(df[complexity_col], df['confidence_norm'])[0]
    r_conf_acc = stats.pearsonr(df['confidence_norm'], df[correct_col])[0]

    # metacognitive bias: confidence–accuracy gap by complexity
    low_complexity = df[complexity_col] <= df[complexity_col].median()
    high_complexity = ~low_complexity
    gap_low = df.loc[low_complexity, 'confidence_norm'].mean() - df.loc[low_complexity, correct_col].mean()
    gap_high = df.loc[high_complexity, 'confidence_norm'].mean() - df.loc[high_complexity, correct_col].mean()

    return {
        'by_bin': agg.to_dict(orient='records'),
        'corr_complexity_accuracy': float(r_acc_comp),
        'corr_complexity_confidence': float(r_conf_comp),
        'corr_confidence_accuracy': float(r_conf_acc),
        'confidence_accuracy_gap_low_complexity': float(gap_low),
        'confidence_accuracy_gap_high_complexity': float(gap_high),
        'metacognitive_bias': float(gap_high - gap_low),  # > 0: more overconfidence at high complexity
    }


def paired_ttest_calibration_gap(df, subject_col, complexity_col='complexity', correct_col='is_correct',
                                 confidence_col='confidence', trials_per_subject=None):
    """
    Paired Samples T-test: (Confidence - Accuracy) Easy vs Hard trials per subject.
    Tests whether the gap differs between easy and hard trials within subjects.
    """
    df = df.copy()
    if subject_col not in df.columns:
        if trials_per_subject is None:
            trials_per_subject = max(1, len(df) // 33)
        df[subject_col] = np.arange(len(df)) // trials_per_subject
        # note: subject ID inferred from row order (trials assumed consecutive per subject)
    conf_min, conf_max = df[confidence_col].min(), df[confidence_col].max()
    if conf_max > conf_min:
        df['confidence_norm'] = (df[confidence_col] - conf_min) / (conf_max - conf_min)
    else:
        df['confidence_norm'] = 0.5
    df['gap'] = df['confidence_norm'] - df[correct_col].astype(float)
    median_comp = df[complexity_col].median()
    easy = df[df[complexity_col] <= median_comp]
    hard = df[df[complexity_col] > median_comp]
    gap_easy = easy.groupby(subject_col)['gap'].mean()
    gap_hard = hard.groupby(subject_col)['gap'].mean()
    subj_common = gap_easy.index.intersection(gap_hard.index)
    if len(subj_common) < 3:
        return None
    g_e = gap_easy.loc[subj_common].values
    g_h = gap_hard.loc[subj_common].values
    t_stat, p_val = stats.ttest_rel(g_h, g_e)
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'n_subjects': int(len(subj_common)),
        'mean_gap_easy': float(np.mean(g_e)),
        'mean_gap_hard': float(np.mean(g_h)),
        'mean_difference': float(np.mean(g_h) - np.mean(g_e)),
        'significant': bool(p_val < 0.05),
    }


def plot_calibration_by_complexity(df, complexity_col='complexity', correct_col='is_correct',
                                   confidence_col='confidence', n_bins=5, save_path=None):
    """
    Plot: at high complexity, accuracy drops but confidence stays high (metacognitive bias).
    """
    df = df.copy()
    conf_min, conf_max = df[confidence_col].min(), df[confidence_col].max()
    if conf_max > conf_min:
        df['confidence_norm'] = (df[confidence_col] - conf_min) / (conf_max - conf_min)
    else:
        df['confidence_norm'] = 0.5

    bins = pd.qcut(df[complexity_col], q=n_bins, duplicates='drop')
    agg = df.groupby(bins).agg({
        correct_col: 'mean',
        'confidence_norm': 'mean',
        complexity_col: 'mean',
    })
    agg.columns = ['accuracy', 'confidence', 'complexity_mean']
    agg = agg.reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(9, 6))
    x = np.arange(len(agg))
    width = 0.35

    bars1 = ax1.bar(x - width/2, agg['accuracy'], width, label='Accuracy', color='steelblue', alpha=0.8)
    ax1.set_ylabel('Accuracy (proportion correct)', color='steelblue', fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis='y', labelcolor='steelblue')

    ax2 = ax1.twinx()
    bars2 = ax2.plot(x, agg['confidence'], 'o-', color='darkorange', linewidth=2, markersize=10,
                     label='Confidence (normalized)')
    ax2.set_ylabel('Confidence (0–1)', color='darkorange', fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis='y', labelcolor='darkorange')

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{v:.1f}' for v in agg['complexity_mean']], rotation=30)
    ax1.set_xlabel('Complexity (bin mean)', fontsize=12)
    ax1.set_title('Calibration: Confidence vs Accuracy by Complexity\n'
                  'Metacognitive bias: accuracy drops at high complexity, confidence stays high',
                  fontsize=13)
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


# main

def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(OUT_ADVANCED, exist_ok=True)
    os.makedirs(FIG_LAB, exist_ok=True)
    os.makedirs(FIG_CONS, exist_ok=True)

    # import main pipeline for data
    from .evidence_integration_analysis import (
        load_data, preprocess_data_lab, preprocess_data_cons,
        build_features_lab, run_random_forest,
        compute_residuals, cluster_residuals,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df_lab, df_cons, _ = load_data()
    df_lab = df_lab.dropna(how='all', axis=1).dropna()
    df_lab, cols_a, cols_b = preprocess_data_lab(df_lab)
    df_lab['economic_loss'] = np.maximum(df_lab['EV_left'], df_lab['EV_right']) - np.where(
        df_lab['sequenceChoiceLeft'] == 1, df_lab['EV_left'], df_lab['EV_right']
    )

    X_lab = build_features_lab(df_lab, cols_a, cols_b)
    y_h = df_lab['sequenceChoiceLeft']
    train_idx, test_idx = train_test_split(
        np.arange(len(X_lab)), test_size=0.2, stratify=y_h, random_state=42
    )
    rf_res = run_random_forest(X_lab, y_h, train_idx, test_idx, cv=5)

    scaler = rf_res['scaler']
    residuals = compute_residuals(rf_res['rf_human'], X_lab, scaler, df_lab['normative_choice'])
    X_s = scaler.transform(X_lab)
    labels, _ = cluster_residuals(residuals, X_s, n_clusters=4)

    results = {}

    #  glmm
    print("\nGLMM (GEE)")
    glmm = run_glmm(df_lab, choice_col='sequenceChoiceLeft', subject_col='subject')
    if glmm is not None:
        print(glmm.summary())
        results['glmm'] = {
            'params': {k: float(v) for k, v in glmm.params.items()},
            'pvalues': {k: float(v) for k, v in glmm.pvalues.items()},
        }
        inter_p = glmm.pvalues.get('EvStr_x_Complexity', 1.0)
        print(f"\nInteraction Evidence_Strength * Complexity: p = {inter_p:.4f}")
        if inter_p < 0.05:
            print("    Significant: complexity changes how evidence is used.")


    # psychometric function
    print("\nPsychometric function (Easy vs Hard)")
    plot_psychometric_easy_hard(
        df_lab, save_path=os.path.join(FIG_LAB, 'psychometric_easy_hard.png')
    )
    print(f"Saved: {os.path.join(FIG_LAB, 'psychometric_easy_hard.png')}")

    median_c = df_lab['complexity'].median()
    easy = df_lab[df_lab['complexity'] <= median_c]
    hard = df_lab[df_lab['complexity'] > median_c]
    for name, sub in [('Easy', easy), ('Hard', hard)]:
        k, x0, _ = fit_psychometric(sub, 'EV_diff', 'sequenceChoiceLeft')
        if k is not None:
            print(f"  {name}: k (sensitivity) = {k:.4f}, x0 (bias) = {x0:.3f}")
            results[f'psychometric_{name.lower()}'] = {'k': float(k), 'x0': float(x0)}

    # leaky integrator
    print("\nLeaky Integrator")
    leak, acc = fit_leaky_integrator(df_lab, cols_a, cols_b, 'sequenceChoiceLeft')
    print(f"  Optimal λ (leak) = {leak:.4f}")
    print(f"  Accuracy on data: {acc:.4f}")
    if leak < 0.95:
        print("  Interpretation: λ < 1 — forgetting early sequence (Leakage/Recency).")
    results['leaky_integrator'] = {'leak': float(leak), 'accuracy': float(acc)}

    # cluster interpretation
    print("\nCluster interpretation")
    interp = interpret_clusters(
        df_lab, labels, residuals,
        ev_col='EV_diff', complexity_col='complexity', correct_col='is_correct', n_clusters=4
    )
    results['cluster_interpretations'] = interp
    for item in interp:
        print(f"  Cluster {item['cluster']} (n={item['n']}): {item['interpretation']}")

    # calibration (dataConsA: confidence vs accuracy, metacognitive bias)
    print("\nCalibration: Confidence vs Accuracy (dataConsA)")
    df_cons = df_cons.dropna(how='all', axis=1).dropna(subset=['confidence'])
    df_cons, _, _ = preprocess_data_cons(df_cons)

    if 'confidence' in df_cons.columns and len(df_cons) > 0:
        cal = run_calibration_analysis(
            df_cons, complexity_col='complexity', correct_col='is_correct',
            confidence_col='confidence', n_bins=5
        )
        results['calibration'] = cal

        print(f"  Corr(complexity, accuracy):    {cal['corr_complexity_accuracy']:.4f}")
        print(f"  Corr(complexity, confidence):  {cal['corr_complexity_confidence']:.4f}")
        print(f"  Corr(confidence, accuracy):    {cal['corr_confidence_accuracy']:.4f}")
        print(f"  Confidence–accuracy gap (low complexity):  {cal['confidence_accuracy_gap_low_complexity']:.4f}")
        print(f"  Confidence–accuracy gap (high complexity): {cal['confidence_accuracy_gap_high_complexity']:.4f}")
        print(f"  Metacognitive bias (gap_high - gap_low):   {cal['metacognitive_bias']:.4f}")

        ttest_res = paired_ttest_calibration_gap(
            df_cons, subject_col='subject', complexity_col='complexity',
            correct_col='is_correct', confidence_col='confidence',
            trials_per_subject=200
        )
        if ttest_res is not None:
            cal['paired_ttest'] = ttest_res
            print(f"  Paired t-test (gap Hard - gap Easy): t={ttest_res['t_statistic']:.3f}, p={ttest_res['p_value']:.4f}, n={ttest_res['n_subjects']}")
            if ttest_res['significant']:
                print("  >>> Gap difference is statistically significant (p < 0.05): metacognitive bias is not noise.")
            else:
                print("  >>> Gap difference is not significant (p >= 0.05): could be noise.")

        if cal['metacognitive_bias'] > 0.02:
            print("  Metacognitive bias present: at high complexity, confidence stays high while accuracy drops.")

        plot_calibration_by_complexity(
            df_cons, complexity_col='complexity', correct_col='is_correct',
            confidence_col='confidence', n_bins=5,
            save_path=os.path.join(FIG_CONS, 'calibration_confidence_vs_accuracy.png')
        )

    out_path = ADVANCED_ANALYSIS_JSON
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
