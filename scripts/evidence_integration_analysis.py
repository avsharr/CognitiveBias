"""End-to-end pipeline: preprocessing → models → bias / plots. Implementation is split into
``scripts.preprocessing`` (data) and ``scripts.models`` (training).
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from .paths import (
    FIG_COMBINED,
    FIG_CONS,
    FIG_FRAMING,
    FIG_LAB,
    PRE_COMBINED,
    PRE_CONS,
    PRE_FRAMING,
    PRE_LAB,
    RES_COMBINED,
    RES_CONS,
    RES_FRAMING,
    RES_LAB,
    SUBDIRS_TO_CREATE,
    SUMMARY_JSON,
)
from .preprocessing.data_pipeline import (
    PAIR_FEATURE_COLS,
    build_features_cons,
    build_features_framing,
    build_features_lab,
    build_features_unified,
    load_data,
    make_unified_block,
    preprocess_data_cons,
    preprocess_data_framing,
    preprocess_data_lab,
    save_preprocessed_csv,
    unified_sequence_matrix,
)
from .models.training import HP, _run_bundle, _summarize_bundle, run_random_forest

# Re-export for ``advanced_analysis``
from .preprocessing.data_pipeline import build_features_lab, load_data, preprocess_data_cons, preprocess_data_lab  # noqa: F401


def _relative_artifact_paths(dataset_name: str) -> dict:
    """Stable relative paths from repo root (for docs / metrics sidecar)."""
    if dataset_name == 'combined':
        outputs_sub = 'outputs/combined/'
    else:
        outputs_sub = f'outputs/separate_analysis/{dataset_name}/'
    return {
        'preprocessing': f'preprocessing/{dataset_name}/',
        'models': f'models/{dataset_name}/',
        'figures': f'figures/{dataset_name}/',
        'outputs': outputs_sub,
    }


def save_dataset_metrics_json(results_subdir: str, dataset_name: str, models_block: dict, extra: dict | None = None):
    """Write ``outputs/separate_analysis/<dataset>/metrics.json`` or ``outputs/combined/metrics.json``."""
    os.makedirs(results_subdir, exist_ok=True)
    payload = {
        'dataset': dataset_name,
        'hyperparameters': HP,
        'artifact_locations': _relative_artifact_paths(dataset_name),
        'models': models_block,
    }
    if extra:
        payload['combined_meta'] = extra
    out_path = os.path.join(results_subdir, 'metrics.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f'Saved: {out_path}', flush=True)


def compute_residuals(rf_model, X, scaler, ev_optimal_left: pd.Series):
    """Остаток: P(модель предсказывает «лево») − индикатор EV-оптимального левого (0/1). Без отдельной «нормативной модели»."""
    X_s = scaler.transform(X)
    proba = rf_model.predict_proba(X_s)[:, 1]
    return proba - ev_optimal_left.astype(float).values


def cluster_residuals(residuals, X_scaled, n_clusters=4):
    feat = np.column_stack([residuals.reshape(-1, 1), X_scaled[:, -3:]])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(feat)
    return labels, kmeans


def compute_economic_loss(df, ev_left_col='EV_left', ev_right_col='EV_right', choice_col='sequenceChoiceLeft'):
    if choice_col not in df.columns:
        choice_col = 'responseLeft'
    ev_chosen = np.where(df[choice_col] == 1, df[ev_left_col], df[ev_right_col])
    ev_optimal = np.maximum(df[ev_left_col], df[ev_right_col])
    return ev_optimal - ev_chosen


def plot_psychometric_curve(df, complexity_col='complexity', correct_col='is_correct', ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    bins = pd.qcut(df[complexity_col], q=10, duplicates='drop')
    curve = df.groupby(bins).agg({correct_col: 'mean', complexity_col: 'mean'})
    ax.plot(curve[complexity_col], curve[correct_col], 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Complexity', fontsize=12)
    ax.set_ylabel('Probability of correct response', fontsize=12)
    ax.set_title('Psychometric curve', fontsize=14)
    ax.grid(True, alpha=0.3)
    return ax


def plot_tsne_clusters(residuals, X_scaled, labels, ax=None, max_samples=2000):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    feat = np.column_stack([residuals.reshape(-1, 1), X_scaled])
    if feat.shape[1] > 50:
        feat = feat[:, :50]
    n = min(len(feat), max_samples)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(feat), n, replace=False)
    feat_s, labels_s = feat[idx], labels[idx]
    perplexity = min(30, n // 4)
    tsne = TSNE(n_components=2, random_state=42, perplexity=max(5, perplexity))
    emb = tsne.fit_transform(feat_s)
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=labels_s, cmap='viridis', alpha=0.6)
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('Bias clusters (t-SNE)', fontsize=14)
    plt.colorbar(scatter, ax=ax, label='Cluster')
    return ax


def plot_economic_loss(df, complexity_col='complexity', loss_col='economic_loss', ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    bins = pd.qcut(df[complexity_col], q=10, duplicates='drop')
    agg = df.groupby(bins).agg({loss_col: 'mean', complexity_col: 'mean'})
    ax.bar(range(len(agg)), agg[loss_col], color='coral', alpha=0.8)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels([f'{v:.1f}' for v in agg[complexity_col]], rotation=45)
    ax.set_xlabel('Complexity (bin mean)', fontsize=12)
    ax.set_ylabel('Lost utility (mean EV)', fontsize=12)
    ax.set_title('Economic effect: losses as complexity increases', fontsize=14)
    return ax


def main():
    print('Start pipeline (LG → RF → XGB; then TensorFlow: LSTM/RNN)...', flush=True)
    for d in SUBDIRS_TO_CREATE:
        os.makedirs(d, exist_ok=True)

    print('EVIDENCE INTEGRATION & BIAS DISCOVERY', flush=True)
    print('Code: scripts/preprocessing (data) · scripts/models (training)', flush=True)
    print('Artifacts: preprocessing/*.csv · figures/*/*.png · outputs/**/*.csv + JSON', flush=True)

    df_lab_raw, df_cons_raw, df_framing_raw = load_data()
    df_lab_raw = df_lab_raw.dropna(how='all', axis=1).dropna()
    df_cons_raw = df_cons_raw.dropna(how='all', axis=1).dropna()
    df_framing_raw = df_framing_raw.dropna(how='all', axis=1).dropna()

    summary = {
        'hyperparameters': HP,
        'data_lab': {},
        'data_cons': {},
        'data_framing': {},
        'combined': {},
    }

    print('\n=== data_lab ===')
    df_lab, cols_a, cols_b = preprocess_data_lab(df_lab_raw)
    X_lab = build_features_lab(df_lab, cols_a, cols_b)
    y_human_lab = df_lab['sequenceChoiceLeft']
    seq_lab = df_lab[cols_a + cols_b].values

    bundle_lab = _run_bundle(
        X_lab, y_human_lab, seq_lab,
        cv_chart_path=os.path.join(FIG_LAB, 'cv_bar_chart.png'),
        dataset_label='data_lab',
        rf_corr_csv_path=os.path.join(RES_LAB, 'rf_features_correlation.csv'),
        rf_corr_heatmap_path=os.path.join(FIG_LAB, 'rf_features_correlation_heatmap.png'),
    )
    summary['data_lab'] = _summarize_bundle(bundle_lab)
    save_dataset_metrics_json(RES_LAB, 'data_lab', summary['data_lab'])
    rf_lab = bundle_lab['rf']

    X_full_s = rf_lab['scaler'].transform(X_lab)
    residuals = compute_residuals(rf_lab['rf_human'], X_lab, rf_lab['scaler'], df_lab['normative_choice'])
    labels, _ = cluster_residuals(residuals, X_full_s)
    df_lab['economic_loss'] = compute_economic_loss(df_lab, choice_col='sequenceChoiceLeft')

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_psychometric_curve(df_lab, ax=axes[0, 0])
    plot_economic_loss(df_lab, ax=axes[0, 1])
    plot_tsne_clusters(residuals, X_full_s[:, :min(20, X_full_s.shape[1])], labels, ax=axes[1, 0])
    fi_df = pd.DataFrame(rf_lab['feature_importance'].items(), columns=['feature', 'importance'])
    fi_df = fi_df.sort_values('importance', ascending=True).tail(15)
    fi_df.plot(x='feature', y='importance', kind='barh', ax=axes[1, 1], legend=False)
    axes[1, 1].set_title('Feature Importance (Random Forest)')
    axes[1, 1].set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_LAB, 'data_lab_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    save_preprocessed_csv(df_lab, PRE_LAB, 'preprocessed.csv')

    print('\n=== data_cons ===')
    df_cons, cols_l, cols_r = preprocess_data_cons(df_cons_raw)
    save_preprocessed_csv(df_cons, PRE_CONS, 'preprocessed.csv')
    X_cons = build_features_cons(df_cons, cols_l, cols_r)
    bundle_cons = _run_bundle(
        X_cons, df_cons['responseLeft'], df_cons[cols_l + cols_r].values,
        cv_chart_path=os.path.join(FIG_CONS, 'cv_bar_chart.png'),
        dataset_label='data_cons',
        rf_corr_csv_path=os.path.join(RES_CONS, 'rf_features_correlation.csv'),
        rf_corr_heatmap_path=os.path.join(FIG_CONS, 'rf_features_correlation_heatmap.png'),
    )
    summary['data_cons'] = _summarize_bundle(bundle_cons)
    save_dataset_metrics_json(RES_CONS, 'data_cons', summary['data_cons'])

    print('\n=== data_framing ===')
    df_framing, left_cols, right_cols = preprocess_data_framing(df_framing_raw)
    save_preprocessed_csv(df_framing, PRE_FRAMING, 'preprocessed.csv')
    X_fr = build_features_framing(df_framing, left_cols, right_cols)
    bundle_fr = _run_bundle(
        X_fr, df_framing['responseLeft'],
        df_framing[left_cols + right_cols].values,
        cv_chart_path=os.path.join(FIG_FRAMING, 'cv_bar_chart.png'),
        dataset_label='data_framing',
        rf_corr_csv_path=os.path.join(RES_FRAMING, 'rf_features_correlation.csv'),
        rf_corr_heatmap_path=os.path.join(FIG_FRAMING, 'rf_features_correlation_heatmap.png'),
    )
    summary['data_framing'] = _summarize_bundle(bundle_fr)
    save_dataset_metrics_json(RES_FRAMING, 'data_framing', summary['data_framing'])

    print('\n=== combined ===')
    unified_lab = make_unified_block(df_lab, cols_a, cols_b, 'sequenceChoiceLeft', 'lab')
    unified_cons = make_unified_block(df_cons, cols_l, cols_r, 'responseLeft', 'cons')
    unified_fram = make_unified_block(df_framing, left_cols, right_cols, 'responseLeft', 'framing')
    df_uni = pd.concat([unified_lab, unified_cons, unified_fram], ignore_index=True)
    # subject у cons в исходном CSV нет — колонка <NA>; не считаем это пропуском для dropna
    _dropna_subset = [c for c in df_uni.columns if c != 'subject']
    df_uni = df_uni.dropna(subset=_dropna_subset)

    export_cols = [
        'subject',
        'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8',
        'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8',
        'human_choice', 'normative_choice',
        'EV_left', 'EV_right', 'EV_diff', 'complexity',
        'is_correct', 'dataset_source',
    ] + PAIR_FEATURE_COLS
    _uni_path_pre = os.path.join(PRE_COMBINED, 'unified_dataset.csv')
    df_uni[export_cols].to_csv(_uni_path_pre, index=False)

    X_comb = build_features_unified(df_uni)
    bundle_comb = _run_bundle(
        X_comb, df_uni['human_choice'], unified_sequence_matrix(df_uni),
        cv_chart_path=os.path.join(FIG_COMBINED, 'cv_bar_chart.png'),
        dataset_label='combined',
        rf_corr_csv_path=os.path.join(RES_COMBINED, 'rf_features_correlation.csv'),
        rf_corr_heatmap_path=os.path.join(FIG_COMBINED, 'rf_features_correlation_heatmap.png'),
    )
    comb_models_only = _summarize_bundle(bundle_comb)
    summary['combined'] = {
        **comb_models_only,
        'random_forest_combined_note': (
            'Random Forest (human choice only) on unified features: sequences + EV_diff + complexity + pair_* + source one-hot.'
        ),
        'n_rows': int(len(df_uni)),
        'rows_by_source': {
            str(k): int(v) for k, v in df_uni['dataset_source'].value_counts().to_dict().items()
        },
        'unified_stats_csv_relative': 'datasets/unified_dataset.csv',
        'unified_stats_note': 'Статистики последовательностей: python -m scripts.merge_data',
        'unified_csv_preprocessing_relative': 'preprocessing/combined/unified_dataset.csv',
    }

    save_dataset_metrics_json(
        RES_COMBINED,
        'combined',
        comb_models_only,
        extra={
            'random_forest_combined_note': summary['combined']['random_forest_combined_note'],
            'n_rows': summary['combined']['n_rows'],
            'rows_by_source': summary['combined']['rows_by_source'],
            'unified_stats_csv_relative': summary['combined']['unified_stats_csv_relative'],
            'unified_stats_note': summary['combined']['unified_stats_note'],
            'unified_csv_preprocessing_relative': summary['combined']['unified_csv_preprocessing_relative'],
        },
    )

    with open(SUMMARY_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\nSaved (all datasets): {SUMMARY_JSON}')


if __name__ == '__main__':
    main()
