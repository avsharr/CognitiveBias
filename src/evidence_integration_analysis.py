import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import os
import json
import itertools

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# hyperparameters
HP = {
    'random_state': 10,
    'test_size': 0.2,
    'cv_folds': 5,
    'ci_alpha': 0.95,  # 95% confidence interval
}

# grid search param grids (optimal search per dataset)
RF_PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
}

LSTM_PARAM_GRID = {
    'units': [32, 64],
    'dropout': [0.2, 0.3],
    'epochs': [25, 30],
    'batch_size': [32, 64],
}

RNN_PARAM_GRID = {
    'units': [32, 64],
    'dropout': [0.2, 0.3],
    'epochs': [25, 30],
    'batch_size': [32, 64],
}

# load and prepare data
def load_data():
    """load all three datasets."""
    base = os.path.dirname(os.path.abspath(__file__))
    
    df_lab = pd.read_csv(os.path.join(base, 'data_lab.csv'))
    df_cons = pd.read_csv(os.path.join(base, 'dataConsA.csv'))
    df_framing = pd.read_csv(os.path.join(base, 'dataHighFramingA.csv'))
    
    return df_lab, df_cons, df_framing


def preprocess_data_lab(df):
    """normative for data_lab.csv"""
    cols_a = [c for c in df.columns if c.startswith('sequenceA')]
    cols_b = [c for c in df.columns if c.startswith('sequenceB')]
    
    ev_a = df[cols_a].mean(axis=1)
    ev_b = df[cols_b].mean(axis=1)
    
    df = df.copy()
    df['EV_left'] = ev_a
    df['EV_right'] = ev_b
    df['EV_diff'] = ev_a - ev_b
    df['normative_choice'] = (ev_a > ev_b).astype(int)  
    df['is_correct'] = (df['sequenceChoiceLeft'] == df['normative_choice']).astype(int)
    
    # complexity
    all_vals = df[cols_a + cols_b]
    df['complexity_diff'] = -np.abs(df['EV_diff'])  # smaller diff = harder
    df['complexity_sd'] = all_vals.std(axis=1)
    df['complexity'] = df['complexity_diff'] + df['complexity_sd']  # combined
    
    return df, cols_a, cols_b


def preprocess_data_cons(df):
    """preprocess for dataConsA.csv"""
    cols_l = [c for c in df.columns if c.startswith('sequenceLeft')]
    cols_r = [c for c in df.columns if c.startswith('sequenceRight')]
    
    ev_l = df[cols_l].mean(axis=1)
    ev_r = df[cols_r].mean(axis=1)
    
    df = df.copy()
    df['EV_left'] = ev_l
    df['EV_right'] = ev_r
    df['EV_diff'] = ev_l - ev_r
    df['normative_choice'] = (ev_l > ev_r).astype(int)
    df['is_correct'] = (df['responseLeft'] == df['normative_choice']).astype(int)
    
    all_vals = df[cols_l + cols_r]
    df['complexity_diff'] = -np.abs(df['EV_diff'])
    df['complexity_sd'] = all_vals.std(axis=1)
    df['complexity'] = df['complexity_diff'] + df['complexity_sd']
    
    return df, cols_l, cols_r


def preprocess_data_framing(df):
    """preprocess for dataHighFramingA.csv. sequences_1..8=left, sequences_9..16=right."""
    seq_cols = [c for c in df.columns if 'sequences' in c.lower()]
    nums = []
    for c in seq_cols:
        n = int(c.split('_')[-1].strip())
        nums.append((n, c))
        
    nums.sort(key=lambda x: x[0])
    seq_cols = [x[1] for x in nums]
    left_cols = seq_cols[:8]
    right_cols = seq_cols[8:16] if len(seq_cols) >= 16 else []
    
    ev_l = df[left_cols].mean(axis=1)
    ev_r = df[right_cols].mean(axis=1)
    
    df = df.copy()
    df['EV_left'] = ev_l
    df['EV_right'] = ev_r
    df['EV_diff'] = ev_l - ev_r
    df['normative_choice'] = (ev_l > ev_r).astype(int)
    df['is_correct'] = (df['responseLeft'] == df['normative_choice']).astype(int)
    
    all_vals = df[left_cols + right_cols]
    df['complexity_diff'] = -np.abs(df['EV_diff'])
    df['complexity_sd'] = all_vals.std(axis=1)
    df['complexity'] = df['complexity_diff'] + df['complexity_sd']
    
    return df, left_cols, right_cols


# RF and LSTM models

def build_features_lab(df, cols_a, cols_b):
    """features for data_lab: all 16 numbers + EV + complexity."""
    feat_cols = cols_a + cols_b + ['EV_diff', 'complexity', 'isDifferentMeanSameVariance']
    X = df[feat_cols].copy()
    X = X.fillna(X.median())
    return X


def build_features_cons(df, cols_l, cols_r):
    feat_cols = cols_l + cols_r + ['EV_diff', 'complexity']
    if 'flip' in df.columns:
        feat_cols.append('flip')
    X = df[[c for c in feat_cols if c in df.columns]].copy()
    X = X.fillna(X.median())
    return X


def build_features_framing(df, left_cols, right_cols):
    feat_cols = left_cols + right_cols + ['EV_diff', 'complexity']
    X = df[[c for c in feat_cols if c in df.columns]].copy()
    X = X.fillna(X.median())
    return X


def confidence_interval(scores, alpha=0.95):
    """95% confidence interval for mean of CV scores."""
    n = len(scores)
    mean = np.mean(scores)
    std = np.std(scores, ddof=1) if n > 1 else 0
    # t-distribution for small n
    t_val = stats.t.ppf((1 + alpha) / 2, n - 1) if n > 1 else 0
    margin = t_val * (std / np.sqrt(n)) if n > 1 else 0
    return mean, margin, mean - margin, mean + margin


def run_random_forest(X, y_human, y_normative, train_idx, test_idx, cv=5):
    """random forest: GridSearchCV for hyperparameters, shared test set, human and normative."""
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_h_train, y_h_test = y_human.iloc[train_idx], y_human.iloc[test_idx]
    y_n_train, y_n_test = y_normative.iloc[train_idx], y_normative.iloc[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=HP['random_state'])
    base_rf = RandomForestClassifier(random_state=HP['random_state'])

    # RF human: GridSearchCV
    gs_human = GridSearchCV(base_rf, RF_PARAM_GRID, cv=cv_split, scoring='accuracy', n_jobs=1, verbose=0)
    gs_human.fit(X_train_s, y_h_train)
    rf_human = gs_human.best_estimator_
    cv_scores_human = np.array([gs_human.cv_results_[f'split{i}_test_score'][gs_human.best_index_] for i in range(cv)])
    acc_test_human = accuracy_score(y_h_test, rf_human.predict(X_test_s))
    ci_human = confidence_interval(cv_scores_human, HP['ci_alpha'])

    # RF normative: GridSearchCV
    gs_norm = GridSearchCV(RandomForestClassifier(random_state=HP['random_state']),
                          RF_PARAM_GRID, cv=cv_split, scoring='accuracy', n_jobs=1, verbose=0)
    gs_norm.fit(X_train_s, y_n_train)
    rf_norm = gs_norm.best_estimator_
    cv_scores_norm = np.array([gs_norm.cv_results_[f'split{i}_test_score'][gs_norm.best_index_] for i in range(cv)])
    acc_test_norm = accuracy_score(y_n_test, rf_norm.predict(X_test_s))
    ci_norm = confidence_interval(cv_scores_norm, HP['ci_alpha'])

    return {
        'rf_human': rf_human,
        'rf_normative': rf_norm,
        'scaler': scaler,
        'acc_test_human': acc_test_human,
        'acc_test_norm': acc_test_norm,
        'cv_scores_human': cv_scores_human,
        'cv_scores_norm': cv_scores_norm,
        'ci_human': ci_human,
        'ci_norm': ci_norm,
        'feature_importance': dict(zip(X.columns, rf_human.feature_importances_)),
        'feature_names': list(X.columns),
        'best_params_human': gs_human.best_params_,
        'best_params_normative': gs_norm.best_params_,
    }


def _seq_to_3d(X_seq):
    """convert sequence (n, 16) to (n, 8, 2)."""
    if X_seq.shape[1] != 16:
        return None
    return np.stack([
        np.column_stack([X_seq[:, i], X_seq[:, i + 8]]) for i in range(8)
    ], axis=1)


def _grid_search_lstm_rnn(X_train, y_train, X_test, y_test, param_grid, model_type='lstm'):
    """grid search for LSTM or RNN; returns best model, acc_test, best_params."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout
    from tensorflow.keras.utils import to_categorical

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    best_val_acc = -1
    best_model = None
    best_params = None

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        tf.keras.backend.clear_session()
        layer = LSTM if model_type == 'lstm' else SimpleRNN
        model = Sequential([
            layer(params['units'], input_shape=(8, 2), return_sequences=False),
            Dropout(params['dropout']),
            Dense(16, activation='relu'),
            Dense(2, activation='softmax'),
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'],
                         validation_split=0.2, verbose=0)
        val_acc = max(hist.history['val_accuracy'])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            best_params = params.copy()

    if best_model is None:
        return None, 0.0, {}
    pred = best_model.predict(X_test, verbose=0)
    acc_test = (pred.argmax(axis=1) == y_test.argmax(axis=1)).mean()
    return best_model, acc_test, best_params


def run_lstm(X_seq, y, train_idx, test_idx):
    """LSTM: grid search over hyperparameters, same test set."""
    X_3d = _seq_to_3d(X_seq)
    if X_3d is None:
        return None

    X_train = X_3d[train_idx]
    X_test = X_3d[test_idx]
    y_train = to_categorical(y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx], 2)
    y_test_cat = to_categorical(y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx], 2)

    model, acc_test, best_params = _grid_search_lstm_rnn(X_train, y_train, X_test, y_test_cat, LSTM_PARAM_GRID, 'lstm')
    if model is None:
        return None
    return {'model': model, 'acc_test': acc_test, 'best_params': best_params}


def run_rnn(X_seq, y, train_idx, test_idx):
    """simple RNN: grid search over hyperparameters, same test set."""
    X_3d = _seq_to_3d(X_seq)
    if X_3d is None:
        return None

    X_train = X_3d[train_idx]
    X_test = X_3d[test_idx]
    y_train = to_categorical(y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx], 2)
    y_test_cat = to_categorical(y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx], 2)

    model, acc_test, best_params = _grid_search_lstm_rnn(X_train, y_train, X_test, y_test_cat, RNN_PARAM_GRID, 'rnn')
    if model is None:
        return None
    return {'model': model, 'acc_test': acc_test, 'best_params': best_params}

# step 4: bias discovery
def compute_residuals(rf_model, X, y_human, y_normative, scaler):
    """residuals = model prediction (probability) - normative response."""
    X_s = scaler.transform(X)
    proba = rf_model.predict_proba(X_s)[:, 1]  # P(choose left)
    residuals = proba - y_normative.astype(float).values
    return residuals


def cluster_residuals(residuals, X_scaled, n_clusters=4):
    """cluster residuals to identify error patterns."""
    # clustering features: residuals + complexity etc.
    feat = np.column_stack([residuals.reshape(-1, 1), X_scaled[:, -3:]])  # residuals + last 3 features
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(feat)
    return labels, kmeans

# economic effect

def compute_economic_loss(df, ev_left_col='EV_left', ev_right_col='EV_right', choice_col='sequenceChoiceLeft'):
    """lost utility: EV chosen - EV optimal."""
    if choice_col not in df.columns:
        choice_col = 'responseLeft'
    
    ev_chosen = np.where(df[choice_col] == 1, df[ev_left_col], df[ev_right_col])
    ev_optimal = np.maximum(df[ev_left_col], df[ev_right_col])
    loss = ev_optimal - ev_chosen
    return loss

# visualizations

def plot_psychometric_curve(df, complexity_col='complexity', correct_col='is_correct', ax=None):
    """psychometric curve: probability of correct response vs complexity."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    bins = pd.qcut(df[complexity_col], q=10, duplicates='drop')
    curve = df.groupby(bins).agg({correct_col: 'mean', complexity_col: 'mean'})
    ax.plot(curve[complexity_col], curve[correct_col], 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Complexity', fontsize=12)
    ax.set_ylabel('Probability of correct response', fontsize=12)
    ax.set_title('Psychometric curve', fontsize=14)
    ax.grid(True, alpha=0.3)
    return ax


def plot_tsne_clusters(residuals, X_scaled, labels, ax=None, max_samples=2000):
    """t-SNE visualization of bias clusters."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    feat = np.column_stack([residuals.reshape(-1, 1), X_scaled])
    if feat.shape[1] > 50:
        feat = feat[:, :50]
    
    # subsample for t-SNE speed
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
    """plot: complexity vs lost utility."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    bins = pd.qcut(df[complexity_col], q=10, duplicates='drop')
    agg = df.groupby(bins).agg({loss_col: 'mean', complexity_col: 'mean'})
    ax.bar(range(len(agg)), agg[loss_col], color='coral', alpha=0.8)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels([f'{v:.1f}' for v in agg[complexity_col]], rotation=45)
    ax.set_xlabel('Complexity (bin mean)', fontsize=12)
    ax.set_ylabel('Lost utility (mean EV)', fontsize=12)
    ax.set_title('Economic effect: losses as complexity increases', fontsize=14)
    return ax


def plot_cv_bar_chart(cv_scores_human, cv_scores_norm, ci_human, ci_norm, save_path=None):
    """bar chart: cross-validation scores by fold + confidence intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # left panel: scores by fold
    folds = np.arange(1, len(cv_scores_human) + 1)
    x = np.arange(len(folds))
    w = 0.35
    axes[0].bar(x - w/2, cv_scores_human, w, label='RF (human)', color='steelblue', alpha=0.8)
    axes[0].bar(x + w/2, cv_scores_norm, w, label='RF (normative)', color='darkorange', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'Fold {i}' for i in folds])
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Cross-Validation: Accuracy by fold')
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)

    # right panel: mean ± CI
    mean_h, margin_h = ci_human[0], ci_human[1]
    mean_n, margin_n = ci_norm[0], ci_norm[1]
    bars = axes[1].bar(['RF (human)', 'RF (normative)'], [mean_h, mean_n],
                       color=['steelblue', 'darkorange'], alpha=0.8)
    axes[1].errorbar([0, 1], [mean_h, mean_n], yerr=[margin_h, margin_n],
                    fmt='none', color='black', capsize=5)
    axes[1].set_ylabel('Accuracy (mean ± 95% CI)')
    axes[1].set_title('Confidence intervals (Cross-Validation)')
    axes[1].set_ylim(0, 1.05)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# main pipeline
def main():
    base = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base, 'outputs'), exist_ok=True)
    
    print("EVIDENCE INTEGRATION & BIAS DISCOVERY")
    
    # load data
    df_lab, df_cons, df_framing = load_data()
    
    # drop missing values
    df_lab = df_lab.dropna(how='all', axis=1).dropna()
    df_cons = df_cons.dropna(how='all', axis=1).dropna()
    df_framing = df_framing.dropna(how='all', axis=1).dropna()
    
    results = {}
    
    # data_lab 
    print("\ndata_lab.csv")
    df_lab, cols_a, cols_b = preprocess_data_lab(df_lab)

    X_lab = build_features_lab(df_lab, cols_a, cols_b)
    y_human = df_lab['sequenceChoiceLeft']
    y_norm = df_lab['normative_choice']

    # shared train/test split for all models (RF, LSTM, RNN)
    train_idx, test_idx = train_test_split(
        np.arange(len(X_lab)), test_size=HP['test_size'],
        stratify=y_human, random_state=HP['random_state']
    )
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)} (same for RF, LSTM, RNN)")

    rf_results = run_random_forest(
        X_lab, y_human, y_norm, train_idx, test_idx, cv=HP['cv_folds']
    )
    results['data_lab'] = rf_results

    ci_h, ci_n = rf_results['ci_human'], rf_results['ci_norm']
    print(f"RF (human):  Test={rf_results['acc_test_human']:.3f}  |  CV {ci_h[0]:.3f} [{ci_h[2]:.3f}, {ci_h[3]:.3f}] 95% CI")
    print(f"RF (norm):   Test={rf_results['acc_test_norm']:.3f}  |  CV {ci_n[0]:.3f} [{ci_n[2]:.3f}, {ci_n[3]:.3f}] 95% CI")

    # cross-validation bar chart
    plot_cv_bar_chart(
        rf_results['cv_scores_human'], rf_results['cv_scores_norm'],
        rf_results['ci_human'], rf_results['ci_norm'],
        save_path=os.path.join(base, 'outputs', 'cv_bar_chart.png')
    )
    print("Saved: outputs/cv_bar_chart.png")

    # top feature importance
    fi = sorted(rf_results['feature_importance'].items(), key=lambda x: -x[1])[:5]
    print("Top 5 features:", fi)

    # residuals and clustering (on full dataset for visualization)
    X_full_s = rf_results['scaler'].transform(X_lab)
    residuals = compute_residuals(rf_results['rf_human'], X_lab, y_human, y_norm, rf_results['scaler'])
    labels, _ = cluster_residuals(residuals, X_full_s)

    df_lab['economic_loss'] = compute_economic_loss(df_lab, choice_col='sequenceChoiceLeft')

    # visualizations data_lab
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_psychometric_curve(df_lab, ax=axes[0, 0])
    plot_economic_loss(df_lab, ax=axes[0, 1])
    plot_tsne_clusters(residuals, X_full_s[:, :min(20, X_full_s.shape[1])], labels, ax=axes[1, 0])

    fi_df = pd.DataFrame(rf_results['feature_importance'].items(), columns=['feature', 'importance'])
    fi_df = fi_df.sort_values('importance', ascending=True).tail(15)
    fi_df.plot(x='feature', y='importance', kind='barh', ax=axes[1, 1], legend=False)
    axes[1, 1].set_title('Feature Importance (Random Forest)')
    axes[1, 1].set_xlabel('Importance')

    plt.tight_layout()
    plt.savefig(os.path.join(base, 'outputs', 'data_lab_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # LSTM and RNN on same test set
    seq_data = df_lab[cols_a + cols_b].values
    lstm_acc = rnn_acc = None

    lstm_res = run_lstm(seq_data, y_human, train_idx, test_idx)
    if lstm_res:
        lstm_acc = lstm_res['acc_test']
        print(f"LSTM (Test set): {lstm_acc:.3f}")
    rnn_res = run_rnn(seq_data, y_human, train_idx, test_idx)
    if rnn_res:
        rnn_acc = rnn_res['acc_test']
        print(f"RNN (Test set):  {rnn_acc:.3f}")

    
    # dataConsA.csv
    print("\ndataConsA.csv")
    df_cons, cols_l, cols_r = preprocess_data_cons(df_cons)
    X_cons = build_features_cons(df_cons, cols_l, cols_r)
    y_h_c = df_cons['responseLeft']
    y_n_c = df_cons['normative_choice']
    tr_c, te_c = train_test_split(np.arange(len(X_cons)), test_size=HP['test_size'],
                                   stratify=y_h_c, random_state=HP['random_state'])
    rf_cons = run_random_forest(X_cons, y_h_c, y_n_c, tr_c, te_c, cv=HP['cv_folds'])
    results['data_cons'] = rf_cons
    c_h = rf_cons['ci_human']
    print(f"RF (human):  Test={rf_cons['acc_test_human']:.3f}  |  CV {c_h[0]:.3f} [{c_h[2]:.3f}, {c_h[3]:.3f}] 95% CI")

    # LSTM and RNN (left/right pairs, same test set)
    lstm_cons, rnn_cons = None, None

    seq_cons = df_cons[cols_l + cols_r].values
    lstm_res_c = run_lstm(seq_cons, y_h_c, tr_c, te_c)
    if lstm_res_c:
        lstm_cons = lstm_res_c['acc_test']
        print(f"LSTM (Test set): {lstm_cons:.3f}")
    rnn_res_c = run_rnn(seq_cons, y_h_c, tr_c, te_c)
    if rnn_res_c:
        rnn_cons = rnn_res_c['acc_test']
        print(f"RNN (Test set):  {rnn_cons:.3f}")

    df_cons['economic_loss'] = compute_economic_loss(df_cons, choice_col='responseLeft')
    
    # DataHighFramingA.csv
    print("\ndataHighFramingA.csv")
    df_framing, left_cols, right_cols = preprocess_data_framing(df_framing)
    X_fr = build_features_framing(df_framing, left_cols, right_cols)
    y_h_f = df_framing['responseLeft']
    y_n_f = df_framing['normative_choice']
    tr_f, te_f = train_test_split(np.arange(len(X_fr)), test_size=HP['test_size'],
                                   stratify=y_h_f, random_state=HP['random_state'])
    rf_fr = run_random_forest(X_fr, y_h_f, y_n_f, tr_f, te_f, cv=HP['cv_folds'])
    results['data_framing'] = rf_fr
    c_f = rf_fr['ci_human']
    print(f"RF (human):  Test={rf_fr['acc_test_human']:.3f}  |  CV {c_f[0]:.3f} [{c_f[2]:.3f}, {c_f[3]:.3f}] 95% CI")

    # LSTM and RNN (left/right pairs, same test set)
    lstm_fr, rnn_fr = None, None
    seq_fr = df_framing[left_cols + right_cols].values
    lstm_res_f = run_lstm(seq_fr, y_h_f, tr_f, te_f)
    if lstm_res_f:
        lstm_fr = lstm_res_f['acc_test']
        print(f"LSTM (Test set): {lstm_fr:.3f}")
    rnn_res_f = run_rnn(seq_fr, y_h_f, tr_f, te_f)
    if rnn_res_f:
        rnn_fr = rnn_res_f['acc_test']
        print(f"RNN (Test set):  {rnn_fr:.3f}")

    df_framing['economic_loss'] = compute_economic_loss(df_framing, choice_col='responseLeft',
                                                        ev_left_col='EV_left', ev_right_col='EV_right')
    
    # final summary
    ch, cn = rf_results['ci_human'], rf_results['ci_norm']
    summary = {
        'hyperparameters': HP,
        'data_lab': {
            'test_acc_rf_human': float(rf_results['acc_test_human']),
            'test_acc_rf_normative': float(rf_results['acc_test_norm']),
            'cv_ci_human': f"{ch[0]:.3f} [{ch[2]:.3f}, {ch[3]:.3f}]",
            'cv_ci_normative': f"{cn[0]:.3f} [{cn[2]:.3f}, {cn[3]:.3f}]",
            'test_acc_lstm': float(lstm_acc) if lstm_acc is not None else None,
            'test_acc_rnn': float(rnn_acc) if rnn_acc is not None else None,
        },
        'data_cons': {
            'test_acc_rf': float(rf_cons['acc_test_human']),
            'test_acc_lstm': float(lstm_cons) if lstm_cons is not None else None,
            'test_acc_rnn': float(rnn_cons) if rnn_cons is not None else None,
            'cv_ci': f"{rf_cons['ci_human'][0]:.3f} [{rf_cons['ci_human'][2]:.3f}, {rf_cons['ci_human'][3]:.3f}]",
        },
        'data_framing': {
            'test_acc_rf': float(rf_fr['acc_test_human']),
            'test_acc_lstm': float(lstm_fr) if lstm_fr is not None else None,
            'test_acc_rnn': float(rnn_fr) if rnn_fr is not None else None,
            'cv_ci': f"{rf_fr['ci_human'][0]:.3f} [{rf_fr['ci_human'][2]:.3f}, {rf_fr['ci_human'][3]:.3f}]",
        },
    }
    
    with open(os.path.join(base, 'outputs', 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
if __name__ == '__main__':
    main()
