"""Tabular and sequence models: training, CV, metrics, bundle runner."""

import itertools
import os
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    XGBClassifier = None
    _HAS_XGB = False

from ..preprocessing.data_pipeline import PAIR_FEATURE_COLS


# hyperparameters
HP = {
    'random_state': 10,
    'test_size': 0.2,
    'cv_folds': 5,
    'ci_alpha': 0.95,  # 95% confidence interval
}

# grid search param grids (tuned per dataset)
RF_PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2],
}

LOGREG_PARAM_GRID = {
    'clf__C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'clf__penalty': ['l2'],
    'clf__solver': ['lbfgs'],
}

XGB_PARAM_GRID = {
    'clf__n_estimators': [100, 250],
    'clf__max_depth': [4, 8, 12],
    'clf__learning_rate': [0.05, 0.1],
    'clf__subsample': [0.9],
    'clf__colsample_bytree': [0.9],
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


def confidence_interval(scores, alpha=0.95):
    """95% confidence interval for mean of CV scores."""
    n = len(scores)
    mean = np.mean(scores)
    std = np.std(scores, ddof=1) if n > 1 else 0
    # t-distribution for small n
    t_val = stats.t.ppf((1 + alpha) / 2, n - 1) if n > 1 else 0
    margin = t_val * (std / np.sqrt(n)) if n > 1 else 0
    return mean, margin, mean - margin, mean + margin


def run_random_forest(X, y_human, train_idx, test_idx, cv=5):
    """Random Forest: GridSearchCV, только предсказание человеческого выбора."""
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_h_train, y_h_test = y_human.iloc[train_idx], y_human.iloc[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=HP['random_state'])
    base_rf = RandomForestClassifier(random_state=HP['random_state'])

    gs_human = GridSearchCV(base_rf, RF_PARAM_GRID, cv=cv_split, scoring='accuracy', n_jobs=1, verbose=0)
    gs_human.fit(X_train_s, y_h_train)
    rf_human = gs_human.best_estimator_
    cv_scores_human = np.array([gs_human.cv_results_[f'split{i}_test_score'][gs_human.best_index_] for i in range(cv)])
    acc_test_human = accuracy_score(y_h_test, rf_human.predict(X_test_s))
    ci_human = confidence_interval(cv_scores_human, HP['ci_alpha'])

    return {
        'rf_human': rf_human,
        'scaler': scaler,
        'acc_test_human': acc_test_human,
        'cv_scores_human': cv_scores_human,
        'ci_human': ci_human,
        'feature_importance': dict(zip(X.columns, rf_human.feature_importances_)),
        'feature_names': list(X.columns),
        'best_params_human': gs_human.best_params_,
    }


def run_logistic_regression(X, y_human, train_idx, test_idx, cv=5):
    """Logistic regression (baseline LG), GridSearchCV + CV fold scores + 95% CI (только human)."""
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_h_train, y_h_test = y_human.iloc[train_idx], y_human.iloc[test_idx]

    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=HP['random_state'])

    pipe_h = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=HP['random_state'], max_iter=8000)),
    ])
    gs_h = GridSearchCV(pipe_h, LOGREG_PARAM_GRID, cv=cv_split, scoring='accuracy', n_jobs=1, verbose=0)
    gs_h.fit(X_train, y_h_train)
    m_h = gs_h.best_estimator_
    cv_scores_human = np.array([gs_h.cv_results_[f'split{i}_test_score'][gs_h.best_index_] for i in range(cv)])
    acc_test_human = accuracy_score(y_h_test, m_h.predict(X_test))
    ci_human = confidence_interval(cv_scores_human, HP['ci_alpha'])

    return {
        'model_human': m_h,
        'acc_test_human': acc_test_human,
        'cv_scores_human': cv_scores_human,
        'ci_human': ci_human,
        'best_params_human': gs_h.best_params_,
    }


def run_xgboost(X, y_human, train_idx, test_idx, cv=5):
    """XGBoost, GridSearchCV + CV (только human)."""
    if not _HAS_XGB:
        return None

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_h_train, y_h_test = y_human.iloc[train_idx], y_human.iloc[test_idx]

    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=HP['random_state'])

    pipe_h = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(
            random_state=HP['random_state'],
            tree_method='hist',
            n_jobs=1,
        )),
    ])
    gs_h = GridSearchCV(pipe_h, XGB_PARAM_GRID, cv=cv_split, scoring='accuracy', n_jobs=1, verbose=0)
    gs_h.fit(X_train, y_h_train)
    m_h = gs_h.best_estimator_
    cv_scores_human = np.array([gs_h.cv_results_[f'split{i}_test_score'][gs_h.best_index_] for i in range(cv)])
    acc_test_human = accuracy_score(y_h_test, m_h.predict(X_test))
    ci_human = confidence_interval(cv_scores_human, HP['ci_alpha'])

    return {
        'model_human': m_h,
        'acc_test_human': acc_test_human,
        'cv_scores_human': cv_scores_human,
        'ci_human': ci_human,
        'best_params_human': gs_h.best_params_,
    }


def save_rf_feature_analysis(X, correlation_csv_path, heatmap_png_path):
    """Feature correlations: CSV under ``outputs/``, heatmap PNG under ``figures/``."""
    analysis_cols = [c for c in (PAIR_FEATURE_COLS + ['EV_diff', 'complexity']) if c in X.columns]
    if len(analysis_cols) < 2:
        return
    os.makedirs(os.path.dirname(correlation_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(heatmap_png_path), exist_ok=True)
    sub = X[analysis_cols].copy().fillna(X[analysis_cols].median())
    corr = sub.corr()
    corr.to_csv(correlation_csv_path, encoding='utf-8')
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('Feature correlation (RF block: pair stats, EV_diff, complexity)')
    plt.tight_layout()
    plt.savefig(heatmap_png_path, dpi=150, bbox_inches='tight')
    plt.close()


def _ci_to_dict(ci_tup):
    m, mar, lo, hi = ci_tup
    return {'mean': float(m), 'margin': float(mar), 'low': float(lo), 'high': float(hi)}


def _seq_to_3d(X_seq):
    """convert sequence (n, 16) to (n, 8, 2)."""
    if X_seq.shape[1] != 16:
        return None
    return np.stack([
        np.column_stack([X_seq[:, i], X_seq[:, i + 8]]) for i in range(8)
    ], axis=1)


def _grid_search_lstm_rnn(X_train, y_train, X_test, y_test, param_grid, model_type='lstm'):
    """grid search for LSTM or RNN; returns best model, acc_test, best_params."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    n_combo = len(combos)
    best_val_acc = -1
    best_model = None
    best_params = None

    for ci, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        print(f"  [{model_type.upper()}] {ci}/{n_combo} (units={params['units']}, epochs={params['epochs']})", flush=True)
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
    from tensorflow.keras.utils import to_categorical

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
    from tensorflow.keras.utils import to_categorical

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


def _keras_fit_eval_valacc(X_tr, y_tr_cat, X_va, y_va_int, params, model_type):
    """Одна сеть, обучение на X_tr, точность на X_va (целые метки 0/1)."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout

    tf.keras.backend.clear_session()
    layer = LSTM if model_type == 'lstm' else SimpleRNN
    model = Sequential([
        layer(params['units'], input_shape=(8, 2), return_sequences=False),
        Dropout(params['dropout']),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        X_tr, y_tr_cat,
        epochs=params['epochs'], batch_size=params['batch_size'],
        verbose=0,
    )
    pred = model.predict(X_va, verbose=0).argmax(axis=1)
    return float((pred == y_va_int).mean())


def run_keras_cv_on_train(seq_16, y_human, train_idx, best_params, model_type, cv_folds=None, dataset_label=''):
    """Стратифицированный K-fold только на train-индексах; лучшие гиперы из grid search; mean ± 95% CI по фолдам."""
    if not best_params:
        return None
    if cv_folds is None:
        cv_folds = HP['cv_folds']
    from tensorflow.keras.utils import to_categorical

    X_3d = _seq_to_3d(seq_16)
    if X_3d is None:
        return None

    y_all = y_human.iloc[train_idx].values if hasattr(y_human, 'iloc') else np.asarray(y_human)[train_idx]
    X_tr_all = X_3d[train_idx]
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=HP['random_state'])
    scores = []
    for fi, (tr_loc, va_loc) in enumerate(skf.split(X_tr_all, y_all), start=1):
        print(f"  [{model_type.upper()}] CV по train {fi}/{cv_folds} [{dataset_label}]", flush=True)
        X_a = X_tr_all[tr_loc]
        X_b = X_tr_all[va_loc]
        y_a = to_categorical(y_all[tr_loc], 2)
        y_b_int = y_all[va_loc]
        acc = _keras_fit_eval_valacc(X_a, y_a, X_b, y_b_int, best_params, model_type)
        scores.append(acc)
    arr = np.asarray(scores, dtype=np.float64)
    return {
        'cv_scores': arr,
        'ci': confidence_interval(scores, HP['ci_alpha']),
    }


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


def plot_cv_bar_chart(cv_scores_human, ci_human, save_path=None):
    """CV accuracy по фолдам + среднее ± 95% CI (RF на человеческом выборе)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    folds = np.arange(1, len(cv_scores_human) + 1)
    x = np.arange(len(folds))
    axes[0].bar(x, cv_scores_human, color='steelblue', alpha=0.8, label='RF (human)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'Fold {i}' for i in folds])
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Cross-Validation: accuracy by fold')
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)

    mean_h, margin_h = ci_human[0], ci_human[1]
    axes[1].bar(['RF (human)'], [mean_h], color='steelblue', alpha=0.8)
    axes[1].errorbar([0], [mean_h], yerr=[margin_h], fmt='none', color='black', capsize=5)
    axes[1].set_ylabel('Accuracy (mean ± 95% CI)')
    axes[1].set_title('Cross-Validation: mean ± CI')
    axes[1].set_ylim(0, 1.05)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def _summarize_bundle(b):
    """Сводка в JSON: LG, RF, XGB (если есть), LSTM/RNN + CV±95% CI (только human)."""
    rf = b['rf']
    lr = b['logreg']
    xgb = b.get('xgb')
    lstm_res, rnn_res = b.get('lstm'), b.get('rnn')
    lstm_cv, rnn_cv = b.get('lstm_cv'), b.get('rnn_cv')

    ch = rf['ci_human']
    out = {
        'logistic_regression_baseline': {
            'test_acc_human': float(lr['acc_test_human']),
            'cv_ci_human': _ci_to_dict(lr['ci_human']),
        },
        'random_forest': {
            'test_acc_human': float(rf['acc_test_human']),
            'cv_ci_human': _ci_to_dict(ch),
            'cv_ci_human_str': f"{ch[0]:.3f} [{ch[2]:.3f}, {ch[3]:.3f}]",
        },
        'xgboost': None,
        'lstm': {
            'test_acc': float(lstm_res['acc_test']) if lstm_res else None,
        },
        'rnn': {
            'test_acc': float(rnn_res['acc_test']) if rnn_res else None,
        },
    }
    if xgb is not None:
        xh = xgb['ci_human']
        out['xgboost'] = {
            'test_acc_human': float(xgb['acc_test_human']),
            'cv_ci_human': _ci_to_dict(xh),
        }
    if lstm_res and lstm_cv:
        out['lstm']['cv_on_train'] = {
            'fold_scores': [float(x) for x in lstm_cv['cv_scores']],
            'ci_95': _ci_to_dict(lstm_cv['ci']),
        }
    if rnn_res and rnn_cv:
        out['rnn']['cv_on_train'] = {
            'fold_scores': [float(x) for x in rnn_cv['cv_scores']],
            'ci_95': _ci_to_dict(rnn_cv['ci']),
        }
    return out


def _run_bundle(X, y_human, seq_16, cv_chart_path, dataset_label,
                rf_corr_csv_path=None, rf_corr_heatmap_path=None):
    """Один сплит: LG (+GridSearch+CV+CI), RF, XGB, корреляции фич RF, LSTM/RNN + CV по train (только human)."""
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=HP['test_size'],
        stratify=y_human, random_state=HP['random_state']
    )
    print(f"[{dataset_label}] Train: {len(train_idx)}, Test: {len(test_idx)}", flush=True)

    print(f"[{dataset_label}] Logistic Regression (baseline) + GridSearchCV...", flush=True)
    lr_results = run_logistic_regression(X, y_human, train_idx, test_idx, cv=HP['cv_folds'])
    lrh = lr_results['ci_human']
    print(f"[{dataset_label}] LG  human Test={lr_results['acc_test_human']:.3f} | CV {lrh[0]:.3f} [{lrh[2]:.3f},{lrh[3]:.3f}] 95% CI", flush=True)

    print(f"[{dataset_label}] Random Forest + GridSearchCV...", flush=True)
    rf_results = run_random_forest(X, y_human, train_idx, test_idx, cv=HP['cv_folds'])
    rfh = rf_results['ci_human']
    print(f"[{dataset_label}] RF  human Test={rf_results['acc_test_human']:.3f} | CV {rfh[0]:.3f} [{rfh[2]:.3f},{rfh[3]:.3f}] 95% CI", flush=True)

    if rf_corr_csv_path and rf_corr_heatmap_path:
        save_rf_feature_analysis(X, rf_corr_csv_path, rf_corr_heatmap_path)
        print(f"[{dataset_label}] RF feature correlation CSV: {rf_corr_csv_path}", flush=True)
        print(f"[{dataset_label}] RF feature correlation figure: {rf_corr_heatmap_path}", flush=True)

    if cv_chart_path:
        plot_cv_bar_chart(
            rf_results['cv_scores_human'],
            rf_results['ci_human'],
            save_path=cv_chart_path
        )
        print(f"[{dataset_label}] Saved: {cv_chart_path}")

    xgb_results = None
    if _HAS_XGB:
        print(f"[{dataset_label}] XGBoost + GridSearchCV...", flush=True)
        xgb_results = run_xgboost(X, y_human, train_idx, test_idx, cv=HP['cv_folds'])
        if xgb_results is not None:
            xh = xgb_results['ci_human']
            print(f"[{dataset_label}] XGB human Test={xgb_results['acc_test_human']:.3f} | CV {xh[0]:.3f} [{xh[2]:.3f},{xh[3]:.3f}] 95% CI", flush=True)
    else:
        print(f"[{dataset_label}] XGBoost пропущен (нет пакета xgboost). Установите: pip install xgboost", flush=True)

    lstm_res = rnn_res = None
    lstm_cv = rnn_cv = None
    print(f"[{dataset_label}] Загрузка TensorFlow и подбор LSTM (может занять несколько минут)...", flush=True)
    lstm_res = run_lstm(seq_16, y_human, train_idx, test_idx)
    if lstm_res:
        print(f"[{dataset_label}] LSTM (Test): {lstm_res['acc_test']:.3f}", flush=True)
        print(f"[{dataset_label}] LSTM: {HP['cv_folds']}-fold CV на train (лучшие гиперпараметры)...", flush=True)
        lstm_cv = run_keras_cv_on_train(seq_16, y_human, train_idx, lstm_res['best_params'], 'lstm', dataset_label=dataset_label)
        if lstm_cv:
            lc = lstm_cv['ci']
            print(f"[{dataset_label}] LSTM CV train mean={lc[0]:.3f} ± {lc[1]:.3f} (95% CI [{lc[2]:.3f},{lc[3]:.3f}])", flush=True)

    print(f"[{dataset_label}] Подбор RNN...", flush=True)
    rnn_res = run_rnn(seq_16, y_human, train_idx, test_idx)
    if rnn_res:
        print(f"[{dataset_label}] RNN (Test):  {rnn_res['acc_test']:.3f}", flush=True)
        print(f"[{dataset_label}] RNN: {HP['cv_folds']}-fold CV на train...", flush=True)
        rnn_cv = run_keras_cv_on_train(seq_16, y_human, train_idx, rnn_res['best_params'], 'rnn', dataset_label=dataset_label)
        if rnn_cv:
            rc = rnn_cv['ci']
            print(f"[{dataset_label}] RNN CV train mean={rc[0]:.3f} ± {rc[1]:.3f} (95% CI [{rc[2]:.3f},{rc[3]:.3f}])", flush=True)

    return {
        'train_idx': train_idx,
        'test_idx': test_idx,
        'logreg': lr_results,
        'rf': rf_results,
        'xgb': xgb_results,
        'lstm': lstm_res,
        'rnn': rnn_res,
        'lstm_cv': lstm_cv,
        'rnn_cv': rnn_cv,
    }
