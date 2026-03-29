"""Объединение трёх CSV в единый датасет по статистикам последовательностей + краткий анализ влияния на выбор.

Сохраняет ``datasets/unified_dataset.csv`` (колонки только из STAT_COLS).
Источник ``source`` не входит в матрицу признаков моделей ниже (one-hot не строим).

Фактические имена колонок в репозитории:
  - data_lab.csv: sequenceA1…8 / sequenceB1…8, sequenceChoiceLeft
  - dataConsA.csv: sequenceLeft_* / sequenceRight_*, responseLeft
  - dataHighFramingA.csv: sequences_ 1…16, responseLeft
"""

from __future__ import annotations

import json
import os
import re
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .paths import DATASETS_DIR, OUTPUTS_DIR, UNIFIED_DATASET_CSV

STAT_COLS = [
    'mean_left', 'mean_right', 'var_left', 'var_right',
    'diff_mean', 'mean_var', 'diff_var',
    'response_left', 'correct', 'source',
]

MERGE_ANALYSIS_JSON = os.path.join(OUTPUTS_DIR, 'merge_data_analysis.json')


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.str.strip()
    return out


def _seq_index_from_col(name: str) -> int:
    s = re.sub(r'\s+', '', str(name))
    m = re.search(r'(\d+)$', s)
    return int(m.group(1)) if m else -1


def _lab_lr_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    left = sorted(
        [c for c in df.columns if re.match(r'^sequenceLeft_\d+$', c)],
        key=_seq_index_from_col,
    )
    right = sorted(
        [c for c in df.columns if re.match(r'^sequenceRight_\d+$', c)],
        key=_seq_index_from_col,
    )
    if len(left) == 8 and len(right) == 8:
        return left, right
    left = sorted(
        [c for c in df.columns if c.startswith('sequenceA') and c[9:].isdigit()],
        key=lambda x: int(x.replace('sequenceA', '')),
    )
    right = sorted(
        [c for c in df.columns if c.startswith('sequenceB') and c[9:].isdigit()],
        key=lambda x: int(x.replace('sequenceB', '')),
    )
    if len(left) != 8 or len(right) != 8:
        raise ValueError('data_lab: ожидаются 8+8 колонок sequenceA*/sequenceB* или sequenceLeft_*/sequenceRight_*')
    return left, right


def _cons_lr_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    left = sorted(
        [c for c in df.columns if c.startswith('sequenceLeft_')],
        key=_seq_index_from_col,
    )
    right = sorted(
        [c for c in df.columns if c.startswith('sequenceRight_')],
        key=_seq_index_from_col,
    )
    if len(left) != 8 or len(right) != 8:
        raise ValueError('dataConsA: ожидаются sequenceLeft_1…8 и sequenceRight_1…8')
    return left, right


def _framing_lr_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    seq_cols = [c for c in df.columns if 'sequence' in c.lower()]
    seq_cols = sorted(seq_cols, key=_seq_index_from_col)
    seq_cols = [c for c in seq_cols if _seq_index_from_col(c) > 0]
    if len(seq_cols) < 16:
        raise ValueError('dataHighFramingA: ожидаются 16 колонок sequences_*')
    seq_cols = seq_cols[:16]
    return seq_cols[:8], seq_cols[8:16]


def _response_target_lab(df: pd.DataFrame) -> pd.Series:
    if 'sequenceChoiceLeft' in df.columns:
        return df['sequenceChoiceLeft'].astype(int)
    if 'responseLeft' in df.columns:
        return df['responseLeft'].astype(int)
    raise ValueError('data_lab: нет sequenceChoiceLeft / responseLeft')


def _response_target_cons_framing(df: pd.DataFrame) -> pd.Series:
    if 'responseLeft' not in df.columns:
        raise ValueError('ожидается responseLeft')
    return df['responseLeft'].astype(int)


def _compute_correct(mean_left: np.ndarray, mean_right: np.ndarray, response_left: np.ndarray) -> np.ndarray:
    """1, если выбрана сторона с большим средним; при равенстве средних — засчитываем попытку как верную."""
    ml = np.asarray(mean_left, dtype=float)
    mr = np.asarray(mean_right, dtype=float)
    r = np.asarray(response_left, dtype=int)
    correct = np.ones(len(r), dtype=int)
    mask_l = ml > mr
    mask_r = mr > ml
    correct[mask_l] = (r[mask_l] == 1).astype(int)
    correct[mask_r] = (r[mask_r] == 0).astype(int)
    return correct


def _block_from_df(
    df: pd.DataFrame,
    left_cols: Sequence[str],
    right_cols: Sequence[str],
    response: pd.Series,
    source: str,
) -> pd.DataFrame:
    L = df[list(left_cols)].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    R = df[list(right_cols)].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    mean_left = L.mean(axis=1)
    mean_right = R.mean(axis=1)
    var_left = L.var(axis=1, ddof=1)
    var_right = R.var(axis=1, ddof=1)
    diff_mean = mean_left - mean_right
    mean_var = (var_left + var_right) / 2.0
    diff_var = var_left - var_right
    resp = response.to_numpy(dtype=int)
    correct = _compute_correct(mean_left, mean_right, resp)
    return pd.DataFrame({
        'mean_left': mean_left,
        'mean_right': mean_right,
        'var_left': var_left,
        'var_right': var_right,
        'diff_mean': diff_mean,
        'mean_var': mean_var,
        'diff_var': diff_var,
        'response_left': resp,
        'correct': correct,
        'source': source,
    })


def load_and_transform_blocks() -> pd.DataFrame:
    path_lab = os.path.join(DATASETS_DIR, 'data_lab.csv')
    path_cons = os.path.join(DATASETS_DIR, 'dataConsA.csv')
    path_framing = os.path.join(DATASETS_DIR, 'dataHighFramingA.csv')

    d_lab = _strip_columns(pd.read_csv(path_lab)).dropna(how='all', axis=1).dropna()
    d_cons = _strip_columns(pd.read_csv(path_cons)).dropna(how='all', axis=1).dropna()
    d_fr = _strip_columns(pd.read_csv(path_framing)).dropna(how='all', axis=1).dropna()

    ll, lr = _lab_lr_columns(d_lab)
    cl, cr = _cons_lr_columns(d_cons)
    fl, fr = _framing_lr_columns(d_fr)

    b_lab = _block_from_df(d_lab, ll, lr, _response_target_lab(d_lab), 'data_lab')
    b_cons = _block_from_df(d_cons, cl, cr, _response_target_cons_framing(d_cons), 'data_cons')
    b_fr = _block_from_df(d_fr, fl, fr, _response_target_cons_framing(d_fr), 'data_framing')

    out = pd.concat([b_lab, b_cons, b_fr], ignore_index=True)
    return out.dropna(subset=[c for c in STAT_COLS if c != 'source'])


def analyze_sequence_stats(df: pd.DataFrame) -> dict:
    """Влияние статистик последовательностей на response_left (без one-hot по source)."""
    feature_cols = [
        'mean_left', 'mean_right', 'var_left', 'var_right',
        'diff_mean', 'mean_var', 'diff_var',
    ]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(df[feature_cols].median())
    y = df['response_left'].astype(int)

    corrs = {}
    for col in feature_cols:
        r, p = stats.pointbiserialr(y, X[col])
        corrs[col] = {'r_pointbiserial': float(r), 'p_value': float(p)}

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, random_state=42)),
    ])
    pipe.fit(X, y)
    acc = accuracy_score(y, pipe.predict(X))
    try:
        auc = roc_auc_score(y, pipe.predict_proba(X)[:, 1])
    except ValueError:
        auc = float('nan')
    cv_acc = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')

    coefs = dict(zip(feature_cols, pipe.named_steps['clf'].coef_.ravel().tolist()))
    intercept = float(pipe.named_steps['clf'].intercept_[0])

    by_src = df.groupby('source').agg(
        n=('response_left', 'count'),
        p_choose_left=('response_left', 'mean'),
        accuracy=('correct', 'mean'),
        mean_diff_mean=('diff_mean', 'mean'),
        mean_mean_var=('mean_var', 'mean'),
    ).reset_index()
    by_src_dict = by_src.to_dict(orient='records')

    return {
        'n_rows': int(len(df)),
        'point_biserial_response_left': corrs,
        'logistic_standardized': {
            'train_accuracy': float(acc),
            'roc_auc_train': float(auc),
            'cv_accuracy_mean': float(cv_acc.mean()),
            'cv_accuracy_std': float(cv_acc.std()),
            'coefficients': coefs,
            'intercept': intercept,
        },
        'by_source': by_src_dict,
    }


def main() -> None:
    os.makedirs(os.path.dirname(UNIFIED_DATASET_CSV), exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    df = load_and_transform_blocks()
    df[STAT_COLS].to_csv(UNIFIED_DATASET_CSV, index=False)
    print(f'Сохранено: {UNIFIED_DATASET_CSV} ({len(df)} строк)', flush=True)

    report = analyze_sequence_stats(df)
    report['output_csv'] = UNIFIED_DATASET_CSV
    with open(MERGE_ANALYSIS_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'Анализ: {MERGE_ANALYSIS_JSON}', flush=True)

    print('\nСводка по источникам:', flush=True)
    for row in report['by_source']:
        print(
            f"  {row['source']}: n={row['n']}, P(left)={row['p_choose_left']:.3f}, "
            f"accuracy={row['accuracy']:.3f}, mean(diff_mean)={row['mean_diff_mean']:.4f}",
            flush=True,
        )
    print('\nPoint-biserial (response_left ~ признак):', flush=True)
    for k, v in report['point_biserial_response_left'].items():
        print(f"  {k}: r={v['r_pointbiserial']:.4f}, p={v['p_value']:.2e}", flush=True)
    lg = report['logistic_standardized']
    print(
        f"\nLogisticRegression (стандартизованные признаки, без source): "
        f"acc(train)={lg['train_accuracy']:.3f}, AUC={lg['roc_auc_train']:.3f}, "
        f"CV acc={lg['cv_accuracy_mean']:.3f} ± {lg['cv_accuracy_std']:.3f}",
        flush=True,
    )


if __name__ == '__main__':
    main()
