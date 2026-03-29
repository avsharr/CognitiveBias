"""Loading raw CSVs, per-dataset preprocessing, feature matrices, unified dataset."""

import os

import numpy as np
import pandas as pd
from scipy import stats

from ..paths import DATASETS_DIR


def load_data():
    df_lab = pd.read_csv(os.path.join(DATASETS_DIR, 'data_lab.csv'))
    df_cons = pd.read_csv(os.path.join(DATASETS_DIR, 'dataConsA.csv'))
    df_framing = pd.read_csv(os.path.join(DATASETS_DIR, 'dataHighFramingA.csv'))
    return df_lab, df_cons, df_framing


def add_pair_features_to_df(df, left_cols, right_cols):
    out = df.copy()
    L = out[left_cols].values.astype(np.float64)
    R = out[right_cols].values.astype(np.float64)
    D = L - R
    out['pair_diff_std'] = D.std(axis=1)
    out['pair_diff_max'] = D.max(axis=1)
    out['pair_diff_min'] = D.min(axis=1)
    out['pair_winner_balance'] = (L > R).sum(axis=1) - (R > L).sum(axis=1)
    ent = np.empty(len(out), dtype=np.float64)
    for i in range(len(out)):
        _d = D[i]
        _, counts = np.unique(_d, return_counts=True)
        p = counts.astype(np.float64) / counts.sum()
        ent[i] = stats.entropy(p)
    out['pair_diff_entropy'] = ent
    return out


PAIR_FEATURE_COLS = [
    'pair_diff_std', 'pair_diff_max', 'pair_diff_min',
    'pair_winner_balance', 'pair_diff_entropy',
]


def preprocess_data_lab(df):
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
    all_vals = df[cols_a + cols_b]
    df['complexity_diff'] = -np.abs(df['EV_diff'])
    df['complexity_sd'] = all_vals.std(axis=1)
    df['complexity'] = df['complexity_diff'] + df['complexity_sd']
    df = add_pair_features_to_df(df, cols_a, cols_b)
    return df, cols_a, cols_b


def preprocess_data_cons(df):
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
    df = add_pair_features_to_df(df, cols_l, cols_r)
    # leftCorrect дублирует normative_choice (EV); в анализе используем responseLeft → human_choice и is_correct
    lc = [c for c in df.columns if c.lower() == 'leftcorrect']
    if lc:
        df = df.drop(columns=lc)
    return df, cols_l, cols_r


def preprocess_data_framing(df):
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
    df = add_pair_features_to_df(df, left_cols, right_cols)
    return df, left_cols, right_cols


def build_features_lab(df, cols_a, cols_b):
    feat_cols = cols_a + cols_b + ['EV_diff', 'complexity', 'isDifferentMeanSameVariance'] + PAIR_FEATURE_COLS
    X = df[feat_cols].copy()
    X = X.fillna(X.median())
    return X


def build_features_cons(df, cols_l, cols_r):
    feat_cols = cols_l + cols_r + ['EV_diff', 'complexity'] + PAIR_FEATURE_COLS
    if 'flip' in df.columns:
        feat_cols.append('flip')
    X = df[[c for c in feat_cols if c in df.columns]].copy()
    X = X.fillna(X.median())
    return X


def build_features_framing(df, left_cols, right_cols):
    feat_cols = left_cols + right_cols + ['EV_diff', 'complexity'] + PAIR_FEATURE_COLS
    X = df[[c for c in feat_cols if c in df.columns]].copy()
    X = X.fillna(X.median())
    return X


def make_unified_block(df, left_cols, right_cols, human_choice_col, source_tag):
    u = pd.DataFrame()
    for i in range(8):
        u[f'L{i + 1}'] = df[left_cols[i]].values
        u[f'R{i + 1}'] = df[right_cols[i]].values
    u['human_choice'] = df[human_choice_col].astype(int).values
    u['normative_choice'] = df['normative_choice'].astype(int).values
    u['EV_left'] = df['EV_left'].values
    u['EV_right'] = df['EV_right'].values
    u['EV_diff'] = df['EV_diff'].values
    u['complexity'] = df['complexity'].values
    u['is_correct'] = df['is_correct'].astype(int).values
    u['dataset_source'] = source_tag
    n = len(df)
    if 'subject' in df.columns:
        u['subject'] = pd.to_numeric(df['subject'], errors='coerce').astype('Int64').values
    else:
        u['subject'] = pd.array([pd.NA] * n, dtype='Int64')
    lc = [f'L{i}' for i in range(1, 9)]
    rc = [f'R{i}' for i in range(1, 9)]
    u = add_pair_features_to_df(u, lc, rc)
    return u


def build_features_unified(df_uni):
    seq_cols = [f'L{i}' for i in range(1, 9)] + [f'R{i}' for i in range(1, 9)]
    base_cols = seq_cols + ['EV_diff', 'complexity'] + PAIR_FEATURE_COLS
    X = df_uni[base_cols].copy()
    X = X.fillna(X.median())
    dums = pd.get_dummies(df_uni['dataset_source'], prefix='src', dtype=float)
    X = pd.concat([X, dums], axis=1)
    return X


def unified_sequence_matrix(df_uni):
    cols = [f'L{i}' for i in range(1, 9)] + [f'R{i}' for i in range(1, 9)]
    return df_uni[cols].values.astype(np.float64)


def save_preprocessed_csv(df, preprocessing_dir, filename='preprocessed.csv'):
    os.makedirs(preprocessing_dir, exist_ok=True)
    df.to_csv(os.path.join(preprocessing_dir, filename), index=False)
