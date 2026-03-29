"""Project root and artifact directories (datasets, preprocessing, figures, models, outputs).

Source code for data transforms lives under ``scripts/preprocessing/``;
model training under ``scripts/models/``. Do not edit ``build/`` — it is a setuptools cache.
"""

import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = _PROJECT_ROOT

DATASETS_DIR = os.path.join(_PROJECT_ROOT, "datasets")
# Сводный CSV по статистикам последовательностей (scripts.merge_data); не путать с preprocessing/combined/unified_dataset.csv
UNIFIED_DATASET_CSV = os.path.join(DATASETS_DIR, "unified_dataset.csv")

PREPROCESSING_DIR = os.path.join(_PROJECT_ROOT, "preprocessing")
PRE_LAB = os.path.join(PREPROCESSING_DIR, "data_lab")
PRE_CONS = os.path.join(PREPROCESSING_DIR, "data_cons")
PRE_FRAMING = os.path.join(PREPROCESSING_DIR, "data_framing")
PRE_COMBINED = os.path.join(PREPROCESSING_DIR, "combined")

# Serialized checkpoints (.pkl, .joblib, .h5, …)
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
MOD_LAB = os.path.join(MODELS_DIR, "data_lab")
MOD_CONS = os.path.join(MODELS_DIR, "data_cons")
MOD_FRAMING = os.path.join(MODELS_DIR, "data_framing")
MOD_COMBINED = os.path.join(MODELS_DIR, "combined")

FIGURES_DIR = os.path.join(_PROJECT_ROOT, "figures")
FIG_LAB = os.path.join(FIGURES_DIR, "data_lab")
FIG_CONS = os.path.join(FIGURES_DIR, "data_cons")
FIG_FRAMING = os.path.join(FIGURES_DIR, "data_framing")
FIG_COMBINED = os.path.join(FIGURES_DIR, "combined")

# Tabular / JSON results only (.csv, .json)
OUTPUTS_DIR = os.path.join(_PROJECT_ROOT, "outputs")
OUT_COMBINED = os.path.join(OUTPUTS_DIR, "combined")
OUT_SEPARATE = os.path.join(OUTPUTS_DIR, "separate_analysis")
OUT_ADVANCED = os.path.join(OUTPUTS_DIR, "advanced_analysis")

RES_LAB = os.path.join(OUT_SEPARATE, "data_lab")
RES_CONS = os.path.join(OUT_SEPARATE, "data_cons")
RES_FRAMING = os.path.join(OUT_SEPARATE, "data_framing")
RES_COMBINED = OUT_COMBINED

SUMMARY_JSON = os.path.join(OUTPUTS_DIR, "summary.json")
ADVANCED_ANALYSIS_JSON = os.path.join(OUT_ADVANCED, "advanced_analysis_results.json")

RESULTS_DIR = OUTPUTS_DIR

SUBDIRS_TO_CREATE = [
    PRE_LAB, PRE_CONS, PRE_FRAMING, PRE_COMBINED,
    MOD_LAB, MOD_CONS, MOD_FRAMING, MOD_COMBINED,
    FIG_LAB, FIG_CONS, FIG_FRAMING, FIG_COMBINED,
    RES_LAB, RES_CONS, RES_FRAMING, RES_COMBINED,
    OUT_SEPARATE, OUT_ADVANCED,
    OUTPUTS_DIR,
]
