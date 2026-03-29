# Evidence integration and bias discovery (lottery choice)

Python project for analyzing information search and evidence integration in multi-outcome lottery choice experiments. Raw trials live under `datasets/`; **all data transforms** are implemented in `scripts/preprocessing/`, and **all model training / evaluation helpers** in `scripts/models/`. The editable installable package is the `scripts` namespace (see `pyproject.toml`).

> **Do not edit `build/`** — it is created by setuptools and may contain stale copies of files. The source of truth is `scripts/`.

## Repository layout

| Path | Purpose |
|------|---------|
| `datasets/` | Raw CSVs plus **`unified_dataset.csv`** (статистики последовательностей: `python -m scripts.merge_data`) |
| `preprocessing/` | Processed tables written by the pipeline (`preprocessed.csv` per split, `combined/unified_dataset.csv`) |
| `models/` | Serialized checkpoints (`.pkl`, `.joblib`, `.h5`, …) per dataset, when saved |
| `figures/` | Plots (`.png`) per dataset: CV bars, RF correlation heatmaps, lab bias panels, advanced-analysis figures |
| `outputs/` | Tabular / JSON only: `summary.json`, `merge_data_analysis.json`; `combined/` и `separate_analysis/`; `advanced_analysis/` |
| `scripts/paths.py` | Central path constants |
| `scripts/preprocessing/` | Load → clean → feature construction → unified merge |
| `scripts/models/` | Logistic regression, RF, XGBoost, LSTM/RNN, grids, CV, bundle runner |
| `scripts/merge_data.py` | Три CSV → `unified_dataset.csv` + анализ влияния mean/var на выбор → `outputs/merge_data_analysis.json` |
| `scripts/evidence_integration_analysis.py` | Orchestrates preprocessing + models + bias / plots |
| `scripts/advanced_analysis.py` | GEE, psychometric curves, leaky integrator, calibration |
| `main.py` | CLI entrypoint |
| `tests/` | Lightweight import / path checks |

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
# optional: pip install -e ".[dev]"
```

## Running

From the repository root:

```bash
python main.py              # full evidence pipeline (slow: GridSearch + TensorFlow)
python main.py advanced     # secondary analyses
python main.py all          # both
python -u main.py           # unbuffered stdout
python -m scripts.merge_data   # unified_dataset.csv (stats) + merge_data_analysis.json
```

Equivalent modules:

```bash
python -m scripts.evidence_integration_analysis
python -m scripts.advanced_analysis
```

## Tests

```bash
pytest
```

## Four analysis targets (no fourth raw CSV)

The pipeline always runs on **four logical datasets**:

1. `data_lab` — `datasets/data_lab.csv`
2. `data_cons` — `datasets/dataConsA.csv`
3. `data_framing` — `datasets/dataHighFramingA.csv`
4. **`combined`** — строится при **`python main.py`**: единая схема `L1`…`R8` → **`preprocessing/combined/unified_dataset.csv`**. Отдельно **`python -m scripts.merge_data`** пишет **`datasets/unified_dataset.csv`** (только mean/var/diff по последовательностям + `source`). Отдельного сырого CSV для combined нет.

Artifacts are **split by dataset** under `preprocessing/<name>/`, `figures/<name>/`, `models/<name>/`, and tabular outputs under `outputs/separate_analysis/<name>/` (three raw-backed datasets) or `outputs/combined/` (merged run). **`outputs/summary.json`** aggregates all four model blocks. **`scripts/paths.py`** is the single source of path constants.

If you delete `preprocessing/`, `figures/`, `models/`, or `outputs/` (or only parts of them), the next `python main.py` run recreates directories via `os.makedirs` and **regenerates** artifacts for each stage.

## Pipeline summary

1. **Normative benchmark** — EV of left vs right sequences, `is_correct`.
2. **Complexity** — `complexity_diff`, `complexity_sd`, combined score.
3. **Pair features** — spread / min / max of within-trial differences, winner balance, entropy.
4. **Models** — logistic regression, random forest, XGBoost, LSTM, RNN; stratified train/test; grid search + CV with 95% CIs where applicable.
5. **Bias discovery (lab)** — RF residuals, KMeans, t-SNE, economic loss plots.
6. **Combined** — Stacked unified table and the same model stack on merged data.

Dependencies are listed in `pyproject.toml` (pandas, scikit-learn, TensorFlow, xgboost, statsmodels, etc.).
