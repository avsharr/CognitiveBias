# Evidence Integration and AI-assisted Bias Discovery

Analysis pipeline for information search and evidence integration in lottery choice tasks.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python evidence_integration_analysis.py
python advanced_analysis.py   # GLMM, psychometric, Leaky Integrator
```

## Structure

| File | Description |
|------|-------------|
| `data_lab.csv` | Main experiment (willingness to pay for information) |
| `dataConsA.csv` | Consistency and confidence |
| `dataHighFramingA.csv` | Scale/framing effect |
| `evidence_integration_analysis.py` | Main pipeline |
| `advanced_analysis.py` | GLMM, psychometric, Leaky Integrator, cluster interpretation |
| `outputs/` | Results (plots, summary.json) |

## Implemented steps

1. **Normative agent**: EV of left/right sequence, `is_correct`
2. **Complexity**: `complexity_diff` (mean difference), `complexity_sd` (standard deviation)
3. **Random Forest**: trained on human and normative, feature importance
4. **Bias Discovery**: residuals, KMeans clustering
5. **Economic effect**: lost utility by complexity bins
6. **Visualizations**: psychometric curve, t-SNE clusters, economic loss
## Checklist

- ✓ Data cleaned (StandardScaler)
- ✓ RF trained on human and normative
- ✓ Confidence intervals via Cross-Validation (±std)
- ✓ Psychometric curves and t-SNE

