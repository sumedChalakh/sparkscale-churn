# SparkScale Churn

**Zaalima Development — Q4 ML Assignment | Project 4**

End-to-end distributed churn prediction pipeline built on PySpark 3.5.3. Covers synthetic data generation at 2M rows, Spark SQL feature engineering, distributed MLlib model training (LR / RF / GBT), and a full evaluation suite with ROC/PR curves, confusion matrices, feature importance, and threshold tuning — all running locally via Docker Spark cluster.

---

## Results

### Model Comparison (Week 4 evaluation on 10% sample ≈ 200K rows)

| Model | AUC | F1 | Accuracy | Precision (churn) | Recall (churn) |
|-------|--------|---------|---------|--------------------|----------------|
| LR | 0.8480 | 0.7982 | 0.8055 | 0.6684 | 0.5356 |
| RF | 0.8460 | 0.7924 | 0.8045 | 0.6898 | 0.4836 |
| **GBT** ★ | **0.8759** | **0.8145** | **0.8205** | **0.6989** | **0.5730** |

### Threshold Tuning — GBT

Default threshold (0.5) gives F1 = 0.63. Tuning to **0.30** yields:

| Threshold | Precision | Recall | F1 | Accuracy |
|-----------|-----------|--------|----|----------|
| 0.30 ★ | 0.587 | 0.785 | **0.671** | 0.795 |
| 0.50 | 0.699 | 0.573 | 0.630 | 0.821 |

At threshold 0.30, the model catches **78.5% of actual churners** — more valuable for telco retention campaigns where false negatives (missed churners) cost more than false positives.

### Top Features (GBT)

`charge_ratio` › `MonthlyCharges` › `feature_46` › `service_count` › `TotalCharges`

Charge-related signals dominate, confirming that pricing pressure is the primary churn driver in this dataset.

---

## Stack

- **PySpark 3.5.3** — distributed ETL, feature engineering, MLlib training
- **Python 3.8** — local execution via conda env (`first`)
- **Docker** — `apache/spark:3.5.3` cluster (1 master + 2 workers) + Jupyter
- **scikit-learn** — ROC/PR curve computation from probability scores
- **matplotlib / pandas / pyarrow** — evaluation plots and parquet I/O
- **Dataset** — IBM Telco Churn, synthetically scaled to 2M rows

---

## Repository Structure

```
sparkscale-churn/
├── src/
│   ├── etl.py                    # Week 1 — raw data ingest, schema validation, row checks
│   ├── feature_engineering.py    # Week 2 — Spark SQL transforms, DAG/query-plan inspection
│   └── week3_ml_pipeline.py      # modular Week 3 pipeline (imported by top-level runner)
│
├── data/
│   ├── raw/                      # source CSV (IBM Telco Churn, 2M rows synthetic)
│   └── features/
│       └── churn_features.parquet  # engineered feature store (Week 2 output)
│
├── week4_output/
│   ├── roc_pr_curves.png         # ROC + Precision-Recall curves, all 3 models
│   ├── feature_importance.png    # GBT vs RF side-by-side importance chart
│   ├── threshold_tuning.png      # P/R/F1/Accuracy vs threshold (GBT)
│   ├── confusion_matrices.png    # 3-panel heatmap (LR, RF, GBT)
│   └── summary.json              # all metrics, best threshold, top features
│
├── week3_ml_pipeline.py          # Week 3 — LR / RF / GBT training with CrossValidator
├── week4_evaluation.py           # Week 4 — full evaluation and reporting script
├── week4_batch_predict.py        # Week 4 — production-style monthly batch scoring job
├── docker-compose.yml            # Spark cluster: master + 2 workers + Jupyter
├── requirements.txt              # reproducible Python deps
└── .gitignore
```

---

## How to Run

### 0. Environment setup

```bash
# Install dependencies into the conda env
C:/Users/ACER/anaconda3/envs/first/python.exe -m pip install -r requirements.txt
```

### 1. Start Spark cluster (optional — local mode works without it)

```bash
docker-compose up -d
# Spark UI → http://localhost:8080
# Jupyter  → http://localhost:8888
```

### 2. Week 3 — model training

```powershell
# Fast validation (10% sample, ~5 min)
$env:WEEK3_FAST='1'; $env:WEEK3_SAMPLE_FRACTION='0.1'
C:/Users/ACER/anaconda3/envs/first/python.exe week3_ml_pipeline.py

# Full 2M row run
C:/Users/ACER/anaconda3/envs/first/python.exe week3_ml_pipeline.py
```

### 3. Week 4 — evaluation

```powershell
C:/Users/ACER/anaconda3/envs/first/python.exe week4_evaluation.py
# Outputs written to week4_output/
```

### 4. Week 4 — batch prediction job

```powershell
# Default (parquet output)
$env:BATCH_INPUT='data/features/churn_features.parquet'
$env:BATCH_MODEL_ROOT='models'
$env:BATCH_MODEL_NAME='GBT'
$env:BATCH_OUTPUT='batch_output/churn_predictions.parquet'
C:/Users/ACER/anaconda3/envs/first/python.exe week4_batch_predict.py

# CSV output
$env:BATCH_OUTPUT_FORMAT='csv'
C:/Users/ACER/anaconda3/envs/first/python.exe week4_batch_predict.py
```

---

## Week-by-Week Breakdown

### Week 1 — Infrastructure & ETL
- Docker Spark cluster: `apache/spark:3.5.3` image, 1 master + 2 workers (2G/2 cores each), Jupyter sidecar
- Synthetic dataset generation: IBM Telco Churn scaled to 2M rows
- Distributed ETL with schema validation and row count checks via `src/etl.py`

### Week 2 — Feature Engineering
- Spark SQL transforms producing engineered columns: `tenure_bucket`, `complaint_proxy`, `service_count`, `charge_ratio`, `charge_per_month`
- DAG and query plan inspection to validate Spark execution
- Output: `data/features/churn_features.parquet`

### Week 3 — Distributed ML Training
- Three models via `spark.ml` Pipeline + CrossValidator
- Full mode uses **3-fold CV**; fast mode uses **2-fold CV**
- Fast mode: `WEEK3_FAST=1` uses a configurable sample fraction (`WEEK3_SAMPLE_FRACTION`, default 0.2)
- Saved models are written to `models/LR_best`, `models/RF_best`, and `models/GBT_best`

### Week 4 — Evaluation & Reporting
- Per-class Precision / Recall / F1 for all 3 models
- ROC curve (sklearn `roc_curve`) + PR curve (`precision_recall_curve`) with sklearn, bypassing PySpark's sparse native implementation
- GBT and RF feature importances from `clf.featureImportances`
- Threshold sweep (0.20 → 0.80, step 0.05) on GBT — optimal at 0.30
- Confusion matrix heatmaps (3-panel)
- `summary.json` consolidating all metrics, best threshold, and top-5 features
- `week4_batch_predict.py` — reusable monthly scoring job with env-var config

---

## Windows Compatibility

This codebase runs on local Windows (no HADOOP_HOME / winutils) with built-in fallbacks:

- **Parquet reads** — falls back to explicit part-file listing when `NativeIO.Windows.access0` throws `UnsatisfiedLinkError`
- **Model loading** — falls back to inline retraining when `sc.textFile` on model metadata fails due to the same native issue
- **WARN messages** — `Did not find winutils.exe` and `NativeCodeLoader` warnings are expected and do not affect output correctness

---

## Evaluation Outputs

All files written to `week4_output/` after running `week4_evaluation.py`:

| File | Description |
|------|-------------|
| `roc_pr_curves.png` | ROC + PR side-by-side, all 3 models with AUC / AP labels |
| `feature_importance.png` | GBT vs RF horizontal bar chart, sorted by GBT importance |
| `threshold_tuning.png` | P/R/F1/Accuracy vs threshold, optimal marked at 0.30 |
| `confusion_matrices.png` | 3-panel heatmap with TN/FP/FN/TP counts |
| `summary.json` | Full metrics dict, best model, best threshold, top-5 features |

---

## Notes

- Large generated artifacts (`data/features/`, scaled parquet directories) are excluded from Git via `.gitignore`. Recreate by rerunning Week 2 and Week 3 scripts.
- RF numbers vary slightly across runs due to fallback retraining on Windows (no fixed seed propagated through CrossValidator on local mode).
- Current default sample behavior in evaluation is controlled by env var `WEEK4_SAMPLE` (default `0.10`; set to `none` for full-data evaluation).
