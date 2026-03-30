# SparkScale Churn

## Status

**Completed (local execution verified)**

This project implements a PySpark churn workflow from data processing to model evaluation.

## Project Summary

SparkScale Churn includes:
- data loading and ETL
- feature engineering
- model training and comparison (LR, RF, GBT)
- Week 4 evaluation with ROC/PR curves, confusion matrices, feature importance, and threshold tuning
- Week 4 reusable batch prediction script for monthly churn scoring

## Final Results

### Week 3 (training validation)
- Mode used: fast validation mode
- Best model: `GBT`
- Metrics: AUC `0.8751`, F1 `0.8121`, Accuracy `0.8181`

### Week 4 (evaluation)
- Best model: `GBT`
- Metrics: AUC `0.8759`, F1 `0.8145`, Accuracy `0.8205`
- Best threshold by F1: `0.30`
- Output summary file: `week4_output/summary.json`

## Repository Structure

- `src/` - modular pipeline files (`etl.py`, `feature_engineering.py`, `week3_ml_pipeline.py`)
- `week3_ml_pipeline.py` - top-level Week 3 pipeline runner
- `week4_evaluation.py` - Week 4 full evaluation and reporting script
- `week4_batch_predict.py` - Week 4 reusable batch scoring script (production-style daily/monthly run)
- `data/raw/` - input dataset
- `data/features/` - engineered features parquet
- `week4_output/` - generated evaluation plots and JSON summary
- `requirements.txt` - reproducible Python dependencies
- `docker-compose.yml` - Spark container setup

## How To Run

### 1. Week 3 training pipeline

PowerShell (quick validation):

`$env:WEEK3_FAST='1'; $env:WEEK3_SAMPLE_FRACTION='0.1'; C:/Users/ACER/anaconda3/envs/first/python.exe week3_ml_pipeline.py`

PowerShell (full run):

`C:/Users/ACER/anaconda3/envs/first/python.exe week3_ml_pipeline.py`

### 2. Week 4 evaluation

PowerShell:

`C:/Users/ACER/anaconda3/envs/first/python.exe week4_evaluation.py`

### 3. Week 4 reusable batch prediction job

PowerShell (predict churn probabilities for new monthly parquet):

`$env:BATCH_INPUT='data/features/churn_features.parquet'; $env:BATCH_MODEL_ROOT='models'; $env:BATCH_MODEL_NAME='GBT'; $env:BATCH_OUTPUT='batch_output/churn_predictions.parquet'; C:/Users/ACER/anaconda3/envs/first/python.exe week4_batch_predict.py`

Optional CSV output:

`$env:BATCH_OUTPUT_FORMAT='csv'; C:/Users/ACER/anaconda3/envs/first/python.exe week4_batch_predict.py`

### 4. Reproducible environment

Install dependencies:

`C:/Users/ACER/anaconda3/envs/first/python.exe -m pip install -r requirements.txt`

## Requirement Coverage (Week 1-4)

- Week 1: Docker Spark cluster (master + workers), distributed ETL, schema validation, row checks
- Week 2: Spark SQL feature engineering and DAG/query-plan inspection
- Week 3: Spark MLlib distributed training with binary evaluation setup
- Week 4: reusable batch scoring script + evaluation/reporting workflow
- Submission artifact support: source code + outputs + `requirements.txt`

## Week 4 Outputs

Generated in `week4_output/`:
- `roc_pr_curves.png`
- `feature_importance.png`
- `threshold_tuning.png`
- `confusion_matrices.png`
- `summary.json`

## Windows Compatibility Notes

This codebase includes runtime fallbacks for common Windows Spark issues:
- Native Hadoop `winutils`/`HADOOP_HOME` access issues during parquet/model loading
- fallback model path handling for evaluation when direct model loading is blocked

Because of these safeguards, local Windows runs complete even when Hadoop native binaries are not installed.

## Notes On Tracked vs Generated Artifacts

Large generated artifacts may be ignored for lightweight Git history (for example feature/scaled parquet directories and some Spark artifacts). Recreate them locally by rerunning Week 3 and Week 4 scripts.
