# SparkScale Churn

## Status

**Project In Progress**

This repository is actively being developed. Structure, pipelines, and outputs may change as the project evolves.

## Overview

SparkScale Churn is a PySpark-based churn prediction project with:
- data ingestion and ETL
- feature engineering
- model training and evaluation (LR, RF, GBT)
- Docker-based Spark workflow support

## Repository Structure

- `src/`: pipeline and module source files
- `data/raw/`: source dataset
- `data/features/`: generated feature outputs
- `data/scaled/`: generated scaled datasets
- `models/`: generated trained model artifacts

## Note About Ignored Artifacts

Large generated artifacts are intentionally excluded from Git to keep the repository lightweight and push-friendly:
- `data/features/churn_features.parquet/`
- `data/scaled/`
- `models/`
- `*.crc`

## How To Regenerate Excluded Artifacts

Run the project pipelines to recreate generated files locally.

Example flow:

1. Run ETL:
   - `python src/etl.py`
2. Run feature engineering:
   - `python src/feature_engineering.py`
3. Run ML training pipeline:
   - `python src/week3_ml_pipeline.py`

After running these steps, generated outputs will be available in:
- `data/features/`
- `data/scaled/`
- `models/`

## Current Focus

- improving pipeline reliability
- tuning model performance
- documenting reproducible local and Docker workflows
