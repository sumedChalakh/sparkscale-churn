import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import NumericType
from py4j.protocol import Py4JJavaError

# Config
DATA = os.getenv("WEEK4_DATA", "data/features/churn_features.parquet")
MDL_DIR = os.getenv("WEEK4_MODELS", "models")
OUT_DIR = os.getenv("WEEK4_OUT", "week4_output")
SAMPLE = os.getenv("WEEK4_SAMPLE", "0.10")
SAMPLE = None if SAMPLE.lower() == "none" else float(SAMPLE)
SEED = int(os.getenv("WEEK4_SEED", "42"))

os.makedirs(OUT_DIR, exist_ok=True)

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def load_parquet_windows_safe(spark_session, parquet_path):
    try:
        return spark_session.read.parquet(parquet_path)
    except Py4JJavaError as err:
        if "NativeIO$Windows.access0" not in str(err):
            raise

        part_files = sorted(str(p) for p in Path(parquet_path).glob("part-*.parquet"))
        if not part_files:
            raise

        print("Detected Windows NativeIO issue. Falling back to explicit parquet part files...")
        return spark_session.read.parquet(*part_files)

spark = (
    SparkSession.builder
    .appName("SparkScale-W4-Eval")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "32")
    .config("spark.hadoop.io.native.lib.available", "false")
    .config("spark.driver.extraJavaOptions", "-Dhadoop.native.lib=false")
    .config("spark.executor.extraJavaOptions", "-Dhadoop.native.lib=false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

print("Spark ready")

# Load data
if not Path(DATA).exists():
    raise FileNotFoundError(f"Data path not found: {DATA}")

df = load_parquet_windows_safe(spark, DATA)
if SAMPLE is not None:
    df = df.sample(False, SAMPLE, seed=SEED)
df.cache()
n = df.count()
print(f"Data loaded: {n:,} rows")

if "label" not in df.columns:
    if "Churn" in df.columns:
        df = df.withColumn("label", col("Churn").cast("double"))
    else:
        raise ValueError("Expected either 'label' or 'Churn' column in input data.")

MODELS = ["LR", "RF", "GBT"]


def build_fallback_training_df(source_df):
    if "features" in source_df.columns:
        return source_df.select("features", "label")

    feature_cols = [
        field.name for field in source_df.schema.fields
        if field.name != "label" and isinstance(field.dataType, NumericType)
    ]
    if not feature_cols:
        raise ValueError("No usable numeric feature columns found for fallback training.")

    from pyspark.ml.feature import VectorAssembler
    va = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    return va.transform(source_df).select("features", "label")


def fallback_train_model(model_name, source_df):
    train_df, _ = source_df.randomSplit([0.8, 0.2], seed=SEED)
    if model_name == "LR":
        clf = LogisticRegression(featuresCol="features", labelCol="label", maxIter=40)
    elif model_name == "RF":
        clf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=20, maxDepth=6)
    else:
        clf = GBTClassifier(featuresCol="features", labelCol="label", maxIter=20, maxDepth=5, stepSize=0.1)

    model = Pipeline(stages=[clf]).fit(train_df)
    return model

# Load models and predictions
preds = {}
fitted_models = {}
for m in MODELS:
    path = Path(MDL_DIR) / f"{m}_best"
    model = None

    if path.exists():
        try:
            model = PipelineModel.load(str(path))
        except Py4JJavaError as err:
            if "NativeIO$Windows.access0" in str(err):
                print(f"{m} model load hit Windows NativeIO issue, using fallback training.")
            else:
                raise
    else:
        print(f"{m} saved model not found at {path}, using fallback training.")

    if model is None:
        fallback_df = build_fallback_training_df(df)
        model = fallback_train_model(m, fallback_df)

    fitted_models[m] = model
    out_df = model.transform(df)
    preds[m] = out_df.select(
        col("label").cast("double").alias("label"),
        col("prediction").cast("double").alias("prediction"),
        col("probability").alias("probability"),
    ).cache()
    print(f"{m} predictions ready")

print("\nPer-class metrics")

mc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
bin_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="probability")

results = {}
for m in MODELS:
    p = preds[m]
    auc = bin_eval.evaluate(p, {bin_eval.metricName: "areaUnderROC"})
    f1 = mc_eval.evaluate(p, {mc_eval.metricName: "f1"})
    acc = mc_eval.evaluate(p, {mc_eval.metricName: "accuracy"})

    counts = {
        (int(r["label"]), int(r["prediction"])): int(r["count"])
        for r in p.groupBy("label", "prediction").count().collect()
    }
    tn = counts.get((0, 0), 0)
    fp = counts.get((0, 1), 0)
    fn = counts.get((1, 0), 0)
    tp = counts.get((1, 1), 0)
    cm = np.array([[tn, fp], [fn, tp]], dtype=int)

    prec0 = tn / (tn + fn) if (tn + fn) else 0.0
    rec0 = tn / (tn + fp) if (tn + fp) else 0.0
    f1_0 = (2 * prec0 * rec0 / (prec0 + rec0)) if (prec0 + rec0) else 0.0

    prec1 = tp / (tp + fp) if (tp + fp) else 0.0
    rec1 = tp / (tp + fn) if (tp + fn) else 0.0
    f1_1 = (2 * prec1 * rec1 / (prec1 + rec1)) if (prec1 + rec1) else 0.0

    results[m] = {
        "auc": auc,
        "f1": f1,
        "acc": acc,
        "prec0": prec0,
        "rec0": rec0,
        "f1_0": f1_0,
        "prec1": prec1,
        "rec1": rec1,
        "f1_1": f1_1,
        "cm": cm,
    }

    print(f"\n{m}:")
    print(f"  AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}")
    print(f"  Class 0  Prec={prec0:.4f}  Rec={rec0:.4f}  F1={f1_0:.4f}")
    print(f"  Class 1  Prec={prec1:.4f}  Rec={rec1:.4f}  F1={f1_1:.4f}")
    print(f"  CM:\n    TN={cm[0,0]}  FP={cm[0,1]}\n    FN={cm[1,0]}  TP={cm[1,1]}")

print("\nComputing ROC / PR curves")


def get_roc_pr(spark_df):
    def trapezoid_area(y_vals, x_vals):
        if len(y_vals) < 2 or len(x_vals) < 2:
            return 0.0
        return float(np.sum((x_vals[1:] - x_vals[:-1]) * (y_vals[1:] + y_vals[:-1]) * 0.5))

    pdf = spark_df.select(
        col("label").cast("double").alias("label"),
        vector_to_array(col("probability"))[1].cast("double").alias("score"),
    ).toPandas()

    y = pdf["label"].to_numpy(dtype=float)
    s = pdf["score"].to_numpy(dtype=float)

    order = np.argsort(-s)
    y_sorted = y[order]
    s_sorted = s[order]

    p_total = max((y == 1).sum(), 1)
    n_total = max((y == 0).sum(), 1)

    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)

    distinct = np.r_[True, s_sorted[1:] != s_sorted[:-1]] if len(s_sorted) > 0 else np.array([], dtype=bool)
    tp = tp_cum[distinct] if len(tp_cum) else np.array([0])
    fp = fp_cum[distinct] if len(fp_cum) else np.array([0])

    tpr = np.r_[0.0, tp / p_total, 1.0]
    fpr = np.r_[0.0, fp / n_total, 1.0]
    roc_auc = trapezoid_area(tpr, fpr)

    rec = np.r_[0.0, tp / p_total] if len(tp) else np.array([0.0])
    prec = np.r_[1.0, tp / np.maximum(tp + fp, 1)] if len(tp) else np.array([1.0])
    ap = trapezoid_area(prec, rec)

    return fpr, tpr, roc_auc, prec, rec, ap


curves = {}
for m in MODELS:
    fpr, tpr, roc_auc, prec, rec, ap = get_roc_pr(preds[m])
    curves[m] = {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "prec": prec,
        "rec": rec,
        "ap": ap,
    }
    print(f"  {m} -> AUC={roc_auc:.4f}  AP={ap:.4f}")

colors = {"LR": "#4C72B0", "RF": "#DD8452", "GBT": "#55A868"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("SparkScale Churn - ROC and PR Curves (Week 4)", fontsize=13, fontweight="bold")

ax1, ax2 = axes
for m in MODELS:
    c = curves[m]
    ax1.plot(c["fpr"], c["tpr"], color=colors[m], lw=2, label=f"{m} (AUC={c['roc_auc']:.4f})")
    ax2.plot(c["rec"], c["prec"], color=colors[m], lw=2, label=f"{m} (AP={c['ap']:.4f})")

ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
ax1.set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
ax1.legend(loc="lower right", fontsize=9)

ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
ax2.legend(loc="upper right", fontsize=9)

for ax in axes:
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

plt.tight_layout()
roc_path = Path(OUT_DIR) / "roc_pr_curves.png"
plt.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {roc_path}")

print("\nFeature Importance")

BASE_FEAT_NAMES = [
    "tenure", "MonthlyCharges", "TotalCharges", "service_count", "charge_ratio", "charge_per_month",
    "complaint_proxy", "Contract_1yr", "Contract_2yr", "InternetService_Fiber", "InternetService_No",
    "PaymentMethod_CC", "PaymentMethod_Check", "PaymentMethod_Mail", "OnlineSecurity_Yes",
    "TechSupport_Yes", "PaperlessBilling_Yes", "SeniorCitizen", "Partner_Yes", "Dependents_Yes",
]


def get_feat_imp(model_name):
    model = fitted_models.get(model_name)
    if model is None:
        path = Path(MDL_DIR) / f"{model_name}_best"
        model = PipelineModel.load(str(path))
    clf = model.stages[-1]
    return clf.featureImportances.toArray()


gbt_imp_raw = get_feat_imp("GBT")
rf_imp_raw = get_feat_imp("RF")

n_feats = max(len(BASE_FEAT_NAMES), len(gbt_imp_raw), len(rf_imp_raw))
feat_names = BASE_FEAT_NAMES + [f"feature_{i}" for i in range(len(BASE_FEAT_NAMES), n_feats)]

gbt_imp = np.pad(gbt_imp_raw, (0, n_feats - len(gbt_imp_raw)))
rf_imp = np.pad(rf_imp_raw, (0, n_feats - len(rf_imp_raw)))

idx = np.argsort(gbt_imp)
names_s = [feat_names[i] for i in idx]
gbt_s = gbt_imp[idx]
rf_s = rf_imp[idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle("Feature Importance - GBT vs RF", fontsize=13, fontweight="bold")

for ax, imp, title, color in [
    (axes[0], gbt_s, "GBT Feature Importance", "#55A868"),
    (axes[1], rf_s, "RF Feature Importance", "#DD8452"),
]:
    bars = ax.barh(names_s, imp, color=color, alpha=0.82, edgecolor="white")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, imp):
        ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=7)

plt.tight_layout()
fi_path = Path(OUT_DIR) / "feature_importance.png"
plt.savefig(fi_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {fi_path}")

for i in reversed(idx):
    print(f"  {feat_names[i]:<25} GBT={gbt_imp[i]:.4f}  RF={rf_imp[i]:.4f}")

print("\nThreshold Tuning - GBT")

gbt_pdf = preds["GBT"].toPandas()
gbt_pdf["score"] = gbt_pdf["probability"].apply(lambda v: float(v[1]))
y_true = gbt_pdf["label"].values

thresholds = np.arange(0.20, 0.81, 0.05)
th_rows = []
for t in thresholds:
    y_pred = (gbt_pdf["score"] >= t).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    acc = (tp + tn) / len(y_true)
    th_rows.append({
        "threshold": round(t, 2),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    })

th_df = pd.DataFrame(th_rows)
best_row = th_df.loc[th_df["f1"].idxmax()]
print(th_df[["threshold", "precision", "recall", "f1", "accuracy"]].to_string(index=False))
print(
    f"\nBest threshold by F1 -> {best_row['threshold']} "
    f"(P={best_row['precision']:.4f}, R={best_row['recall']:.4f}, F1={best_row['f1']:.4f})"
)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("GBT Threshold Tuning", fontsize=13, fontweight="bold")

ax1 = axes[0]
ax1.plot(th_df["threshold"], th_df["precision"], "o-", label="Precision", color="#4C72B0")
ax1.plot(th_df["threshold"], th_df["recall"], "s-", label="Recall", color="#DD8452")
ax1.plot(th_df["threshold"], th_df["f1"], "^-", label="F1", color="#55A868", lw=2)
ax1.axvline(best_row["threshold"], color="red", ls="--", lw=1, label=f"Best t={best_row['threshold']}")
ax1.set(xlabel="Threshold", ylabel="Score", title="Precision / Recall / F1")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(th_df["threshold"], th_df["accuracy"], "D-", color="#C44E52", lw=2)
ax2.axvline(best_row["threshold"], color="red", ls="--", lw=1)
ax2.set(xlabel="Threshold", ylabel="Accuracy", title="Accuracy vs Threshold")
ax2.grid(alpha=0.3)

plt.tight_layout()
th_path = Path(OUT_DIR) / "threshold_tuning.png"
plt.savefig(th_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {th_path}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")

for ax, m in zip(axes, MODELS):
    cm = results[m]["cm"]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(m, fontsize=11)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=11,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
cm_path = Path(OUT_DIR) / "confusion_matrices.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {cm_path}")

summary = {}
for m in MODELS:
    r = results[m]
    summary[m] = {
        "auc": round(r["auc"], 4),
        "f1": round(r["f1"], 4),
        "accuracy": round(r["acc"], 4),
        "class_0": {
            "precision": round(r["prec0"], 4),
            "recall": round(r["rec0"], 4),
            "f1": round(r["f1_0"], 4),
        },
        "class_1": {
            "precision": round(r["prec1"], 4),
            "recall": round(r["rec1"], 4),
            "f1": round(r["f1_1"], 4),
        },
        "confusion_matrix": r["cm"].tolist(),
    }

best_model = max(MODELS, key=lambda m: results[m]["auc"])
summary["best_model"] = best_model
summary["best_threshold"] = float(best_row["threshold"])
summary["threshold_f1"] = round(float(best_row["f1"]), 4)
summary["top_features"] = [feat_names[i] for i in reversed(idx[-5:])]

json_path = Path(OUT_DIR) / "summary.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary JSON: {json_path}")

print("\n" + "=" * 62)
print("  SPARKSCALE CHURN - WEEK 4 FINAL REPORT")
print("=" * 62)
print(f"  Dataset rows used : {n:,}")
print(f"  {'Model':<8} {'AUC':>8} {'F1':>8} {'Acc':>8} {'P(churn)':>10} {'R(churn)':>10}")
print(f"  {'-' * 58}")
for m in MODELS:
    r = results[m]
    star = " *" if m == best_model else ""
    print(
        f"  {m:<8} {r['auc']:>8.4f} {r['f1']:>8.4f} {r['acc']:>8.4f} "
        f"{r['prec1']:>10.4f} {r['rec1']:>10.4f}{star}"
    )
print(f"\n  Best model        : {best_model} (AUC={results[best_model]['auc']:.4f})")
print(
    f"  Best threshold    : {best_row['threshold']} (F1={best_row['f1']:.4f}  "
    f"P={best_row['precision']:.4f}  R={best_row['recall']:.4f})"
)
print(f"  Top-5 features    : {', '.join(summary['top_features'])}")
print("=" * 62)
print(f"\n  Outputs saved to ./{OUT_DIR}/")
print("    roc_pr_curves.png")
print("    feature_importance.png")
print("    threshold_tuning.png")
print("    confusion_matrices.png")
print("    summary.json")
print("=" * 62)

spark.stop()
