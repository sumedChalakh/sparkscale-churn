import os
import sys
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
from pyspark.sql.types import NumericType
from py4j.protocol import Py4JJavaError

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

FAST_MODE = os.getenv("WEEK3_FAST", "0") == "1"
SAMPLE_FRACTION = float(os.getenv("WEEK3_SAMPLE_FRACTION", "0.2")) if FAST_MODE else 1.0


def get_models_and_folds(is_fast_mode):
    if is_fast_mode:
        models = {
            "LR": (
                LogisticRegression(featuresCol="features", labelCol="label", maxIter=50),
                ParamGridBuilder()
                    .addGrid(LogisticRegression().regParam, [0.1])
                    .addGrid(LogisticRegression().elasticNetParam, [0.0])
                    .build()
            ),
            "RF": (
                RandomForestClassifier(featuresCol="features", labelCol="label"),
                ParamGridBuilder()
                    .addGrid(RandomForestClassifier().numTrees, [20])
                    .addGrid(RandomForestClassifier().maxDepth, [5])
                    .build()
            ),
            "GBT": (
                GBTClassifier(featuresCol="features", labelCol="label", maxIter=20),
                ParamGridBuilder()
                    .addGrid(GBTClassifier().maxDepth, [5])
                    .addGrid(GBTClassifier().stepSize, [0.1])
                    .build()
            ),
        }
        return models, 2

    models = {
        "LR": (
            LogisticRegression(featuresCol="features", labelCol="label", maxIter=100),
            ParamGridBuilder()
                .addGrid(LogisticRegression().regParam, [0.01, 0.1])
                .addGrid(LogisticRegression().elasticNetParam, [0.0, 0.5])
                .build()
        ),
        "RF": (
            RandomForestClassifier(featuresCol="features", labelCol="label"),
            ParamGridBuilder()
                .addGrid(RandomForestClassifier().numTrees, [50, 100])
                .addGrid(RandomForestClassifier().maxDepth, [5, 10])
                .build()
        ),
        "GBT": (
            GBTClassifier(featuresCol="features", labelCol="label", maxIter=50),
            ParamGridBuilder()
                .addGrid(GBTClassifier().maxDepth, [5, 8])
                .addGrid(GBTClassifier().stepSize, [0.1, 0.05])
                .build()
        ),
    }
    return models, 3

spark = SparkSession.builder \
    .appName("SparkScale_Week3_ML") \
    .config("spark.sql.shuffle.partitions", "16") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .config("spark.driver.extraJavaOptions", "-Dhadoop.native.lib=false") \
    .config("spark.executor.extraJavaOptions", "-Dhadoop.native.lib=false") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


def load_feature_data(spark_session, parquet_dir):
    try:
        return spark_session.read.parquet(parquet_dir)
    except Py4JJavaError as err:
        if "NativeIO$Windows.access0" not in str(err):
            raise

        part_files = sorted(str(p) for p in Path(parquet_dir).glob("part-*.parquet"))
        if not part_files:
            raise

        print("Detected Windows NativeIO issue. Falling back to explicit parquet part files...")
        return spark_session.read.parquet(*part_files)

# ── Load ──────────────────────────────────────────────────────────────────────
df = load_feature_data(spark, "data/features/churn_features.parquet")
if "Churn" in df.columns:
    df = df.withColumn("label", F.col("Churn").cast("double")).drop("Churn")
elif "label" in df.columns:
    df = df.withColumn("label", F.col("label").cast("double"))
else:
    raise ValueError("Input data must include either 'Churn' or 'label' column.")

if FAST_MODE:
    print(f"FAST validation mode enabled. Sampling {SAMPLE_FRACTION:.0%} of rows.")
    df = df.sample(withReplacement=False, fraction=SAMPLE_FRACTION, seed=42)

train, test = df.randomSplit([0.8, 0.2], seed=42)
train.persist(StorageLevel.DISK_ONLY)
test.persist(StorageLevel.DISK_ONLY)

use_precomputed_features = "features" in df.columns
va = None

if not use_precomputed_features:
    feat_cols = [
        field.name for field in df.schema.fields
        if field.name != "label" and isinstance(field.dataType, NumericType)
    ]
    if not feat_cols:
        raise ValueError("No numeric feature columns found after preprocessing.")
    va = VectorAssembler(inputCols=feat_cols, outputCol="features", handleInvalid="skip")

# ── Evaluators ────────────────────────────────────────────────────────────────
auc_eval = BinaryClassificationEvaluator(metricName="areaUnderROC")
f1_eval  = MulticlassClassificationEvaluator(metricName="f1")
acc_eval = MulticlassClassificationEvaluator(metricName="accuracy")

# ── Models + Grids ────────────────────────────────────────────────────────────
models, num_folds = get_models_and_folds(FAST_MODE)

results = {}

for name, (clf, grid) in models.items():
    print(f"\n{'='*50}\n🔁 Training {name} with CrossValidator ({num_folds}-fold)\n{'='*50}")

    stages = [clf] if use_precomputed_features else [va, clf]
    pipe = Pipeline(stages=stages)

    cv = CrossValidator(
        estimator=pipe,
        estimatorParamMaps=grid,
        evaluator=auc_eval,
        numFolds=num_folds,
        seed=42,
        parallelism=1
    )

    cv_model = cv.fit(train)
    preds    = cv_model.transform(test)

    auc = auc_eval.evaluate(preds)
    f1  = f1_eval.evaluate(preds)
    acc = acc_eval.evaluate(preds)

    results[name] = {"AUC": round(auc, 4), "F1": round(f1, 4), "Acc": round(acc, 4)}
    print(f"  AUC={auc:.4f} | F1={f1:.4f} | Acc={acc:.4f}")

    # On Windows without winutils/HADOOP_HOME, model writes can fail; keep metrics anyway.
    try:
        cv_model.bestModel.write().overwrite().save(f"models/{name}_best")
    except Py4JJavaError as err:
        if "HADOOP_HOME and hadoop.home.dir are unset" in str(err):
            print("  Skipping model save on Windows (missing HADOOP_HOME/winutils).")
        else:
            raise

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n\n📊 WEEK 3 RESULTS SUMMARY")
print(f"{'Model':<8} {'AUC':>8} {'F1':>8} {'Acc':>8}")
print("-" * 35)
for n, m in results.items():
    print(f"{n:<8} {m['AUC']:>8} {m['F1']:>8} {m['Acc']:>8}")

best = max(results, key=lambda x: results[x]["AUC"])
print(f"\n🏆 Best model by AUC: {best} → {results[best]}")

spark.stop()