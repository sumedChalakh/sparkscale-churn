from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F

spark = SparkSession.builder \
    .appName("SparkScale_Week3_ML") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ── Load ──────────────────────────────────────────────────────────────────────
df = spark.read.parquet("data/features/churn_features.parquet")
df = df.withColumn("label", F.col("Churn").cast("double")).drop("Churn")

train, test = df.randomSplit([0.8, 0.2], seed=42)
train.cache()
test.cache()

feat_cols = [c for c in df.columns if c != "label"]
va = VectorAssembler(inputCols=feat_cols, outputCol="features", handleInvalid="skip")

# ── Evaluators ────────────────────────────────────────────────────────────────
auc_eval = BinaryClassificationEvaluator(metricName="areaUnderROC")
f1_eval  = MulticlassClassificationEvaluator(metricName="f1")
acc_eval = MulticlassClassificationEvaluator(metricName="accuracy")

# ── Models + Grids ────────────────────────────────────────────────────────────
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

results = {}

for name, (clf, grid) in models.items():
    print(f"\n{'='*50}\n🔁 Training {name} with CrossValidator (5-fold)\n{'='*50}")

    pipe = Pipeline(stages=[va, clf])

    cv = CrossValidator(
        estimator=pipe,
        estimatorParamMaps=grid,
        evaluator=auc_eval,
        numFolds=5,
        seed=42,
        parallelism=4
    )

    cv_model = cv.fit(train)
    preds    = cv_model.transform(test)

    auc = auc_eval.evaluate(preds)
    f1  = f1_eval.evaluate(preds)
    acc = acc_eval.evaluate(preds)

    results[name] = {"AUC": round(auc, 4), "F1": round(f1, 4), "Acc": round(acc, 4)}
    print(f"  AUC={auc:.4f} | F1={f1:.4f} | Acc={acc:.4f}")

    # save best model
    cv_model.bestModel.write().overwrite().save(f"models/{name}_best")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n\n📊 WEEK 3 RESULTS SUMMARY")
print(f"{'Model':<8} {'AUC':>8} {'F1':>8} {'Acc':>8}")
print("-" * 35)
for n, m in results.items():
    print(f"{n:<8} {m['AUC']:>8} {m['F1']:>8} {m['Acc']:>8}")

best = max(results, key=lambda x: results[x]["AUC"])
print(f"\n🏆 Best model by AUC: {best} → {results[best]}")

spark.stop()