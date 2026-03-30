"""
feature_engineering.py — SparkScale Churn | Week 2
Zaalima Development Q4 ML Assignment
"""

import os
import time

# ── Windows: point Spark to winutils so it can write to local FS ──────────────
os.environ["HADOOP_HOME"]      = r"C:\hadoop"
os.environ["hadoop.home.dir"]  = r"C:\hadoop"
os.environ["PATH"]            = r"C:\hadoop\bin;" + os.environ.get("PATH", "")
# ─────────────────────────────────────────────────────────────────────────────

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

# ─── paths ────────────────────────────────────────────────────────────────────
BASE  = r"C:\Users\ACER\Desktop\Project 4\sparkscale-churn"
IN_P  = os.path.join(BASE, "data", "scaled", "telco_churn_scaled.parquet")
OUT_P = os.path.join(BASE, "data", "features", "churn_features.parquet")
DAG_P = os.path.join(BASE, "data", "features", "query_plan.txt")
os.makedirs(os.path.join(BASE, "data", "features"), exist_ok=True)

# ─── spark session ─────────────────────────────────────────────────────────────
def get_spark():
    return (
        SparkSession.builder
        .appName("SparkScale-FeatureEngineering")
        .master("local[*]")
        .config("spark.driver.memory",              "6g")
        .config("spark.driver.maxResultSize",       "2g")
        .config("spark.memory.fraction",            "0.6")
        .config("spark.memory.storageFraction",     "0.3")
        .config("spark.sql.shuffle.partitions",     "8")
        .config("spark.sql.adaptive.enabled",       "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.debug.maxToStringFields","50")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")  
        .getOrCreate()
    )

# ─── 1. load ───────────────────────────────────────────────────────────────────
def load(spark):
    df = spark.read.parquet(IN_P)
    print(f"[load] rows={df.count():,}  cols={len(df.columns)}")
    df.printSchema()
    return df

# ─── 2. spark sql features ─────────────────────────────────────────────────────
def sql_features(spark, df):
    df.createOrReplaceTempView("telco")

    # tenure bucket
    df = spark.sql("""
        SELECT *,
            CASE
                WHEN tenure BETWEEN 0  AND 12 THEN 'new'
                WHEN tenure BETWEEN 13 AND 24 THEN 'growing'
                WHEN tenure BETWEEN 25 AND 48 THEN 'mature'
                ELSE 'loyal'
            END AS tenure_bucket
        FROM telco
    """)
    df.createOrReplaceTempView("telco")

    # avg monthly charge per contract type — only 3 groups so AQE will broadcast this join
    df = spark.sql("""
        SELECT t.*,
               avg_tbl.avg_charge_by_contract
        FROM telco t
        JOIN (
            SELECT Contract,
                   ROUND(AVG(MonthlyCharges), 2) AS avg_charge_by_contract
            FROM telco
            GROUP BY Contract
        ) avg_tbl ON t.Contract = avg_tbl.Contract
    """)
    df.createOrReplaceTempView("telco")

    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    svc_expr = sum(
        F.when(F.col(c).isin("Yes", "1", 1), 1).otherwise(0)
        for c in service_cols
    )

    df = (
        df
        .withColumn("charge_ratio",
            F.round(F.col("MonthlyCharges") / (F.col("avg_charge_by_contract") + 1e-9), 4))
        .withColumn("charge_per_month",
            F.round(
                F.col("TotalCharges") / F.when(F.col("tenure") == 0, 1).otherwise(F.col("tenure")),
                2))
        .withColumn("complaint_proxy",
            F.when((F.col("TechSupport") == "No") & (F.col("OnlineSecurity") == "No"), 1).otherwise(0))
        .withColumn("service_count", svc_expr)
    )

    print("[sql_features] tenure_bucket | avg_charge_by_contract | "
          "charge_ratio | charge_per_month | complaint_proxy | service_count")
    return df

# ─── 3. encode categoricals ────────────────────────────────────────────────────
CAT_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "tenure_bucket"
]

def encode(df):
    idx_cols = [c + "_idx" for c in CAT_COLS]
    ohe_cols = [c + "_ohe" for c in CAT_COLS]
    indexers = [StringIndexer(inputCol=c, outputCol=o, handleInvalid="keep")
                for c, o in zip(CAT_COLS, idx_cols)]
    encoders = [OneHotEncoder(inputCol=i, outputCol=o, dropLast=True)
                for i, o in zip(idx_cols, ohe_cols)]
    df = Pipeline(stages=indexers + encoders).fit(df).transform(df)
    print(f"[encode] {len(CAT_COLS)} cat cols → StringIndexed + OHE")
    return df, ohe_cols

# ─── 4. assemble + SLIM immediately ────────────────────────────────────────────
NUM_COLS  = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "avg_charge_by_contract", "charge_ratio", "charge_per_month",
    "complaint_proxy", "service_count"
]
KEEP_COLS = [
    "customerID", "tenure_bucket", "complaint_proxy",
    "service_count", "charge_ratio", "charge_per_month"
]

def assemble(df, ohe_cols):
    asm = VectorAssembler(
        inputCols=NUM_COLS + ohe_cols,
        outputCol="features",
        handleInvalid="skip"
    )
    df = asm.transform(df)
    df = df.withColumn(
        "label",
        F.when(F.col("Churn").isin("Yes", "1", 1), 1).otherwise(0).cast("double")
    )
    print(f"[assemble] {len(NUM_COLS + ohe_cols)} inputs → features vector | label added")

    # ── KEY FIX: drop all intermediate & OHE cols RIGHT HERE ──────────────────
    # Carrying 40+ wide columns into cache/write caused the OOM.
    # After this select, each row is just: SparseVector + double + 6 slim cols.
    df = df.select(["features", "label"] + KEEP_COLS)
    print(f"[slim] final schema: {len(df.columns)} cols (features, label + {len(KEEP_COLS)} metadata)")
    return df

# ─── 5. dag export (explain only — no count, no cache) ─────────────────────────
def export_dag(df):
    print("\n" + "="*60)
    print("PHYSICAL PLAN  (Exchange node = shuffle boundary / bottleneck)")
    print("="*60)
    df.explain(mode="formatted")
    try:
        plan = df._jdf.queryExecution().toString()
        with open(DAG_P, "w", encoding="utf-8") as f:
            f.write(plan)
        print(f"[dag] full plan saved → {DAG_P}")
    except Exception as e:
        print(f"[dag] plan export skipped ({e})")

# ─── 6. write then validate on written parquet ─────────────────────────────────
def write(df):
    df.repartition(4).write.mode("overwrite").parquet(OUT_P)
    print(f"[write] ✅ saved → {OUT_P}")

def validate_written(spark):
    df = spark.read.parquet(OUT_P)
    n  = df.count()
    nf = df.filter(F.col("features").isNull()).count()
    nl = df.filter(F.col("label").isNull()).count()
    dist = {int(r["label"]): r["count"]
            for r in df.groupBy("label").count().collect()}

    print(f"\n[validate] rows={n:,}  null_features={nf}  null_label={nl}")
    for lbl, cnt in sorted(dist.items()):
        print(f"  label={lbl}  count={cnt:,}  ({cnt/n*100:.1f}%)")
    assert nf == 0, "NULL features!"
    assert nl == 0, "NULL labels!"
    print("[validate] ✅ all checks passed")

# ─── main ──────────────────────────────────────────────────────────────────────
def main():
    t0    = time.time()
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    df        = load(spark)
    df        = sql_features(spark, df)
    df, ohe_c = encode(df)
    df        = assemble(df, ohe_c)   # slims to final cols inside

    export_dag(df)   # explain() only — no count, no cache
    write(df)
    validate_written(spark)

    print(f"\n✅ Week 2 complete in {time.time()-t0:.1f}s  |  output → {OUT_P}")
    spark.stop()

if __name__ == "__main__":
    main()