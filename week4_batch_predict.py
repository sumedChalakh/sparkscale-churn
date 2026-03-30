import os
import sys
from pathlib import Path

from py4j.protocol import Py4JJavaError
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


def build_spark() -> SparkSession:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    return (
        SparkSession.builder
        .appName("SparkScale-W4-Batch-Predict")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.hadoop.io.native.lib.available", "false")
        .config("spark.driver.extraJavaOptions", "-Dhadoop.native.lib=false")
        .config("spark.executor.extraJavaOptions", "-Dhadoop.native.lib=false")
        .getOrCreate()
    )


def load_parquet_windows_safe(spark: SparkSession, parquet_path: str):
    try:
        return spark.read.parquet(parquet_path)
    except Py4JJavaError as err:
        if "NativeIO$Windows.access0" not in str(err):
            raise

        part_files = sorted(str(p) for p in Path(parquet_path).glob("part-*.parquet"))
        if not part_files:
            raise

        print("Detected Windows NativeIO issue. Falling back to explicit parquet part files...")
        return spark.read.parquet(*part_files)


def main():
    input_path = os.getenv("BATCH_INPUT", "data/features/churn_features.parquet")
    model_root = os.getenv("BATCH_MODEL_ROOT", "models")
    model_name = os.getenv("BATCH_MODEL_NAME", "GBT")
    output_path = os.getenv("BATCH_OUTPUT", "batch_output/churn_predictions.parquet")
    output_format = os.getenv("BATCH_OUTPUT_FORMAT", "parquet").lower()

    if output_format not in {"parquet", "csv"}:
        raise ValueError("BATCH_OUTPUT_FORMAT must be either 'parquet' or 'csv'.")

    model_path = Path(model_root) / f"{model_name}_best"
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input monthly data not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Saved pipeline model not found: {model_path}")

    spark = build_spark()
    spark.sparkContext.setLogLevel("ERROR")

    print(f"Loading monthly data: {input_path}")
    df = load_parquet_windows_safe(spark, input_path)
    print(f"Rows loaded: {df.count():,}")

    print(f"Loading saved pipeline model: {model_path}")
    try:
        model = PipelineModel.load(str(model_path))
    except Py4JJavaError as err:
        if "NativeIO$Windows.access0" in str(err):
            raise RuntimeError(
                "Model loading failed due to Windows Hadoop NativeIO limitation. "
                "Set HADOOP_HOME/winutils or run in Docker Spark cluster for strict production artifact loading."
            ) from err
        raise

    pred = model.transform(df)
    pred = pred.withColumn("churn_probability", vector_to_array(col("probability"))[1].cast("double"))

    out_cols = []
    if "customerID" in pred.columns:
        out_cols.append(col("customerID"))
    out_cols.extend([
        col("prediction").cast("double").alias("prediction"),
        col("churn_probability"),
    ])
    if "label" in pred.columns:
        out_cols.append(col("label").cast("double").alias("label"))

    out_df = pred.select(*out_cols)

    output_parent = Path(output_path).parent
    output_parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing batch predictions to: {output_path}")
    writer = out_df.write.mode("overwrite")
    if output_format == "csv":
        writer.option("header", True).csv(output_path)
    else:
        writer.parquet(output_path)

    print("Batch prediction job completed.")
    spark.stop()


if __name__ == "__main__":
    main()
