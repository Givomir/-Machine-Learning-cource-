import os
import jpype

# Get the path to the JVM (libjvm.so or jvm.dll)
java_jvm_path = jpype.getDefaultJVMPath()

# Set JAVA_HOME to the parent directory of libjvm.so or jvm.dll
java_home_path = os.path.dirname(os.path.dirname(java_jvm_path))
os.environ["JAVA_HOME"] = java_home_path
os.environ["PATH"] += os.pathsep + os.path.join(java_home_path, "bin")

# Start the JVM with jpype if not already started (only if needed for other dependencies)
if not jpype.isJVMStarted():
    jpype.startJVM()

# Now, initialize the Spark session
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("ECGPredictor").getOrCreate()

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

# Initialize Spark
spark = SparkSession.builder.appName("ECGClassification").getOrCreate()

# Load the dataset
# (Assume data is available in a CSV format)
data = spark.read.csv("path/to/ecg_data.csv", header=True, inferSchema=True)

# Assemble the feature columns (e.g., RR intervals)
assembler = VectorAssembler(inputCols=["RR_intervals"], outputCol="features")
data = assembler.transform(data)

# Apply Standard Scaling
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaled_data = scaler.fit(data).transform(data)

# Initialize RandomForestClassifier as an alternative to SVC
rf = RandomForestClassifier(labelCol="label", featuresCol="scaled_features", numTrees=100, maxDepth=5)

# Create a pipeline
pipeline = Pipeline(stages=[assembler, scaler, rf])

# Train the pipeline on the dataset
model = pipeline.fit(scaled_data)

# Save the model (replace with the correct path)
model.write().overwrite().save("path/to/saved_model")
