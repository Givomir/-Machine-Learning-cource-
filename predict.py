import joblib
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.functions import pandas_udf, PandasUDFType

# Step 1: Initialize Spark Session
spark = SparkSession.builder.appName("ECGPredictor").getOrCreate()

# Step 2: Load and Broadcast the Pre-trained Model
model_path = r"C:\stasi\SoftUni_Machine_learning\Pyspark_FastAPI\optimized_svm_model.pkl"
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    print(f"Model file not found: {model_path}. Please make sure the file exists.")
    raise
except Exception as e:
    print(f"An unexpected error occurred while loading the model: {e}")
    raise

# Broadcast the model
broadcast_model = spark.sparkContext.broadcast(model)

# Define the prediction function
def make_prediction(features):
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    label_map = {0: "Normal", 1: "Anomaly"}
    return label_map.get(prediction[0], "Unknown")

# Step 3: Load Dummy Data from CSV
csv_path = r"C:\stasi\SoftUni_Machine_learning\Pyspark_FastAPI\dummy_data.csv"
try:
    # Read the CSV file as a Spark DataFrame
    spark_df = spark.read.csv(csv_path, header=True, inferSchema=True)
    print("Dummy CSV data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure the CSV file is in the correct location.")
    raise

# Step 4: Define the Pandas UDF for Making Predictions on a Single Feature
@pandas_udf("struct<prediction:int, diagnosis:string>", PandasUDFType.SCALAR)
def predict_pandas_udf(feature):
    # Make predictions using the model with a single feature
    predictions = broadcast_model.value.predict(feature.to_numpy().reshape(-1, 1))
    
    # Map predictions to labels
    label_map = {0: "Normal", 1: "Anomaly"}
    labels = [label_map.get(int(pred), "Unknown") for pred in predictions]
    
    # Create the result as a DataFrame with two columns
    return pd.DataFrame({"prediction": predictions, "diagnosis": labels})

# Step 5: Apply the UDF to Make Predictions on `feature_1`
predictions = spark_df.withColumn("prediction", predict_pandas_udf("feature_1"))

# Step 6: Show Predictions
predictions.select("feature_1", "prediction.prediction", "prediction.diagnosis").show()
