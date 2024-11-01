from pyspark.sql import SparkSession
import sys
import os
import sys
from pyspark import SparkContext
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# Initialize Spark session
spark = SparkSession.builder.appName("CheckPythonEnv").getOrCreate()

# Print Python version being used
print("Python interpreter being used by Spark:", sys.executable)

# Stop the Spark session
spark.stop()