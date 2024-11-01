import pandas as pd

# Define some dummy data for testing (e.g., 5 features, 10 rows)
dummy_data = {
    "feature_1": [0.1, 0.5, 0.3, 0.6, 0.9, 1.1, 0.2, 0.4, 0.7, 0.8],
    "feature_2": [1.0, 0.9, 0.7, 0.4, 0.3, 0.8, 0.2, 1.1, 0.5, 0.6],
    "feature_3": [0.5, 0.6, 0.3, 0.7, 0.2, 0.1, 0.9, 0.8, 1.0, 0.4],
    "feature_4": [0.8, 1.0, 0.2, 0.3, 0.6, 0.7, 0.4, 0.9, 0.5, 1.1],
    "feature_5": [0.3, 0.2, 1.1, 0.8, 0.9, 0.7, 0.5, 1.0, 0.4, 0.6],
}

# Create a DataFrame from the dummy data
dummy_df = pd.DataFrame(dummy_data)

# Save the dummy DataFrame to a CSV file
csv_path = r"C:\stasi\SoftUni_Machine_learning\Pyspark_FastAPI\dummy_data.csv"
dummy_df.to_csv(csv_path, index=False)

print(f"Dummy CSV file created at: {csv_path}")
