from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import joblib
import os
import wfdb

# Path to the MIT-BIH database
DATA_DIR = r'C:\\stasi\\SoftUni_Machine_learning\\mit-bih-arrhythmia-database-1.0.0\\mit-bih-arrhythmia-database-1.0.0'

# List of record numbers to process
RECORD_NUMBERS = ['100', '101', '102', '103', '104', '105']  # Use a subset for demonstration

# Extract RR intervals and labels
train_rr_intervals = []
train_labels = []

def extract_rr_intervals(annotation):
    return np.diff(annotation.sample) / annotation.fs  # RR intervals in seconds

def process_records(record_list, rr_intervals_list, labels_list):
    for record_number in record_list:
        record_path = os.path.join(DATA_DIR, record_number)
        try:
            # Load the record and annotations
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')

            # Extract RR intervals and labels for each beat
            rr_intervals = extract_rr_intervals(annotation)
            labels = [1 if sym == 'N' else 0 for sym in annotation.symbol[:-1]]  # 1 for normal, 0 for others

            # Append to the corresponding dataset
            rr_intervals_list.extend(rr_intervals)
            labels_list.extend(labels)
        except Exception as e:
            print(f"Could not process record {record_number}: {e}")

# Process training records
process_records(RECORD_NUMBERS, train_rr_intervals, train_labels)

# Convert lists to numpy arrays
X = np.array(train_rr_intervals).reshape(-1, 1)  # RR intervals as features
y = np.array(train_labels)  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to handle class imbalance
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# Create a pipeline with StandardScaler and SVM classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Standardize the features
    ('svm', SVC(C=100, gamma=1, kernel='rbf', class_weight={0: 3, 1: 1}))  # Step 2: Train SVM
])

# Train the pipeline on the balanced dataset
pipeline.fit(X_res, y_res)

# Evaluate the model on the test data
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the entire pipeline
MODEL_PATH = 'optimized_svm_mode_1.pkl'
joblib.dump(pipeline, MODEL_PATH)
print("Model saved successfully!")