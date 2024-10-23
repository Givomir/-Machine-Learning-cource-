# Corrected config.py - Configuration for FastAPI Application
class Config:
    def __init__(self):
        self.sample_rate = 360  # Sample rate of the ECG recordings
        self.model_path = "optimized_svm_model.pkl"  # Path to the trained model
        self.upload_folder = "C:\\stasi\\SoftUni_Machine_learning\\mit-bih-arrhythmia-database-1.0.0\\mit-bih-arrhythmia-database-1.0.0"

# Function to get configuration

def get_config():
    return Config()