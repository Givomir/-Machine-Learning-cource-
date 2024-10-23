import numpy as np
from scipy.signal import resample, find_peaks
from sklearn.preprocessing import StandardScaler
import joblib

# Define a path to save the scaler
SCALER_PATH = "scaler.pkl"

def preprocess(data, config, training=False):
    """
    Preprocess the input data: Resample, handle NaNs, and scale the data.
    
    Parameters:
    - data: The raw input data to preprocess.
    - config: Configuration with necessary settings like sample_rate.
    - training: Boolean flag indicating whether we're in training mode or inference mode.
    
    Returns:
    - Processed data ready for model input.
    """
    # Sample rate adjustment
    sr = config.sample_rate or 300
    data = np.nan_to_num(data)  # Remove NaNs and Infs

    # Resampling to match the expected sampling rate
    data = resample(data, int(len(data) * 360 / sr))

    # Load or fit the scaler
    if training:
        # During training, fit a new scaler
        scaler = StandardScaler()
        data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

        # Save the fitted scaler for later use
        joblib.dump(scaler, SCALER_PATH)
    else:
        # During inference, load the existing scaler
        scaler = joblib.load(SCALER_PATH)
        data = scaler.transform(data.reshape(-1, 1)).flatten()

    # Reshaping the data to the format needed for the SVM model
    return data, find_peaks(data, distance=150)

# Feature extraction for predicting with the same method used during training
def extract_rr_intervals(annotation):
    rr_intervals = np.diff(annotation.sample) / annotation.fs  # RR intervals in seconds
    return rr_intervals
