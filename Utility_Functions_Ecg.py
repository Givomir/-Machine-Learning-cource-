import numpy as np
from scipy.signal import resample, find_peaks
from sklearn.preprocessing import StandardScaler

def preprocess(data, config):
    # Sample rate adjustment
    sr = config.sample_rate or 300
    data = np.nan_to_num(data)  # Remove NaNs and Infs

    # Resampling to match the expected sampling rate
    data = resample(data, int(len(data) * 360 / sr))

    # Scaling data
    scaler = StandardScaler()
    data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    # Reshaping the data to the format needed for the SVM model
    return data, find_peaks(data, distance=150)

# Feature extraction for predicting with the same method used during training
def extract_rr_intervals(annotation):
    rr_intervals = np.diff(annotation.sample) / annotation.fs  # RR intervals in seconds
    return rr_intervals