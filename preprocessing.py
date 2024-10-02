import numpy as np
import pandas as pd

from scipy.signal import filtfilt, butter, find_peaks
from scipy.stats import skew, kurtosis
from scipy.integrate import cumulative_trapezoid
from sklearn.preprocessing import LabelEncoder

DATA_COLUMNS = ['AccX', 'AccY', 'AccZ', 'VelX', 'VelY', 'VelZ', 'PosX', 'PosY', 'PosZ', 'Pitch', 'Roll', 'Yaw']
EXERCISE_TYPES = ['Ab Rollout', 'Bench', 'DB Bench', 'DB Bicep Curl', 'DB Deadlift', 'DB Goblet Squat', 
                  'DB Side Raise', 'Deadlift', 'Front Squat', 'Horizontal Chest Press', 'Lat Pulldown',
                  'OHP', 'Row', 'Seated Cable Rows', 'Squat', 'Tricep Cable Pushdown']
FEATURE_MAP = {
    'Mean': np.mean,
    'Std': np.std,
    'AbsDev': lambda x: np.mean(np.abs(x - np.mean(x))),
    'Median': np.median,
    'MedianAbsDev': lambda x: np.median(np.abs(x - np.median(x))),
    'IQR': lambda x: np.percentile(x, 75) - np.percentile(x, 25),
    'NegCount': lambda x: np.sum(x < 0),
    'PosCount': lambda x: np.sum(x > 0),
    'AboveMeanCount': lambda x: np.sum(x > np.mean(x)),
    'NumPeaks': lambda x: len(find_peaks(x)[0]),
    'Energy': lambda x: np.sum(x ** 2),
    'Skewness': skew,
    'Kurtosis': kurtosis,
    'AvgResultantAcc': lambda x: np.mean(np.sqrt(x ** 2)),
    'SMA': lambda x: np.sum(np.abs(x))
}


def preprocess_data(df, cutoff_freq, sample_rate):
    
    def filter(data):
        b, a = butter(1, 2 * cutoff_freq / (sample_rate), btype="highpass")
        return filtfilt(b, a, data)
    
    dx = 1/sample_rate

    df['AccX'] = df['AccX'].apply(lambda x: filter(x))
    df['AccY'] = df['AccY'].apply(lambda x: filter(x))
    df['AccZ'] = df['AccZ'].apply(lambda x: filter(x))

    df['VelX'] = df['AccX'].apply(lambda x: cumulative_trapezoid(x, dx=dx))
    df['VelY'] = df['AccY'].apply(lambda x: cumulative_trapezoid(x, dx=dx))
    df['VelZ'] = df['AccZ'].apply(lambda x: cumulative_trapezoid(x, dx=dx))

    df['VelX'] = df['VelX'].apply(lambda x: filter(x))
    df['VelY'] = df['VelY'].apply(lambda x: filter(x))
    df['VelZ'] = df['VelZ'].apply(lambda x: filter(x))

    df['PosX'] = df['VelX'].apply(lambda x: cumulative_trapezoid(x, dx=dx))
    df['PosY'] = df['VelY'].apply(lambda x: cumulative_trapezoid(x, dx=dx))
    df['PosZ'] = df['VelZ'].apply(lambda x: cumulative_trapezoid(x, dx=dx))
    
    df['PosX'] = df['PosX'].apply(lambda x: filter(x))
    df['PosY'] = df['PosY'].apply(lambda x: filter(x))
    df['PosZ'] = df['PosZ'].apply(lambda x: filter(x))

    return df




def extract_labels(df, label_name):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df[label_name])
    return pd.DataFrame(labels, columns=['Label'])


def extract_features(df):
    feature_dict = {}

    for data_column in DATA_COLUMNS:
        for feature_name, feature_func in FEATURE_MAP.items():
            feature_dict[f'{data_column}{feature_name}'] = df[data_column].apply(feature_func)

    return pd.concat(feature_dict, axis=1)


def extract_window_features(df, window_size, step_size):
    windowed_data = []

    for _, row in df.iterrows():
        for i in range(0, len(row['Counter']) - window_size, step_size):
            windowed_data.append({
                'Exercise': row['Exercise'],
                'Counter': row['Counter'][i:i+ window_size],
                'TimeBetweenSamples': row['TimeBetweenSamples'][i:i+window_size],
                'AccX': row['AccX'][i:i+window_size],
                'AccY': row['AccY'][i:i+window_size],
                'AccZ': row['AccZ'][i:i+window_size],
                'VelX': row['VelX'][i:i+window_size],
                'VelY': row['VelY'][i:i+window_size],
                'VelZ': row['VelZ'][i:i+window_size],
                'PosX': row['PosX'][i:i+window_size],
                'PosY': row['PosY'][i:i+window_size],
                'PosZ': row['PosZ'][i:i+window_size],
                'Pitch': row['Pitch'][i:i+window_size],
                'Roll': row['Roll'][i:i+window_size],
                'Yaw': row['Yaw'][i:i+window_size],
                'HeartRate': row['HeartRate'][i:i+window_size],
            })

    window_data = pd.DataFrame(windowed_data)
    return extract_features(window_data)


def extract_window_labels(df, window_size, step_size, sample_rate):
    windowed_labels = []

    def calculate_sample_index(time, sample_rate):
        return int(float(time)* float(sample_rate))

    for _, row in df.iterrows():
        for i in range(0, len(row['Counter']) - window_size, step_size):
            start_index = calculate_sample_index(row['StartRepTime'], sample_rate)
            end_index = calculate_sample_index(row['EndRepTime'], sample_rate)

            windowed_labels.append(1 if i >= start_index and i <= end_index else 0)

    return pd.DataFrame(windowed_labels, columns=['Label'])
