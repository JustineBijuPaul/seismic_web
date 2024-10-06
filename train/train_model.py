import numpy as np
import librosa
import obspy
from sklearn import metrics
import sklearn.preprocessing as preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os
import pandas as pd
import xml.etree.ElementTree as ET

# Constants
SAMPLE_RATE = 100  # Since we're interested in frequencies less than 20Hz
FRAME_SIZE = 512
HOP_LENGTH = 256
N_MELS = 128
FMIN = 0
FMAX = 19

def extract_features(file_path):
    if file_path.endswith('.mseed'):
        st = obspy.read(file_path)
        tr = st[0]
        y = tr.data.astype(np.float32)  # Ensure the data is in floating-point format
        sr = tr.stats.sampling_rate
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        # Ensure the data is numeric
        y = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values.astype(np.float32)
        sr = 100  # Assuming the sample rate is 100Hz for CSV files
    elif file_path.endswith('.xml'):
        tree = ET.parse(file_path)
        root = tree.getroot()
        y = np.array([float(child.text) for child in root if child.text.replace('.', '', 1).isdigit()]).astype(np.float32)
        sr = 100  # Assuming the sample rate is 100Hz for XML files
    else:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    if len(y) == 0:
        print(f"Warning: Empty data for file {file_path}")
        return np.zeros(13)  # Return a zero vector if the data is empty

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def load_data(quake_files, no_quake_files):
    X, y = [], []

    for file in quake_files:
        if os.path.exists(file):
            features = extract_features(file)
            print(f"Extracted features for quake file {file}: {features}")
            X.append(features)
            y.append(1)  # Label 1 for seismic quakes
        else:
            print(f"File not found: {file}")

    for file in no_quake_files:
        if os.path.exists(file):
            features = extract_features(file)
            print(f"Extracted features for no quake file {file}: {features}")
            X.append(features)
            y.append(0)  # Label 0 for no seismic quakes
        else:
            print(f"File not found: {file}")

    return np.array(X), np.array(y)

def train_model(X, y):
    # Shuffle and split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a Logistic Regression Classifier
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("F1 Score:", metrics.f1_score(y_test, y_pred))

    return clf, scaler

def main():
    quake_files =   []

    no_quake_files = []
    X, y = load_data(quake_files, no_quake_files)

    if len(X) == 0 or len(y) == 0:
        print("No data loaded. Please check the paths and ensure the files are present.")
        return

    clf, scaler = train_model(X, y)
        # Save the model and scaler
    joblib.dump(clf, 'earthquake_model.joblib')
    joblib.dump(scaler, 'earthquake_scaler.joblib')

if __name__ == '__main__':
    main()
