# Import required libraries
import numpy as np                  # For numerical operations and array handling
import librosa                      # For audio/signal processing features
import obspy                        # For reading and processing seismic data
from sklearn import metrics         # For model evaluation metrics
import sklearn.preprocessing as preprocessing  # For data normalization
from sklearn.linear_model import LogisticRegression  # The ML model we'll use
from sklearn.model_selection import train_test_split  # For splitting training/test data
import joblib                       # For saving trained models
import os                          # For file operations
import pandas as pd                # For handling CSV data
import xml.etree.ElementTree as ET # For parsing XML files

# Define constants for signal processing
SAMPLE_RATE = 100  # Sampling rate in Hz, focused on frequencies below 20Hz
FRAME_SIZE = 512   # Size of each frame for spectrogram calculation
HOP_LENGTH = 256   # Number of samples between successive frames
N_MELS = 128      # Number of Mel frequency bands
FMIN = 0          # Minimum frequency for analysis
FMAX = 19         # Maximum frequency for analysis

def extract_features(file_path):
    """Extract MFCC features from different types of seismic data files"""
    # Handle different file formats
    if file_path.endswith('.mseed'):
        # Read miniseed format seismic data
        st = obspy.read(file_path)
        tr = st[0]  # Get first trace
        y = tr.data.astype(np.float32)  # Convert to float32 for processing
        sr = tr.stats.sampling_rate
    elif file_path.endswith('.csv'):
        # Read CSV format data
        df = pd.read_csv(file_path)
        # Convert first column to numeric, handling any non-numeric values
        y = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values.astype(np.float32)
        sr = 100  # Assume 100Hz sampling rate for CSV files
    elif file_path.endswith('.xml'):
        # Read XML format data
        tree = ET.parse(file_path)
        root = tree.getroot()
        # Extract numeric values from XML elements
        y = np.array([float(child.text) for child in root if child.text.replace('.', '', 1).isdigit()]).astype(np.float32)
        sr = 100  # Assume 100Hz sampling rate for XML files
    else:
        # For other formats, use librosa's default loader
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Handle empty data case
    if len(y) == 0:
        print(f"Warning: Empty data for file {file_path}")
        return np.zeros(13)  # Return zero vector for empty data

    # Calculate Mel spectrogram
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX,
        n_fft=FRAME_SIZE, hop_length=HOP_LENGTH
    )
    
    # Convert to log scale
    log_S = librosa.power_to_db(S, ref=np.max)
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    
    # Return mean of MFCCs across time
    return np.mean(mfcc, axis=1)

def load_data(quake_files, no_quake_files):
    """Load and process both earthquake and non-earthquake data files"""
    X, y = [], []  # Initialize feature and label lists
    
    # Process earthquake files
    for file in quake_files:
        if os.path.exists(file):
            features = extract_features(file)
            print(f"Extracted features for quake file {file}: {features}")
            X.append(features)
            y.append(1)  # Label 1 for seismic quakes
        else:
            print(f"File not found: {file}")
    
    # Process non-earthquake files
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
    """Train and evaluate the earthquake detection model"""
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize features using StandardScaler
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Initialize and train logistic regression model
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions and evaluate model performance
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("F1 Score:", metrics.f1_score(y_test, y_pred))
    
    return clf, scaler

def main():
    """Main function to orchestrate the training process"""
    # Initialize empty lists for file paths
    quake_files = []    # Add paths to earthquake data files here
    no_quake_files = [] # Add paths to non-earthquake data files here
    
    # Load and process the data
    X, y = load_data(quake_files, no_quake_files)
    
    # Check if we have data to process
    if len(X) == 0 or len(y) == 0:
        print("No data loaded. Please check the paths and ensure the files are present.")
        return
    
    # Train the model and get the scaler
    clf, scaler = train_model(X, y)
    
    # Save the trained model and scaler for later use
    joblib.dump(clf, 'earthquake_model.joblib')
    joblib.dump(scaler, 'earthquake_scaler.joblib')

# Standard Python idiom to ensure main() only runs if script is executed directly
if __name__ == '__main__':
    main()
