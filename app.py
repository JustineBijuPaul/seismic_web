import os
from flask import Flask, request, render_template, jsonify, send_file
import numpy as np
import librosa
import obspy
import sklearn.preprocessing as preprocessing
from sklearn.linear_model import LogisticRegression
import joblib
from datetime import datetime, timedelta
from whitenoise import WhiteNoise

app = Flask(__name__)
application = app

app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/')

# Constants
SAMPLE_RATE = 100
N_MELS = 128
FMIN = 0
FMAX = 19
FRAME_SIZE = 512
HOP_LENGTH = 256

# Load the trained model and scaler
clf = joblib.load('earthquake_model.joblib')
scaler = joblib.load('earthquake_scaler.joblib')

def extract_features(file_path):
    if file_path.endswith('.mseed'):
        st = obspy.read(file_path)
        tr = st[0]
        y = tr.data.astype(np.float32)
        sr = tr.stats.sampling_rate
    else:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    return np.mean(mfcc, axis=1), y, sr

def predict(file_path):
    features, y, sr = extract_features(file_path)
    features = scaler.transform([features])
    prediction = clf.predict(features)
    print(prediction)

    # Identify all points where the amplitude exceeds a certain threshold
    threshold = np.mean(y) + 3 * np.std(y)  # Example threshold
    earthquake_indices = np.where(y > threshold)[0]

    return prediction[0], y, sr, earthquake_indices

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            filename = file.filename
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            prediction, y, sr, earthquake_indices = predict(file_path)
            os.remove(file_path)  # Remove the file after prediction
            time_labels = [str(timedelta(seconds=i / sr)) for i in range(len(y))]
            if prediction == 1:
                amplitudes = [float(y[idx]) for idx in earthquake_indices]  # Convert to float
                return jsonify({
                    'prediction': 'Seismic Activity Detected',
                    'time_indices': earthquake_indices.tolist(),
                    'amplitudes': amplitudes,
                    'time_labels': time_labels,
                    'amplitude_data': y.tolist(),
                    'sampling_rate': sr  # Include the sampling rate in the response
                })
            else:
                return jsonify({
                    'prediction': 'No Seismic Activity Detected',
                    'time_labels': time_labels,
                    'amplitude_data': y.tolist()
                })
    return render_template('upload.html')

@app.route('/download_mseed', methods=['POST'])
def download_mseed():
    data = request.json
    earthquake_indices = data['time_indices']
    amplitudes = data['amplitudes']
    sr = data['sampling_rate']

    # Create a new trace with the earthquake hit points
    tr = obspy.Trace(data=np.array(amplitudes), header={'sampling_rate': sr})
    st = obspy.Stream(tr)

    # Save the trace to an .mseed file
    mseed_file_path = os.path.join('uploads', 'earthquake_hit_points.mseed')
    st.write(mseed_file_path, format='MSEED')

    return send_file(mseed_file_path, as_attachment=True, download_name='earthquake_hit_points.mseed')

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
