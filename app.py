# Import required libraries
import os                          # For operating system operations
from flask import Flask, request, render_template, jsonify  # Core Flask functionality
import numpy as np                 # For numerical operations
import librosa                     # For signal processing
import sklearn.preprocessing as preprocessing  # For data preprocessing
from sklearn.linear_model import LogisticRegression  # ML model
import joblib                      # For loading saved models
from datetime import datetime, timedelta  # For time operations
from whitenoise import WhiteNoise  # For serving static files
from pymongo import MongoClient    # MongoDB database connection
import gridfs                      # For storing large files in MongoDB
from dotenv import load_dotenv     # For loading environment variables
import tempfile                    # For temporary file operations
import csv                         # For CSV file operations
import obspy                       # For seismic data processing
from obspy.core import Trace, Stream  # For seismic data structures
from xml.etree.ElementTree import Element, SubElement, tostring  # For XML creation
from xml.dom.minidom import parseString  # For XML formatting
from flask import send_file        # For file downloads
import io                          # For in-memory file operations
import base64                      # For encoding/decoding base64 data

# Initialize Flask application
app = Flask(__name__)
application = app  # For WSGI compatibility

# Configure WhiteNoise for serving static files
app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/')

# Define constants for signal processing
SAMPLE_RATE = 100     # Sampling rate in Hz
N_MELS = 128         # Number of Mel frequency bands
FMIN = 0             # Minimum frequency for analysis
FMAX = 19            # Maximum frequency for analysis
FRAME_SIZE = 512     # Size of each frame for spectrogram
HOP_LENGTH = 256     # Number of samples between frames

# Load pre-trained model and scaler
clf = joblib.load('earthquake_model.joblib')
scaler = joblib.load('earthquake_scaler.joblib')

# Load environment variables and setup MongoDB connection
load_dotenv()
MONGO_URI = os.getenv("MONGO_URL")
print(f"MONGO_URI: {MONGO_URI}")

DB_NAME = 'seismic_quake'

# Initialize MongoDB client and GridFS
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
fs = gridfs.GridFS(db)

def extract_features(file_id):
    """Extract features from stored file for prediction"""
    # Retrieve file from GridFS and save to temporary file
    with fs.get(file_id) as f:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(f.read())
            temp_file_path = temp_file.name

    # Handle different file formats
    if temp_file_path.endswith('.mseed'):
        # Process miniseed format
        st = obspy.read(temp_file_path)
        tr = st[0]
        y = tr.data.astype(np.float32)
        sr = tr.stats.sampling_rate
    else:
        # Process other formats using librosa
        y, sr = librosa.load(temp_file_path, sr=SAMPLE_RATE)

    # Clean up temporary file
    os.remove(temp_file_path)

    # Calculate Mel spectrogram and MFCC features
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, 
                                     n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    return np.mean(mfcc, axis=1), y, sr

def predict(file_id):
    """Make prediction and identify earthquake events"""
    # Extract features and make prediction
    features, y, sr = extract_features(file_id)
    features = scaler.transform([features])
    prediction = clf.predict(features)
    print(prediction)

    # Identify potential earthquake events using threshold
    threshold = np.mean(y) + 3 * np.std(y)
    earthquake_indices = np.where(y > threshold)[0]

    return prediction[0], y, sr, earthquake_indices

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and analysis"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file:
            # Store file and process it
            file_id = fs.put(file, filename=file.filename)
            prediction, y, sr, earthquake_indices = predict(file_id)
            time_labels = [str(timedelta(seconds=i / sr)) for i in range(len(y))]
            
            if prediction == 1:
                # Return seismic activity data
                amplitudes = [float(y[idx]) for idx in earthquake_indices]
                return jsonify({
                    'prediction': 'Seismic Activity Detected',
                    'time_indices': earthquake_indices.tolist(),
                    'amplitudes': amplitudes,
                    'time_labels': time_labels,
                    'amplitude_data': y.tolist(),
                    'sampling_rate': sr
                })
            else:
                # Return no activity data
                return jsonify({
                    'prediction': 'No Seismic Activity Detected',
                    'time_labels': time_labels,
                    'amplitude_data': y.tolist()
                })
    return render_template('upload.html')

@app.route('/download_png', methods=['POST'])
def download_png():
    """Generate and send PNG visualization"""
    data = request.json
    image_base64 = data['image_base64']
    return send_file(
        io.BytesIO(base64.b64decode(image_base64.split(',')[1])),
        mimetype='image/png',
        as_attachment=True,
        download_name='waveform_chart.png'
    )

@app.route('/download_csv', methods=['POST'])
def download_csv():
    """Generate and send CSV data file"""
    data = request.json
    time_labels = data['time_labels']
    amplitude_data = data['amplitude_data']

    # Create CSV in memory
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Time', 'Amplitude'])
    cw.writerows(zip(time_labels, amplitude_data))

    # Prepare for download
    output = io.BytesIO()
    output.write(si.getvalue().encode())
    output.seek(0)
    si.close()

    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='waveform_data.csv'
    )

@app.route('/download_mseed', methods=['POST'])
def download_mseed():
    """Generate and send MSEED format file"""
    data = request.json
    time_labels = data['time_labels']
    amplitude_data = data['amplitude_data']
    sampling_rate = data['sampling_rate']

    # Create MSEED format data
    trace = Trace(data=np.array(amplitude_data, dtype=np.float32), 
                 header={'sampling_rate': sampling_rate})
    stream = Stream([trace])

    # Prepare for download
    output = io.BytesIO()
    stream.write(output, format='MSEED')
    output.seek(0)

    return send_file(
        output,
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name='waveform_data.mseed'
    )

@app.route('/download_xml', methods=['POST'])
def download_xml():
    """Generate and send XML format file"""
    data = request.json
    time_labels = data['time_labels']
    amplitude_data = data['amplitude_data']

    # Create XML structure
    root = Element('WaveformData')
    for time, amplitude in zip(time_labels, amplitude_data):
        entry = SubElement(root, 'Entry')
        time_elem = SubElement(entry, 'Time')
        time_elem.text = time
        amplitude_elem = SubElement(entry, 'Amplitude')
        amplitude_elem.text = str(amplitude)

    # Format XML with proper indentation
    xml_str = parseString(tostring(root)).toprettyxml(indent="  ")

    # Prepare for download
    output = io.BytesIO()
    output.write(xml_str.encode())
    output.seek(0)

    return send_file(
        output,
        mimetype='application/xml',
        as_attachment=True,
        download_name='waveform_data.xml'
    )

# Run the application
if __name__ == '__main__':
    app.run(debug=False)
