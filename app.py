import os
from flask import Flask, request, render_template, jsonify
import numpy as np
import librosa
import sklearn.preprocessing as preprocessing
from sklearn.linear_model import LogisticRegression
import joblib
from datetime import datetime, timedelta
from whitenoise import WhiteNoise
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv
import tempfile
import csv
import obspy
from obspy.core import Trace, Stream
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

app = Flask(__name__)
application = app

app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/')

SAMPLE_RATE = 100
N_MELS = 128
FMIN = 0
FMAX = 19
FRAME_SIZE = 512
HOP_LENGTH = 256

clf = joblib.load('earthquake_model.joblib')
scaler = joblib.load('earthquake_scaler.joblib')

load_dotenv()
MONGO_URI = os.getenv("MONGO_URL")
print(f"MONGO_URI: {MONGO_URI}")

DB_NAME = 'seismic_quake'

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
fs = gridfs.GridFS(db)

def extract_features(file_id):
    with fs.get(file_id) as f:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(f.read())
            temp_file_path = temp_file.name

    y, sr = librosa.load(temp_file_path, sr=SAMPLE_RATE)
    if temp_file_path.endswith('.mseed'):
        st = obspy.read(temp_file_path)
        tr = st[0]
        y = tr.data.astype(np.float32)
        sr = tr.stats.sampling_rate
    else:
        y, sr = librosa.load(temp_file_path, sr=SAMPLE_RATE)

    os.remove(temp_file_path)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    return np.mean(mfcc, axis=1), y, sr

def predict(file_id):
    features, y, sr = extract_features(file_id)
    features = scaler.transform([features])
    prediction = clf.predict(features)
    print(prediction)

    threshold = np.mean(y) + 3 * np.std(y)
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
            file_id = fs.put(file, filename=file.filename)
            prediction, y, sr, earthquake_indices = predict(file_id)
            time_labels = [str(timedelta(seconds=i / sr)) for i in range(len(y))]
            if prediction == 1:
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
                return jsonify({
                    'prediction': 'No Seismic Activity Detected',
                    'time_labels': time_labels,
                    'amplitude_data': y.tolist()
                })
    return render_template('upload.html')

@app.route('/download_png', methods=['POST'])
def download_png():
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
    data = request.json
    time_labels = data['time_labels']
    amplitude_data = data['amplitude_data']

    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Time', 'Amplitude'])
    cw.writerows(zip(time_labels, amplitude_data))

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
    data = request.json
    time_labels = data['time_labels']
    amplitude_data = data['amplitude_data']
    sampling_rate = data['sampling_rate']

    trace = Trace(data=np.array(amplitude_data, dtype=np.float32), header={'sampling_rate': sampling_rate})
    stream = Stream([trace])

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
    data = request.json
    time_labels = data['time_labels']
    amplitude_data = data['amplitude_data']

    root = Element('WaveformData')
    for time, amplitude in zip(time_labels, amplitude_data):
        entry = SubElement(root, 'Entry')
        time_elem = SubElement(entry, 'Time')
        time_elem.text = time
        amplitude_elem = SubElement(entry, 'Amplitude')
        amplitude_elem.text = str(amplitude)

    xml_str = parseString(tostring(root)).toprettyxml(indent="  ")

    output = io.BytesIO()
    output.write(xml_str.encode())
    output.seek(0)

    return send_file(
        output,
        mimetype='application/xml',
        as_attachment=True,
        download_name='waveform_data.xml'
    )

if __name__ == '__main__':
    app.run(debug=True)