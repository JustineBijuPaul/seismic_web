let monitoringInterval = null;
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let recordingInterval;
let audioContext;
let analyser;
let dataArray;
let animationId;

function startMonitoring(fileId) {
    // Tell server to start monitoring
    fetch('/start_monitoring', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_id: fileId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'Monitoring started') {
            // Check for new data every 5 seconds
            monitoringInterval = setInterval(updateResults, 5000);
            document.getElementById('monitorStatus').textContent = 'Monitoring Active';
        }
    });
}

function stopMonitoring() {
    // Tell server to stop monitoring
    fetch('/stop_monitoring', { method: 'POST' })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'Monitoring stopped') {
            clearInterval(monitoringInterval);
            document.getElementById('monitorStatus').textContent = 'Monitoring Stopped';
        }
    });
}

function updateResults() {
    // Get latest analysis results
    fetch('/get_latest_results')
    .then(response => response.json())
    .then(data => {
        if (!data.error) {
            // Update chart with new data
            updateChart(data.data.time_labels, data.data.amplitude_data);
            // Show alert if seismic activity detected
            if (data.prediction === 1) {
                showAlert('Seismic Activity Detected!');
            }
        }
    });
}

async function startMicrophoneMonitoring() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);

        analyser.fftSize = 2048;
        const bufferLength = analyser.frequencyBinCount;
        dataArray = new Uint8Array(bufferLength);

        mediaRecorder = new MediaRecorder(stream);
        isRecording = true;
        document.getElementById('monitorStatus').textContent = 'Monitoring Active (Microphone)';

        // Start real-time visualization
        const canvas = document.getElementById('waveformChart');
        const canvasCtx = canvas.getContext('2d');
        
        function drawWaveform() {
            animationId = requestAnimationFrame(drawWaveform);
            analyser.getByteTimeDomainData(dataArray);

            canvasCtx.fillStyle = 'rgba(28, 28, 31, 0.3)';
            canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

            canvasCtx.lineWidth = 2;
            canvasCtx.strokeStyle = 'hsla(36, 72%, 70%, 1)';
            canvasCtx.beginPath();

            const sliceWidth = canvas.width * 1.0 / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = v * canvas.height / 2;

                if (i === 0) {
                    canvasCtx.moveTo(x, y);
                } else {
                    canvasCtx.lineTo(x, y);
                }

                x += sliceWidth;
            }

            canvasCtx.lineTo(canvas.width, canvas.height / 2);
            canvasCtx.stroke();
        }

        // Show results section and start visualization
        document.querySelector('.results-section').style.display = 'block';
        drawWaveform();

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const formData = new FormData();
            formData.append('file', audioBlob, 'recording.wav');
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if (data.prediction === 'Seismic Activity Detected') {
                    showAlert('⚠️ Seismic Activity Detected!');
                    updateChart(data.time_labels, data.amplitude_data);
                }
            } catch (error) {
                console.error('Error analyzing audio:', error);
            }
            
            audioChunks = [];
            if (isRecording) {
                mediaRecorder.start();
            }
        };

        // Record in 5-second chunks
        mediaRecorder.start();
        recordingInterval = setInterval(() => {
            if (isRecording) {
                mediaRecorder.stop();
            }
        }, 5000);

    } catch (error) {
        console.error('Error accessing microphone:', error);
        showAlert('Error: Could not access microphone');
    }
}

function stopMicrophoneMonitoring() {
    if (mediaRecorder && isRecording) {
        isRecording = false;
        clearInterval(recordingInterval);
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        document.getElementById('monitorStatus').textContent = 'Monitoring Stopped';
        
        // Stop visualization
        if (animationId) {
            cancelAnimationFrame(animationId);
        }
        if (audioContext) {
            audioContext.close();
        }
    }
}

function showAlert(message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-warning';
    alertDiv.textContent = message;
    document.getElementById('alertContainer').appendChild(alertDiv);
    setTimeout(() => alertDiv.remove(), 5000);
}
