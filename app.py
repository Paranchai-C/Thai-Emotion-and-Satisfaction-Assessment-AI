from flask import Flask, jsonify, render_template, request ,send_file,url_for
from io import BytesIO
import os
import numpy as np
import soundfile as sf
import pyaudio
import wave
import threading
from pydub import AudioSegment
from pydub.silence import detect_silence
from transformers import pipeline


app = Flask(__name__)
# Global variables for recording
recording = False
audio_filename = "recorded_audio.wav"

# Define constants

class_names = ['Angry', 'Happy', 'Neutral']

 # Load the saved model
device = -1  # Use GPU if available; set to -1 for CPU
pipe = pipeline("audio-classification", model="Paranchai/wav2vec2-large-xlsr-53-th-speech-emotion-recognition-3c", device=device)

 #show
score=0
value=""

score_mapping = {
    'Anger': -2,
    'Happiness': 2,
    'Neutral': 0
}

def split_audio_at_silence(audio_path, silence_thresh=-50, min_silence_len=1000):
    try:
        # Load audio file
        audio = AudioSegment.from_file(audio_path)

        # Detect silent segments
        silent_ranges = detect_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )

        # If no silence detected, return the original audio
        if not silent_ranges:
            print("No silent parts detected.")
            return [audio]

        # Split audio based on detected silences
        segments = []
        start_time = 0

        for start, end in silent_ranges:
            if start_time < start:  # Ensure we don't add empty segments
                segments.append(audio[start_time:start])
            start_time = end

        # Add the last segment if there's any remaining audio
        if start_time < len(audio):
            segments.append(audio[start_time:])

        print(f"Segments created: {len(segments)}")
        return segments

    except Exception as e:
        print(f"Error processing audio file: {e}")
        return []

def save_segments_and_predict(segments, pipe, output_prefix):
    for i, segment in enumerate(segments):
        # Save the segment to a file
        segment_file_name = f"{output_prefix}_part_{i + 1}.wav"
        segment.export(segment_file_name, format="wav")
        print(f"Saved segment: {segment_file_name}")
    print("save_segments_and_predict")
    

def load_and_predict_audio_segment(segment_path, pipe, label_counts):
    print("load_and_predict_audio_segment")
    # Load the saved audio file
    
    results = pipe(segment_path)
    top_class = max(results, key=lambda x: x['score'])
    top_label = top_class['label']

    print(f"{segment_path}: {top_label}")

    # Update label counts
    if top_label in label_counts:
        label_counts[top_label] += 1
    else:
        label_counts[top_label] = 1

    # Delete the segment file after processing
    print("load_and_predict_audio_segment")
    os.remove(segment_path)

def record_audio():
    global recording
    chunk = 1024  # Number of frames per buffer
    format = pyaudio.paInt16  # Sample format
    channels = 1  # Number of channels
    rate = 44100  # Sampling rate
    p = pyaudio.PyAudio()

    # เปิดไฟล์ในโหมดเขียน
    with wave.open(audio_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)

        stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

        while recording:
            data = stream.read(chunk)
            wf.writeframes(data)  # เขียนข้อมูลเสียงลงไฟล์ทันที

        stream.stop_stream()
        stream.close()
        p.terminate()
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getData')
def getData():
    return jsonify({
        'score': score,
        "value":value
        }), 200

@app.route('/show')
def show():
    return render_template('show.html')

@app.route('/loading')
def loading():
    return render_template('loading.html')

@app.route('/getShow')
def getShow():
    return jsonify({'redirect_url': url_for('show')})

@app.route('/audio', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No audio part'}), 400
    
    audio_file = request.files['file']
    audio_data = BytesIO(audio_file.read())

    output_file_prefix = "test2"     
    
    segments = split_audio_at_silence(audio_data, silence_thresh=-50, min_silence_len=1000)
    
    save_segments_and_predict(segments, pipe, output_file_prefix)

    label_counts = {}
    for i in range(len(segments)):
        segment_file_name = f"{output_file_prefix}_part_{i + 1}.wav"
        load_and_predict_audio_segment(segment_file_name, pipe, label_counts)

    # Calculate satisfaction score
    satisfaction_score = 0
    total_count = sum(label_counts.values())  # Total count of emotions

    for label, count in label_counts.items():
        if label in score_mapping:
            satisfaction_score += score_mapping[label] * count

    # Normalize the satisfaction score
    if total_count > 0:  # Check to avoid division by zero
        normalized_score = satisfaction_score / total_count
    else:
        normalized_score = 0

    global score, value
    end_result = ((normalized_score + 2)/4)*100
    score = end_result
    if 0 <= end_result <= 20:
        value = "very poor"
    elif 21 <= end_result <= 40:
        value = "poor"
    elif 41 <= end_result <= 60:
        value = "average"
    elif 61 <= end_result <= 80:
        value = "good"
    elif 81 <= end_result <= 100:
        value = "excellent"

    print(f"Satisfaction Score: {satisfaction_score}")
    print(f"Normalized Satisfaction Score: {normalized_score}")
   

    return jsonify({'redirect_url': url_for('loading')})

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording
    recording = True
    threading.Thread(target=record_audio).start() 
    return jsonify({'status': 'recording started'}), 200

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording
    recording = False

    return send_file(audio_filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)