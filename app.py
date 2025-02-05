# from flask import Flask, render_template, request, jsonify
# import librosa
# import numpy as np
# import tensorflow as tf
# import shutil
# import os
# import torch
# from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# app = Flask(__name__)

# processor = Wav2Vec2Processor.from_pretrained('./saved_model')
# model = Wav2Vec2ForSequenceClassification.from_pretrained('./saved_model')
# model.eval()

# emotion_labels = ["sad", "fear", "disgust", "happy", "pleasant_surprise", "anger", "neutral"]  # Modify based on dataset
# def preprocess_audio(audio_path, processor, max_length=32000):
#     speech, sr = librosa.load(audio_path, sr=16000)  # Load audio at 16kHz

#     # Truncate or pad the audio to the required length
#     if len(speech) > max_length:
#         speech = speech[:max_length]
#     else:
#         speech = np.pad(speech, (0, max_length - len(speech)), 'constant')

#     # Process input
#     input_data = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
#     return input_data.input_values
# def predict_emotion(audio_path):
#     input_values = preprocess_audio(audio_path, processor)

#     # Ensure input has batch dimension [1, sequence_length]
#     input_values = input_values.to(model.device)

#     # Get model output
#     with torch.no_grad():
#         logits = model(input_values).logits

#     predicted_label = torch.argmax(logits, dim=1).item()  # Get predicted class index
#     return predicted_label

# @app.route("/")
# def home():
#     return render_template("index.html")  # Serve the HTML page

# @app.route('/upload-audio', methods=['POST'])
# def upload_audio():
#     if 'audio' not in request.files:
#         return jsonify({"error": "No audio file found in the request."}), 400

#     audio_file = request.files['audio']
#     audio_file.save('uploaded_audio.wav')  
#     predicted_label = predict_emotion('uploaded_audio.wav')
#     predicted_emotion = emotion_labels[predicted_label]
#     print(predicted_emotion)
#     return jsonify({"predicted_emotion": predicted_emotion}), 200


# if __name__ == '__main__':
#     app.run(debug=True)
import os
import cv2
import numpy as np
import librosa
import torch
import shutil
from flask import Flask, request, render_template, Response, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

app = Flask(__name__)

# Load models for video and audio processing
video_model = load_model("model.h5")
audio_processor = Wav2Vec2Processor.from_pretrained('./saved_model')
audio_model = Wav2Vec2ForSequenceClassification.from_pretrained('./saved_model')
audio_model.eval()

labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
audio_labels = ["sad", "fear", "disgust", "happy", "pleasant_surprise", "anger", "neutral"]

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Global variables
cap = None
is_streaming = False
uploaded_video_path = None

# Video Processing
def preprocess_frame(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    processed_faces = []
    face_boxes = []
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        processed_faces.append(face)
        face_boxes.append((x, y, w, h))
    return processed_faces, face_boxes

@app.route('/')
def index():
    return render_template("index.html")

def generate_frames(video_path=None):
    global cap, is_streaming
    cap = cv2.VideoCapture(video_path) if video_path else None
    
    while is_streaming:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces, boxes = preprocess_frame(frame)
        for face, box in zip(faces, boxes):
            pred = video_model.predict(face)[0]
            emotion = labels[np.argmax(pred)]
            (x, y, w, h) = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
def generate_live_stream_frames():
    global cap
    cap = cv2.VideoCapture(0)  # Access the webcam

    while is_streaming:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform face detection and emotion prediction here
        faces, boxes = preprocess_frame(frame)
        for face, box in zip(faces, boxes):
            pred = video_model.predict(face)[0]
            emotion = labels[np.argmax(pred)]
            (x, y, w, h) = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Send frame as a response to the client
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/upload-video', methods=['POST'])
def upload_video():
    global uploaded_video_path, is_streaming
    file = request.files.get('file')
    if not file or file.filename == '':
        return "No file selected", 400
    
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)
    uploaded_video_path = file_path
    is_streaming = True
    return "Video uploaded successfully", 200
@app.route('/start-video', methods=['POST'])
def start_video():
    
    global is_streaming
    # global is_streaming
    if not is_streaming:
        is_streaming = True
    # If no video path (for live stream), start webcam feed
    if not uploaded_video_path:
        is_streaming = True
        return Response(generate_live_stream_frames(), mimetype='multipart/x-mixed-replace;boundary=frame')

    # Ensure JSON response in case of failure
    return jsonify({"error": "Invalid video source"}), 400  # Sending a proper error message

@app.route('/start-video-live', methods=['POST'])
def start_video_live():
    
    global is_streaming
    is_streaming = True
    return 'live Video started'  ,200
    # return jsonify({"error": "Invalid video source"}), 400  # Sending a proper error message

@app.route('/video-stream-live')
def video_stream_live():
   
    if not uploaded_video_path:
        return Response(generate_live_stream_frames(), mimetype='multipart/x-mixed-replace;boundary=frame')
    return "No video uploaded", 400

@app.route('/video-stream')
def video_stream():
    if uploaded_video_path:
        return Response(generate_frames(uploaded_video_path), mimetype='multipart/x-mixed-replace;boundary=frame')
    return "No video uploaded", 400

@app.route('/clear-uploaded-video', methods=['POST'])
def clear_uploaded_video():
    global uploaded_video_path
    uploaded_video_path = None
    return "Uploaded video cleared", 200

@app.route('/stop-video', methods=['POST'])
def stop_video():
    global is_streaming, cap
    if is_streaming:
        is_streaming = False
        if cap:
            cap.release()
            cap = None
        uploaded_video_path = None
        return "Video stream stopped", 200
    return "No video stream running", 400

@app.route('/stop-video-live', methods=['POST'])
def stop_video_live():
    global is_streaming, cap
    if is_streaming:
        is_streaming = False
        if cap:
            cap.release()
            cap = None
        return "Video stream stopped", 200
    return "No video stream running", 400



# Audio Processing
def preprocess_audio(audio_path):
    speech, _ = librosa.load(audio_path, sr=16000)
    speech = np.pad(speech, (0, max(32000 - len(speech), 0)), 'constant')[:32000]
    return audio_processor(speech, sampling_rate=16000, return_tensors="pt", padding=True).input_values

def predict_emotion(audio_path):
    input_values = preprocess_audio(audio_path).to(audio_model.device)
    with torch.no_grad():
        logits = audio_model(input_values).logits
    return audio_labels[torch.argmax(logits, dim=1).item()]

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({"error": "No audio file found."}), 400
    
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], 'uploaded_audio.wav')
    audio_file.save(file_path)
    predicted_emotion = predict_emotion(file_path)
    return jsonify({"predicted_emotion": predicted_emotion}), 200

if __name__ == '__main__':
    app.run(debug=True)