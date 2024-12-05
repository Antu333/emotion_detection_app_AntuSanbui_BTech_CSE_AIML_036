import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, Response, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

model_path = os.path.join(os.path.dirname(__file__), 'fer_model.h5')
dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'fer2013.csv')
csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_emotions.csv')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
csv_path = "C:\\Users\\Antu Sanbui\\Desktop\\project\\emotion detection app\\dataset\\fer2013.csv"

def load_fer2013_data(csv_path):
    print("Loading and preprocessing dataset")
    data = pd.read_csv(csv_path)
    pixels = data['pixels'].tolist()
    emotions = data['emotion'].values
    images = np.array([np.fromstring(pixel, sep=' ') for pixel in pixels], dtype=np.float32)
    images = images.reshape(-1, 48, 48, 1) / 255.0  
    labels = to_categorical(emotions, num_classes=7)  
    return images, labels

def train_fer_model():
    print("Training the model...")
    images, labels = load_fer2013_data(dataset_path)
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)
    model.save(model_path)
    print(f"Model trained and saved as {model_path}")
    return model

if not os.path.exists(model_path):
    model = train_fer_model()
else:
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)  
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = gray_frame[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48)) / 255.0
                face = np.expand_dims(face, axis=0).reshape(1, 48, 48, 1)

                predictions = model.predict(face)
                emotion = emotion_labels[np.argmax(predictions)]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    cap = cv2.VideoCapture(0)  
    ret, frame = cap.read()
    if ret:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_image.jpg')
        cv2.imwrite(file_path, frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        emotion = "No Face Detected"
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = gray_frame[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48)) / 255.0
                face = np.expand_dims(face, axis=0).reshape(1, 48, 48, 1)
                predictions = model.predict(face)
                emotion = emotion_labels[np.argmax(predictions)]

        data = {'Image': [file_path], 'Emotion': [emotion]}
        df = pd.DataFrame(data)
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)

        cap.release()
        return jsonify({'message': 'Image captured successfully', 'emotion': emotion})
    cap.release()
    return jsonify({'message': 'Failed to capture image'}), 400

if __name__ == '__main__':
    app.run(debug=True)
