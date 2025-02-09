import os
import io
import wave
import threading
import numpy as np
import librosa
import tensorflow as tf
import streamlit as st
import pyaudio
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from PIL import Image

# Load the pretrained VGG16 model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze most layers and leave last few trainable
for layer in base_model.layers[:-5]:
    layer.trainable = False

# Add custom classification layers
x = Flatten()(base_model.output)
x = Dense(256, activation="relu", kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)  # Prevent overfitting
x = Dense(8, activation="softmax")(x)  # 8 emotion classes

model = Model(inputs=base_model.input, outputs=x)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Callbacks for regularization
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)

def extract_mel_spectrogram(audio_bytes):
    audio_stream = io.BytesIO(audio_bytes)
    with wave.open(audio_stream, 'rb') as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    y = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    # Standardize audio length
    target_length = 16000  # Standard length for speech models (1 second)
    if len(y) > target_length:
        y = y[:target_length]  # Trim
    else:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')  # Pad

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Convert to image
    fig, ax = plt.subplots(figsize=(3, 3), dpi=80)
    ax.axis('off')
    librosa.display.specshow(mel_spec_db, sr=sr, cmap='viridis')
    plt.tight_layout(pad=0)
    
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)

    img = Image.fromarray(img)
    img = img.resize((224, 224))
    
    return np.expand_dims(np.array(img) / 255.0, axis=0)  # Normalize & add batch dim

def predict_emotion(audio_bytes):
    features = extract_mel_spectrogram(audio_bytes)
    num_samples = 5  # Run multiple predictions
    predictions = np.array([model.predict(features) for _ in range(num_samples)])
    avg_prediction = np.mean(predictions, axis=0)
    emotion_label = np.argmax(avg_prediction)

    emotion_map = {0: "neutral", 1: "calm", 2: "happy", 3: "sad", 4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"}
    return emotion_map.get(emotion_label, "Unknown")

def start_recording():
    global recording, frames
    recording = True
    frames = []

    def record():
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100,
                            input=True, frames_per_buffer=1024)
        while recording:
            data = stream.read(1024)
            frames.append(data)
        stream.stop_stream()
        stream.close()
    
    thread = threading.Thread(target=record)
    thread.start()

def stop_recording():
    global recording
    recording = False
    audio_bytes = b''.join(frames)
    audio_stream = io.BytesIO()
    with wave.open(audio_stream, 'wb') as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wave_file.setframerate(44100)
        wave_file.writeframes(audio_bytes)
    return audio_stream.getvalue()

st.title("Speech Emotion Detection with VGG16")
st.write("🎤 Click **Start Recording** to begin speaking, and **Stop Recording** to analyze.")

if st.button("Start Recording 🎙️"):
    start_recording()
    st.write("Recording... Speak now!")

if st.button("Stop Recording ⏹️"):
    audio_bytes = stop_recording()
    predicted_emotion = predict_emotion(audio_bytes)
    st.write(f"Predicted Emotion: **{predicted_emotion}**")

st.write("---")
st.write("📂 **Upload an audio file** for emotion detection.")

uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if uploaded_file:
    audio_bytes = uploaded_file.read()
    predicted_emotion = predict_emotion(audio_bytes)
    st.write(f"Predicted Emotion: **{predicted_emotion}**")

st.write('---')
