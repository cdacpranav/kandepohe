import os
import io
import wave
import threading
import numpy as np
import librosa
import tensorflow as tf
import streamlit as st
import pyaudio
import cv2

# 🔍 Define Model Path (Cross-Platform)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/Speech_emotion_vgg16_model.h5")

# 🔍 Debug: Check if the model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at `{MODEL_PATH}`.")
    st.write("📂 **Checking models directory contents:**")
    
    MODEL_DIR = os.path.dirname(MODEL_PATH)
    if os.path.exists(MODEL_DIR):
        st.write(f"Files in `{MODEL_DIR}`:", os.listdir(MODEL_DIR))
    else:
        st.write(f"❌ Directory `{MODEL_DIR}` not found!")

    st.write("📌 **Manually upload the model to the `models/` folder.**")
    st.stop()

# 🔥 Load the Model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 🎤 Audio Recording Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

audio = pyaudio.PyAudio()
recording = False
frames = []

# 🔍 Extract Mel Spectrogram for VGG16 Model
def extract_mel_spectrogram(audio_bytes, img_size=(224, 224)):
    try:
        audio_stream = io.BytesIO(audio_bytes)
        with wave.open(audio_stream, 'rb') as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        y = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Resize to VGG16 input size (224x224)
        mel_spec_resized = cv2.resize(mel_spec_db, img_size, interpolation=cv2.INTER_LINEAR)

        # Normalize to range [0,1]
        mel_spec_resized = (mel_spec_resized - mel_spec_resized.min()) / (mel_spec_resized.max() - mel_spec_resized.min())

        # Convert to 3-channel image for VGG16 (RGB-like format)
        mel_spec_rgb = np.stack([mel_spec_resized] * 3, axis=-1)

        return np.expand_dims(mel_spec_rgb, axis=0)  # Add batch dimension
    except Exception as e:
        st.error(f"❌ Error in feature extraction: {e}")
        return None


# 🎤 Start Audio Recording
def start_recording():
    global recording, frames
    recording = True
    frames = []

    def record():
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        while recording:
            data = stream.read(CHUNK)
            frames.append(data)
        stream.stop_stream()
        stream.close()

    thread = threading.Thread(target=record)
    thread.start()


# ⏹ Stop Audio Recording & Process Audio
def stop_recording():
    global recording
    recording = False

    # Convert recorded frames to a bytes object
    audio_bytes = b''.join(frames)

    # Create an in-memory WAV file
    audio_stream = io.BytesIO()
    with wave.open(audio_stream, 'wb') as wave_file:
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)
        wave_file.writeframes(audio_bytes)

    return audio_stream.getvalue()  # Return audio bytes


# 📌 Streamlit UI
st.title("🎙️ Speech Emotion Detection")

st.write("🎤 Click **Start Recording** to begin speaking, and **Stop Recording** to analyze.")

# 🟢 Start Recording Button
if st.button("Start Recording 🎙️"):
    start_recording()
    st.write("🎤 Recording... Speak now!")

# ⏹ Stop Recording Button & Process Audio
if st.button("Stop Recording ⏹️"):
    audio_bytes = stop_recording()

    # Extract Features & Make Prediction
    features = extract_mel_spectrogram(audio_bytes)

    if features is not None:
        try:
            prediction = model.predict(features)
            emotion_label = np.argmax(prediction)

            # 🎭 Emotion Mapping
            emotion_map = {
                0: "neutral", 1: "calm", 2: "happy", 3: "sad",
                4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
            }
            st.success(f"🎭 **Predicted Emotion:** {emotion_map.get(emotion_label, 'Unknown')}")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# 📂 File Upload Option
st.write("---")
st.write("📂 **Upload an audio file** for emotion detection.")

uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if uploaded_file:
    audio_bytes = uploaded_file.read()

    features = extract_mel_spectrogram(audio_bytes)

    if features is not None:
        try:
            prediction = model.predict(features)
            emotion_label = np.argmax(prediction)

            emotion_map = {
                0: "neutral", 1: "calm", 2: "happy", 3: "sad",
                4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
            }
            st.success(f"🎭 **Predicted Emotion:** {emotion_map.get(emotion_label, 'Unknown')}")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

st.write("\n" * 10)
st.write("---")
st.write("**Project by**")
st.write("[🔗 Pranav Harke](https://www.linkedin.com/in/pranavharke)")
st.write("[🔗 Parvej Pathan](https://www.linkedin.com/in/parvejkhan-pathan-891527337/)")
