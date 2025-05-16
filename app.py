import os
import tempfile
import pickle
import numpy as np
import librosa
import librosa.display
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import gdown

# ---------------- Google-Drive file IDs ----------------
MODEL_FILE_ID   = "1rNjobTEVv7iylFSPinYdohpZS-rs5YRJ"
SCALER_FILE_ID  = "1aEN09Avk2G9_Oso7Nl3kLit4cCFphtMo"
ENCODER_FILE_ID = "1INMy6xujSKvW__RAeVUEJRjTqvPdzeQ8"
# -------------------------------------------------------

MODEL_PATH        = "CNN_full_model.h5"
SCALER_PATH       = "scaler2.pickle"
ENCODER_PATH      = "encoder2.pickle"
TARGET_FEATURE_LEN = 2376          # length used during training

# ───────── helpers ──────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _download_once(file_id: str, out_path: str):
    if not os.path.exists(out_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, out_path, quiet=False)
    return out_path


@st.cache_resource(show_spinner="Loading model …")
def load_assets():
    _download_once(MODEL_FILE_ID,   MODEL_PATH)
    _download_once(SCALER_FILE_ID,  SCALER_PATH)
    _download_once(ENCODER_FILE_ID, ENCODER_PATH)

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH,  "rb") as f: scaler  = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f: encoder = pickle.load(f)
    return model, scaler, encoder


def extract_features(y, sr):
    zcr  = librosa.feature.zero_crossing_rate(y=y)[0]
    rms  = librosa.feature.rms(y=y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T.flatten()
    return np.hstack([zcr, rms, mfcc])


def prepare_example(path, scaler):
    y, sr = librosa.load(path, duration=2.5, offset=0.6)
    feat  = extract_features(y, sr)

    # Pad / truncate to fixed length
    feat = np.pad(feat, (0, max(0, TARGET_FEATURE_LEN - len(feat))))[:TARGET_FEATURE_LEN]

    feat = scaler.transform(feat.reshape(1, -1))
    return np.expand_dims(feat, axis=2)       # (1, 2376, 1)

# ───────── Streamlit UI ─────────────────────────────────
st.set_page_config(page_title="Speech Emotion Recognizer", layout="centered")
st.title("🎙️ Speech Emotion Recognizer")

model, scaler, encoder = load_assets()

uploaded = st.file_uploader(
    "Upload a WAV file (≈3 s of speech)", type=["wav", "wave", "flac", "mp3"]
)

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.audio(uploaded, format="audio/wav")

    # Waveform & spectrogram
    y, sr = librosa.load(tmp_path)
    fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    ax[0].set(title="Waveform")
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)  # ← y= added
    img  = librosa.power_to_db(spec)
    librosa.display.specshow(img, sr=sr, x_axis="time", y_axis="mel", ax=ax[1])
    ax[1].set(title="Log-Mel Spectrogram")
    st.pyplot(fig)

    # Prediction
    feat  = prepare_example(tmp_path, scaler)
    probs = model.predict(feat, verbose=0)[0]
    label = encoder.inverse_transform([probs])[0][0]

    st.subheader(f"🗣️ Detected emotion: **{label}**")
    st.bar_chart(dict(zip(encoder.categories_[0], probs)), use_container_width=True)

    os.unlink(tmp_path)
