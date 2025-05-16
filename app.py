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
import pandas as pd
from datetime import datetime
import pytz

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
def download_once(file_id: str, out_path: str):
    if not os.path.exists(out_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, out_path, quiet=False)
    return out_path

@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    download_once(MODEL_FILE_ID,   MODEL_PATH)
    download_once(SCALER_FILE_ID,  SCALER_PATH)
    download_once(ENCODER_FILE_ID, ENCODER_PATH)
    
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH,  "rb") as f: scaler  = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f: encoder = pickle.load(f)
    return model, scaler, encoder

def extract_features(y, sr):
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    
    # Root mean square energy
    rms = librosa.feature.rms(y=y)[0]
    
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T.flatten()
    
    return np.hstack([zcr, rms, mfcc])

def prepare_example(path, scaler):
    y, sr = librosa.load(path, duration=2.5, offset=0.6)
    feat = extract_features(y, sr)
    
    # Pad / truncate to fixed length
    feat = np.pad(feat, (0, max(0, TARGET_FEATURE_LEN - len(feat))))[:TARGET_FEATURE_LEN]
    feat = scaler.transform(feat.reshape(1, -1))
    return np.expand_dims(feat, axis=2)       # (1, 2376, 1)

def get_color_for_emotion(emotion):
    """Return a color based on the emotion for consistent visualization"""
    colors = {
        'angry': '#FF5252',      # Red
        'disgust': '#8BC34A',    # Light Green
        'fear': '#7B1FA2',       # Purple
        'happy': '#FFC107',      # Amber
        'neutral': '#78909C',    # Blue Grey
        'sad': '#3F51B5',        # Indigo
        'surprise': '#FF9800',   # Orange
        'calm': '#009688',       # Teal
        # Add more emotions as needed
    }
    # Default color if the emotion is not in our mapping
    return colors.get(emotion.lower(), '#607D8B')  # Default: Blue Grey

def create_waveform_and_spectrogram(y, sr):
    """Create a better-styled waveform and spectrogram plot"""
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'hspace': 0.3})
    
    # Waveform plot
    librosa.display.waveshow(y, sr=sr, ax=ax[0], alpha=0.8)
    ax[0].set_title("Waveform", fontsize=14, fontweight='bold')
    ax[0].set_ylabel("Amplitude", fontsize=12)
    ax[0].grid(True, alpha=0.3)
    
    # Spectrogram
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    img = librosa.power_to_db(spec)
    librosa.display.specshow(img, sr=sr, x_axis="time", y_axis="mel", ax=ax[1], cmap='viridis')
    ax[1].set_title("Log-Mel Spectrogram", fontsize=14, fontweight='bold')
    ax[1].set_xlabel("Time (s)", fontsize=12)
    ax[1].set_ylabel("Frequency (Hz)", fontsize=12)
    
    fig.tight_layout()
    return fig

def create_emotion_chart(emotions, probabilities, detected_emotion):
    """Create a nicer visualization for emotion probabilities"""
    df = pd.DataFrame({
        'Emotion': emotions,
        'Probability': probabilities
    })
    df = df.sort_values('Probability', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        df['Emotion'], 
        df['Probability'], 
        color=[get_color_for_emotion(emotion) if emotion == detected_emotion 
              else get_color_for_emotion(emotion) for emotion in df['Emotion']]
    )
    
    # Highlight detected emotion
    for i, bar in enumerate(bars):
        if df['Emotion'].iloc[i] == detected_emotion:
            bar.set_alpha(1.0)
            bar.set_edgecolor('black')
            bar.set_linewidth(2)
        else:
            bar.set_alpha(0.7)
    
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Emotion Probability Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add data labels
    for i, v in enumerate(df['Probability']):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=10)
        
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def save_to_history(emotion, prob_dict, audio_length):
    """Save prediction to history dataframe"""
    timestamp = datetime.now(pytz.timezone('UTC')).strftime("%Y-%m-%d %H:%M:%S")
    
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = pd.DataFrame(
            columns=['Timestamp', 'Detected Emotion', 'Top Probability', 'Audio Length (s)']
        )
    
    new_row = pd.DataFrame({
        'Timestamp': [timestamp],
        'Detected Emotion': [emotion],
        'Top Probability': [max(prob_dict.values())],
        'Audio Length (s)': [audio_length]
    })
    
    st.session_state.prediction_history = pd.concat([new_row, st.session_state.prediction_history]).reset_index(drop=True)

# ───────── Streamlit UI ─────────────────────────────────

def set_app_style():
    """Set custom styling for the app"""
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.8rem;
            font-weight: 600;
            margin-top: 1.5rem;
        }
        .emotion-header {
            font-size: 2rem;
            font-weight: 700;
            margin-top: 1rem;
            margin-bottom: 1rem;
            padding: 0.5rem;
            border-radius: 10px;
            text-align: center;
        }
        .info-text {
            font-size: 1rem;
            color: #555;
        }
        .stAudio {
            margin-top: 1rem;
            margin-bottom: 1rem;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 10px;
        }
        .history-section {
            margin-top: 2rem;
            padding: 1rem;
            border-radius: 10px;
            background-color: #f5f5f5;
        }
        .instructions {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    # Set page config
    st.set_page_config(
        page_title="Speech Emotion Analyzer",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    set_app_style()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🛠️ Options")
        
        st.markdown("### Audio Settings")
        audio_duration = st.slider(
            "Audio Duration (seconds)",
            min_value=1.0,
            max_value=5.0,
            value=2.5,
            step=0.5,
            help="Select how many seconds of audio to analyze"
        )
        
        audio_offset = st.slider(
            "Audio Offset (seconds)",
            min_value=0.0,
            max_value=2.0,
            value=0.6,
            step=0.1,
            help="Select the starting point in the audio file"
        )
        
        st.markdown("### Display Options")
        show_waveform = st.checkbox("Show Waveform", value=True)
        show_spectrogram = st.checkbox("Show Spectrogram", value=True)
        show_history = st.checkbox("Show Prediction History", value=True)
        
        st.markdown("---")
        st.markdown("""
        ### About
        This app uses a CNN model to detect emotions in speech. 
        Upload a short audio clip to analyze the emotional content!
        
        Made with:
        - TensorFlow
        - Librosa
        - Streamlit
        """)

    # Main content
    st.markdown('<h1 class="main-header">🎙️ Speech Emotion Analyzer</h1>', unsafe_allow_html=True)
    
    # Load model and assets
    model, scaler, encoder = load_assets()
    
    # Instructions
    with st.expander("ℹ️ Instructions", expanded=True):
        st.markdown("""
        <div class="instructions">
        <p>1. Upload an audio file (WAV, FLAC, or MP3) containing a short speech segment.</p>
        <p>2. The model works best with clear speech containing emotional content.</p>
        <p>3. Adjust the audio duration and offset in the sidebar if needed.</p>
        <p>4. The app will display the detected emotion along with probability scores.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # File uploader
    uploaded = st.file_uploader(
        "Upload an audio file (speech)",
        type=["wav", "wave", "flac", "mp3"],
        help="Upload a clear speech recording for best results"
    )
    
    col1, col2 = st.columns([3, 2])
    
    if uploaded is not None:
        with st.spinner("Processing audio..."):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            
            with col1:
                st.markdown('<h3 class="sub-header">Audio Input</h3>', unsafe_allow_html=True)
                st.audio(uploaded, format="audio/wav")
                
                # Load audio and get basic info
                y, sr = librosa.load(tmp_path, sr=None)
                audio_length = len(y) / sr
                
                st.markdown(f"""
                <div class="info-text">
                    <strong>File:</strong> {uploaded.name}<br>
                    <strong>Sample Rate:</strong> {sr} Hz<br>
                    <strong>Duration:</strong> {audio_length:.2f} seconds
                </div>
                """, unsafe_allow_html=True)
                
                if show_waveform or show_spectrogram:
                    # Create tabs for audio visualizations
                    tab1, tab2 = st.tabs(["Waveform", "Spectrogram"])
                    
                    # Waveform
                    if show_waveform:
                        with tab1:
                            fig, ax = plt.subplots(figsize=(10, 3))
                            librosa.display.waveshow(y, sr=sr, ax=ax)
                            ax.set_title("Audio Waveform")
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                    
                    # Spectrogram            
                    if show_spectrogram:
                        with tab2:
                            fig, ax = plt.subplots(figsize=(10, 3))
                            spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
                            img = librosa.power_to_db(spec)
                            librosa.display.specshow(img, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap='viridis')
                            ax.set_title("Log-Mel Spectrogram")
                            fig.colorbar(ax.collections[0], ax=ax, format="%+2.0f dB")
                            st.pyplot(fig)
            
            with col2:
                # Prediction
                feat = prepare_example(tmp_path, scaler)
                probs = model.predict(feat, verbose=0)[0]
                label = encoder.inverse_transform([np.argmax(probs, axis=0).reshape(1)])[0][0]
                prob_dict = dict(zip(encoder.categories_[0], probs))
                
                # Save prediction to history
                save_to_history(label, prob_dict, audio_length)
                
                # Show emotion result
                emotion_color = get_color_for_emotion(label)
                st.markdown(
                    f"""
                    <div class="emotion-header" style="background-color: {emotion_color}20; color: {emotion_color};">
                        Detected Emotion: {label.upper()}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Create and display emotion probability chart
                emotion_fig = create_emotion_chart(encoder.categories_[0], probs, label)
                st.pyplot(emotion_fig)
                
                # Add confidence meter
                top_confidence = max(probs)
                st.markdown("### Confidence Level")
                st.progress(float(top_confidence))
                st.markdown(f"<div style='text-align: center; font-size: 1.2rem;'>{top_confidence:.1%}</div>", unsafe_allow_html=True)
            
            os.unlink(tmp_path)
    
    else:
        st.info("👆 Upload an audio file to get started!")
    
    # Display prediction history
    if show_history and 'prediction_history' in st.session_state and not st.session_state.prediction_history.empty:
        st.markdown("---")
        st.markdown('<h3 class="sub-header">Prediction History</h3>', unsafe_allow_html=True)
        st.dataframe(
            st.session_state.prediction_history,
            use_container_width=True,
            column_config={
                'Top Probability': st.column_config.ProgressColumn(
                    "Confidence",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                ),
                'Audio Length (s)': st.column_config.NumberColumn(
                    "Duration (s)",
                    format="%.2f s"
                ),
            }
        )
        
        if st.button("Clear History"):
            st.session_state.prediction_history = pd.DataFrame(
                columns=['Timestamp', 'Detected Emotion', 'Top Probability', 'Audio Length (s)']
            )
            st.experimental_rerun()

if __name__ == "__main__":
    main()
