import os
import tempfile
import pickle
import numpy as np
import pandas as pd
import librosa
import librosa.display
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import gdown
import time
from datetime import datetime
import pytz
import sounddevice as sd
from scipy.io.wavfile import write
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Emotion Voice Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Google-Drive file IDs ----------------
MODEL_FILE_ID   = "1rNjobTEVv7iylFSPinYdohpZS-rs5YRJ"
SCALER_FILE_ID  = "1aEN09Avk2G9_Oso7Nl3kLit4cCFphtMo"
ENCODER_FILE_ID = "1INMy6xujSKvW__RAeVUEJRjTqvPdzeQ8"
# -------------------------------------------------------
MODEL_PATH        = "CNN_full_model.h5"
SCALER_PATH       = "scaler2.pickle"
ENCODER_PATH      = "encoder2.pickle"
TARGET_FEATURE_LEN = 2376          # length used during training

# ───────── Helper Functions ─────────────────────────────────────

@st.cache_resource(show_spinner=False)
def download_once(file_id: str, out_path: str):
    """Download a file from Google Drive if it doesn't exist locally."""
    if not os.path.exists(out_path):
        with st.spinner(f"Downloading {os.path.basename(out_path)}..."):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, out_path, quiet=False)
    return out_path

@st.cache_resource(show_spinner="Loading model and assets...")
def load_assets():
    """Load the model, scaler, and encoder."""
    download_once(MODEL_FILE_ID,   MODEL_PATH)
    download_once(SCALER_FILE_ID,  SCALER_PATH)
    download_once(ENCODER_FILE_ID, ENCODER_PATH)
    
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH,  "rb") as f: scaler  = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f: encoder = pickle.load(f)
    return model, scaler, encoder

def extract_features(y, sr):
    """Extract audio features: zero crossing rate, RMS energy, and MFCCs."""
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    
    # Root mean square energy
    rms = librosa.feature.rms(y=y)[0]
    
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T.flatten()
    
    # Combine all features
    return np.hstack([zcr, rms, mfcc])

def prepare_example(path, scaler):
    """Load audio file, extract features, and prepare for prediction."""
    y, sr = librosa.load(path, duration=2.5, offset=0.6)
    feat = extract_features(y, sr)
    
    # Pad or truncate to fixed length
    feat = np.pad(feat, (0, max(0, TARGET_FEATURE_LEN - len(feat))))[:TARGET_FEATURE_LEN]
    feat = scaler.transform(feat.reshape(1, -1))
    return np.expand_dims(feat, axis=2)       # (1, 2376, 1)

def get_emotion_emoji(emotion):
    """Return emoji corresponding to detected emotion."""
    emoji_map = {
        'angry': '😠',
        'disgust': '🤢',
        'fear': '😨',
        'happy': '😄',
        'neutral': '😐',
        'ps': '😌',  # peaceful/calm
        'sad': '😢'
    }
    return emoji_map.get(emotion.lower(), '❓')

def get_emotion_color(emotion):
    """Return color corresponding to detected emotion."""
    color_map = {
        'angry': '#FF5733',     # Red
        'disgust': '#66FF66',   # Green
        'fear': '#9370DB',      # Purple
        'happy': '#FFFF00',     # Yellow
        'neutral': '#A9A9A9',   # Gray
        'ps': '#87CEEB',        # Light Blue
        'sad': '#0000FF'        # Blue
    }
    return color_map.get(emotion.lower(), '#000000')

def get_emotion_description(emotion):
    """Return description of detected emotion."""
    descriptions = {
        'angry': "Anger is characterized by tension and hostility arising from frustration, real or perceived wrong.",
        'disgust': "Disgust is an emotional response of revulsion to something considered offensive or unpleasant.",
        'fear': "Fear is a distressing emotion aroused by a perceived threat.",
        'happy': "Happiness is an emotional state characterized by feelings of joy, satisfaction, contentment, and fulfillment.",
        'neutral': "Neutral emotion indicates a calm, balanced emotional state without strong positive or negative feelings.",
        'ps': "Peaceful/calm emotions reflect tranquility, serenity, and a relaxed state of mind.",
        'sad': "Sadness is an emotional pain associated with feelings of disadvantage, loss, despair, grief, helplessness, and sorrow."
    }
    return descriptions.get(emotion.lower(), "No description available.")

def record_audio(duration=3, sample_rate=22050):
    """Record audio from microphone."""
    st.write("🎙️ Recording...")
    progress_bar = st.progress(0)
    
    # Create a recording placeholder
    recording_placeholder = st.empty()
    recording_placeholder.markdown("🔴 Recording in progress...")
    
    # Record audio
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    
    # Show progress
    for i in range(100):
        time.sleep(duration/100)
        progress_bar.progress(i + 1)
    
    sd.wait()
    recording_placeholder.markdown("✅ Recording complete!")
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_file.name, sample_rate, audio_data)
    
    return temp_file.name

def create_spectrogram_custom(y, sr):
    """Create a customized spectrogram with better coloring."""
    # Create a custom colormap
    colors = [(0.2, 0.2, 0.6), (0.6, 0.8, 0.9), (1, 1, 0.8), (1, 0.7, 0.4), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
    
    plt.figure(figsize=(10, 4))
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='mel', cmap=cmap)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    
    # Convert plot to image
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    plt.close()
    
    return buffer

def get_history_dataframe():
    """Get or create a history dataframe to store previous analysis results."""
    if 'history' not in st.session_state:
        st.session_state.history = pd.DataFrame(columns=[
            'Timestamp', 'Audio File', 'Detected Emotion', 'Confidence'
        ])
    return st.session_state.history

def add_to_history(filename, emotion, confidence):
    """Add a new entry to the history dataframe."""
    history_df = get_history_dataframe()
    new_row = pd.DataFrame({
        'Timestamp': [datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S UTC')],
        'Audio File': [os.path.basename(filename)],
        'Detected Emotion': [emotion],
        'Confidence': [f"{confidence:.2%}"]
    })
    st.session_state.history = pd.concat([new_row, history_df]).reset_index(drop=True)

def get_base64_of_bin_file(file_path):
    """Get base64 encoded binary file for embedding in page."""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def get_waveform_plot(y, sr):
    """Create a waveform plot for the audio."""
    plt.figure(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, alpha=0.8)
    plt.title('Waveform')
    plt.tight_layout()
    
    # Convert plot to image
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    plt.close()
    
    return buffer

# ───────── Custom CSS ─────────────────────────────────────

def apply_custom_css():
    """Apply custom CSS styling to the application."""
    st.markdown("""
    <style>
        /* App-wide styling */
        .main {
            background-color: #f5f7fa;
        }
        
        .stApp {
            font-family: 'Arial', sans-serif;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #1E3A8A;
            font-weight: 600;
        }
        
        /* Card-like containers */
        .css-card {
            border-radius: 10px;
            padding: 20px;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        
        /* Button styling */
        .stButton>button {
            border-radius: 5px;
            border: 1px solid #4CAF50;
            padding: 10px 24px;
            background-color: #4CAF50;
            color: white;
            font-weight: 500;
        }
        
        .stButton>button:hover {
            background-color: #45a049;
            border: 1px solid #45a049;
        }
        
        /* Styling for results section */
        .emotion-result {
            font-size: 24px;
            font-weight: bold;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            margin: 10px 0;
        }
        
        /* Custom tabs and containers */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #f1f3f9;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 16px;
            border: none;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: white !important;
            border-top: 2px solid #4CAF50;
            border-left: 1px solid #ddd;
            border-right: 1px solid #ddd;
            border-bottom: none;
        }
        
        /* Upload button styling */
        .uploadedFile {
            border-radius: 5px;
            border: 1px solid #ccc;
            padding: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

# ───────── Main Application ─────────────────────────────────────

def main():
    # Apply custom CSS
    apply_custom_css()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/listening--v1.png", width=80)
        st.title("Emotion Voice Analyzer")
        st.markdown("---")
        
        st.subheader("🔍 About")
        st.markdown("""
        This application analyzes speech audio to detect emotions. Upload an audio file or record yourself speaking to determine the emotional tone.
        
        The AI model recognizes these emotions:
        - 😠 Angry
        - 🤢 Disgust
        - 😨 Fear
        - 😄 Happy
        - 😐 Neutral
        - 😌 Peaceful/Calm
        - 😢 Sad
        """)
        
        st.markdown("---")
        st.subheader("⚙️ Settings")
        
        # Audio settings
        st.subheader("Audio Settings")
        duration = st.slider("Recording Duration (seconds)", 2, 10, 3)
        audio_offset = st.slider("Audio Offset (seconds)", 0.0, 1.0, 0.6, 0.1, 
                               help="Start point for analysis in the audio file")
        
        # Visualization settings
        st.subheader("Visualization Settings")
        show_waveform = st.checkbox("Show Waveform", True)
        show_spectrogram = st.checkbox("Show Spectrogram", True)
        show_history = st.checkbox("Show History", True)
        
        st.markdown("---")
        st.caption("© 2025 Emotion Voice Analyzer")
    
    # Main content
    st.title("🎭 Emotion Voice Analyzer")
    st.markdown("Analyze the emotional tone in speech audio using advanced AI")
    
    # Load model and assets
    model, scaler, encoder = load_assets()
    
    # Create tabs for different input methods
    tabs = st.tabs(["📤 Upload Audio", "🎙️ Record Audio", "📊 History"])
    
    # Tab 1: Upload Audio
    with tabs[0]:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.subheader("Upload Audio File")
        st.markdown("Upload a short audio clip (WAV, MP3, or FLAC format) containing speech.")
        
        uploaded = st.file_uploader(
            "Choose an audio file (2-5 seconds of speech recommended)", 
            type=["wav", "wave", "flac", "mp3"], 
            key="file_uploader"
        )
        
        analyze_button = st.button("Analyze Uploaded Audio", key="analyze_upload")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded is not None and analyze_button:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            
            with st.spinner("Analyzing audio..."):
                # Process the audio file
                process_audio_file(tmp_path, model, scaler, encoder, show_waveform, show_spectrogram)
    
    # Tab 2: Record Audio
    with tabs[1]:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.subheader("Record Your Voice")
        st.markdown(f"Record {duration} seconds of audio using your microphone to analyze the emotional tone.")
        
        record_col1, record_col2 = st.columns([1, 1])
        with record_col1:
            record_button = st.button("🎙️ Start Recording", key="record_button")
        
        if record_button:
            try:
                audio_path = record_audio(duration=duration)
                st.audio(audio_path, format="audio/wav")
                
                with st.spinner("Analyzing your recording..."):
                    # Process the audio file
                    process_audio_file(audio_path, model, scaler, encoder, show_waveform, show_spectrogram)
            except Exception as e:
                st.error(f"Error recording audio: {e}")
                st.info("Make sure your microphone is properly connected and permissions are granted to the browser.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: History
    with tabs[2]:
        if show_history:
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            st.subheader("Analysis History")
            
            history_df = get_history_dataframe()
            if len(history_df) > 0:
                st.dataframe(history_df, use_container_width=True)
                
                # Add download button for history
                csv = history_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download History as CSV",
                    csv,
                    "emotion_analysis_history.csv",
                    "text/csv",
                    key='download-csv'
                )
            else:
                st.info("No analysis history yet. Upload or record audio to see results here.")
            st.markdown('</div>', unsafe_allow_html=True)

def process_audio_file(audio_path, model, scaler, encoder, show_waveform=True, show_spectrogram=True):
    """Process an audio file and display the analysis results."""
    try:
        # Load audio for visualization
        y, sr = librosa.load(audio_path)
        
        # Display audio player
        st.subheader("Audio Sample")
        st.audio(audio_path)
        
        # Create columns for visualizations
        if show_waveform or show_spectrogram:
            st.subheader("Audio Visualizations")
            vis_cols = st.columns([1, 1])
            
            # Display waveform
            if show_waveform:
                with vis_cols[0]:
                    waveform_img = get_waveform_plot(y, sr)
                    st.image(waveform_img, use_column_width=True)
            
            # Display spectrogram
            if show_spectrogram:
                with vis_cols[1]:
                    spectrogram_img = create_spectrogram_custom(y, sr)
                    st.image(spectrogram_img, use_column_width=True)
        
        # Prepare audio features for prediction
        with st.spinner("Extracting audio features..."):
            feat = prepare_example(audio_path, scaler)
        
        # Make prediction
        with st.spinner("Analyzing emotion..."):
            probs = model.predict(feat, verbose=0)[0]
            emotion_label = encoder.inverse_transform([np.argmax(probs).reshape(1)])[0][0]
            confidence = np.max(probs)
            
            # Add to history
            add_to_history(audio_path, emotion_label, confidence)
        
        # Display results
        st.markdown("---")
        st.subheader("Analysis Results")
        
        # Create columns for results
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            emoji = get_emotion_emoji(emotion_label)
            emotion_color = get_emotion_color(emotion_label)
            
            st.markdown(f"""
            <div class="emotion-result" style="background-color: {emotion_color}20; color: {emotion_color}">
                {emoji} {emotion_label.upper()} {emoji}
            </div>
            <p style="text-align: center; font-size: 18px;">
                Confidence: <strong>{confidence:.2%}</strong>
            </p>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.markdown("### Emotion Distribution")
            # Create a DataFrame for chart display
            emotion_df = pd.DataFrame({
                'Emotion': encoder.categories_[0],
                'Probability': probs
            })
            emotion_df = emotion_df.sort_values('Probability', ascending=False)
            
            # Custom bar chart with colors
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(emotion_df['Emotion'], emotion_df['Probability'], color=[get_emotion_color(e) for e in emotion_df['Emotion']])
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.set_title('Emotion Probability Distribution')
            
            # Add percentage labels
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + 0.01
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1%}',
                        va='center', fontsize=10)
            
            st.pyplot(fig)
        
        # Display emotion description
        st.markdown(f"""
        <div style="background-color: {emotion_color}10; border-left: 5px solid {emotion_color}; padding: 15px; border-radius: 5px;">
            <h4>About {emotion_label.capitalize()} Emotion:</h4>
            <p>{get_emotion_description(emotion_label)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Remove the temporary file
        os.unlink(audio_path)
        
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        st.info("Please try uploading a different audio file or recording again.")

if __name__ == "__main__":
    main()
