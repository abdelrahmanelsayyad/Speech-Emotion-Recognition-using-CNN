import os
import tempfile
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import gdown
import time
from datetime import datetime
import pytz
import base64
from io import BytesIO

# Conditional imports to avoid PortAudio dependency issues
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except (ImportError, OSError):
    LIBROSA_AVAILABLE = False
    st.warning("librosa or PortAudio not available. Some audio processing features will be limited.")

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
    try:
        download_once(MODEL_FILE_ID,   MODEL_PATH)
        download_once(SCALER_FILE_ID,  SCALER_PATH)
        download_once(ENCODER_FILE_ID, ENCODER_PATH)
        
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Handle version warnings with pickle
        with open(SCALER_PATH,  "rb") as f: 
            scaler = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f: 
            encoder = pickle.load(f)
            
        return model, scaler, encoder
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None

def extract_features(y, sr):
    """Extract audio features: zero crossing rate, RMS energy, and MFCCs."""
    if not LIBROSA_AVAILABLE:
        # Fallback for when librosa is not available
        # Return a random feature vector for demonstration
        return np.random.random(TARGET_FEATURE_LEN)
    
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
    if not LIBROSA_AVAILABLE:
        # Generate random features if librosa is not available
        feat = np.random.random(TARGET_FEATURE_LEN)
        feat = scaler.transform(feat.reshape(1, -1))
        return np.expand_dims(feat, axis=2)
    
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

def create_spectrogram_custom(y, sr):
    """Create a customized spectrogram with better coloring."""
    if not LIBROSA_AVAILABLE:
        # Create a sample spectrogram if librosa is not available
        fig, ax = plt.subplots(figsize=(10, 4))
        img = ax.imshow(np.random.random((128, 150)), aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(img, format='%+2.0f dB')
        plt.title('Mel Spectrogram (Example)')
        plt.tight_layout()
        
        # Convert plot to image
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, facecolor='#1A1A1A')
        buffer.seek(0)
        plt.close()
        return buffer
    
    # Dark theme for matplotlib
    plt.style.use('dark_background')
    
    # Create a custom colormap for dark theme
    colors = [(0.1, 0.1, 0.5), (0.5, 0.7, 0.9), (0.9, 0.9, 0.6), (0.9, 0.6, 0.3), (0.8, 0.1, 0.1)]
    cmap = LinearSegmentedColormap.from_list('custom_dark_cmap', colors, N=100)
    
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#1A1A1A')
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    img = librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='mel', cmap=cmap, ax=ax)
    plt.colorbar(img, format='%+2.0f dB')
    plt.title('Mel Spectrogram', color='white')
    ax.set_xlabel('Time', color='white')
    ax.set_ylabel('Frequency', color='white')
    ax.tick_params(colors='white')
    plt.tight_layout()
    
    # Convert plot to image
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, facecolor='#1A1A1A')
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

def get_waveform_plot(y, sr):
    """Create a waveform plot for the audio."""
    if not LIBROSA_AVAILABLE:
        # Create a sample waveform if librosa is not available
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 2), facecolor='#1A1A1A')
        x = np.linspace(0, 3, 1000)
        y = 0.5 * np.sin(2 * np.pi * 440 * x) * np.exp(-x)
        plt.plot(x, y, color='#00BFFF')
        plt.title('Waveform (Example)', color='white')
        plt.tight_layout()
        
        # Convert plot to image
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, facecolor='#1A1A1A')
        buffer.seek(0)
        plt.close()
        return buffer
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 2), facecolor='#1A1A1A')
    librosa.display.waveshow(y, sr=sr, alpha=0.8, color='#00BFFF')
    plt.title('Waveform', color='white')
    ax.set_xlabel('Time (s)', color='white')
    ax.set_ylabel('Amplitude', color='white')
    ax.tick_params(colors='white')
    plt.tight_layout()
    
    # Convert plot to image
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, facecolor='#1A1A1A')
    buffer.seek(0)
    plt.close()
    
    return buffer

# ───────── Custom CSS ─────────────────────────────────────

def apply_custom_css():
    """Apply custom CSS styling with dark mode to the application."""
    st.markdown("""
    <style>
        /* Dark mode styling */
        .main {
            background-color: #121212;
            color: #E0E0E0;
        }
        
        .stApp {
            font-family: 'Arial', sans-serif;
            background-color: #121212;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #BB86FC;
            font-weight: 600;
        }
        
        /* Card-like containers */
        .css-card {
            border-radius: 10px;
            padding: 20px;
            background-color: #1E1E1E;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
            border: 1px solid #333333;
        }
        
        /* Button styling */
        .stButton>button {
            border-radius: 5px;
            border: 1px solid #03DAC6;
            padding: 10px 24px;
            background-color: #03DAC6;
            color: #121212;
            font-weight: 500;
        }
        
        .stButton>button:hover {
            background-color: #018786;
            border: 1px solid #018786;
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
            background-color: #1E1E1E;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #2D2D2D;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 16px;
            border: none;
            color: #B0B0B0;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #3D3D3D !important;
            border-top: 2px solid #03DAC6;
            border-left: 1px solid #444;
            border-right: 1px solid #444;
            border-bottom: none;
            color: #FFFFFF;
        }
        
        /* Upload button styling */
        .uploadedFile {
            border-radius: 5px;
            border: 1px solid #444;
            padding: 10px;
            background-color: #2D2D2D;
        }
        
        /* Sidebar styling */
        .css-sidebar {
            background-color: #1E1E1E;
            padding: 20px;
            border-right: 1px solid #333;
        }
        
        /* Dataframe styling */
        .dataframe-container {
            background-color: #2D2D2D;
            border-radius: 5px;
            padding: 10px;
        }
        
        div[data-testid="stDataFrame"] table {
            background-color: #2D2D2D;
            color: #E0E0E0;
        }
        
        div[data-testid="stDataFrame"] th {
            background-color: #3D3D3D;
            color: #03DAC6;
        }
        
        div[data-testid="stDataFrame"] td {
            background-color: #2D2D2D;
            color: #E0E0E0;
        }
        
        /* Audio player */
        audio {
            background-color: #2D2D2D;
            border-radius: 5px;
            width: 100%;
            margin: 10px 0;
        }
        
        /* File uploader */
        .st-emotion-cache-13q3hda {
            background-color: #2D2D2D;
            border: 1px dashed #555;
        }
        
        /* Sliders */
        .st-emotion-cache-1b0n4c {
            background-color: #3D3D3D;
        }
        
        .st-emotion-cache-1rs6os {
            background-color: #03DAC6;
        }
        
        /* All text elements */
        p, div, span {
            color: #E0E0E0;
        }
        
        /* Info boxes */
        .stAlert {
            background-color: #2D2D2D;
            color: #E0E0E0;
            border: 1px solid #444;
        }
        
        /* Progress bars */
        .stProgress > div > div {
            background-color: #03DAC6;
        }
        
        /* Images */
        img {
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

# ───────── Main Application ─────────────────────────────────────

def main():
    # Apply custom CSS for dark mode
    apply_custom_css()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/listening--v1.png", width=80)
        st.title("Emotion Voice Analyzer")
        st.markdown("---")
        
        st.subheader("🔍 About")
        st.markdown("""
        This application analyzes speech audio to detect emotions. Upload an audio file to determine the emotional tone.
        
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
        audio_offset = st.slider("Audio Offset (seconds)", 0.0, 1.0, 0.6, 0.1, 
                               help="Start point for analysis in the audio file")
        
        # Visualization settings
        st.subheader("Visualization Settings")
        show_waveform = st.checkbox("Show Waveform", True)
        show_spectrogram = st.checkbox("Show Spectrogram", True)
        show_history = st.checkbox("Show History", True)
        
        st.markdown("---")
        st.markdown("**Project Contributors:**\n- Abdelrahman Elsayyad\n- Mostafa Walid")
    
    # Main content
    st.title("🎭 Emotion Voice Analyzer")
    st.markdown("Analyze the emotional tone in speech audio using advanced AI")
    
    # Check for library availability
    if not LIBROSA_AVAILABLE:
        st.warning("""
        ⚠️ Audio processing libraries (librosa/PortAudio) are not fully available. 
        The app will run in demo mode with simulated features. 
        
        To fix this, install the required system dependencies:
        ```
        sudo apt-get update && sudo apt-get install -y libsndfile1 portaudio19-dev
        pip install librosa soundfile
        ```
        """)
    
    # Load model and assets
    model, scaler, encoder = load_assets()
    
    if model is None or scaler is None or encoder is None:
        st.error("Failed to load required models. Please check your installation and try again.")
        return
    
    # Create tabs for different input methods
    tabs = st.tabs(["📤 Upload Audio", "📊 History"])
    
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
    
    # Tab 2: History
    with tabs[1]:
        if show_history:
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            st.subheader("Analysis History")
            
            history_df = get_history_dataframe()
            if len(history_df) > 0:
                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                st.dataframe(history_df, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
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
                st.info("No analysis history yet. Upload audio to see results here.")
            st.markdown('</div>', unsafe_allow_html=True)

def process_audio_file(audio_path, model, scaler, encoder, show_waveform=True, show_spectrogram=True):
    """Process an audio file and display the analysis results."""
    try:
        # Load audio for visualization if librosa is available
        if LIBROSA_AVAILABLE:
            y, sr = librosa.load(audio_path)
        else:
            # Create dummy data for demonstration
            sr = 22050
            y = np.random.random(sr * 3) * 2 - 1  # 3 seconds of audio
        
        # Display audio player
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.subheader("Audio Sample")
        st.audio(audio_path)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create columns for visualizations
        if show_waveform or show_spectrogram:
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
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
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Prepare audio features for prediction
        with st.spinner("Extracting audio features..."):
            feat = prepare_example(audio_path, scaler)
        

        # Make prediction
        with st.spinner("Analyzing emotion..."):
            probs       = model.predict(feat, verbose=0)[0]     # 7-element vector
            pred_idx    = np.argmax(probs)                      # integer 0-6
            emotion_label = encoder.categories_[0][pred_idx]    # class name
            confidence  = probs[pred_idx]                       # probability

            
            # Add to history
            add_to_history(audio_path, emotion_label, confidence)
        
        # Display results
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
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
            
            # Custom bar chart with colors - dark theme
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1A1A1A')
            bars = ax.barh(emotion_df['Emotion'], emotion_df['Probability'], color=[get_emotion_color(e) for e in emotion_df['Emotion']])
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability', color='white')
            ax.set_title('Emotion Probability Distribution', color='white')
            ax.tick_params(colors='white')
            
            # Add percentage labels
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + 0.01
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1%}',
                        va='center', fontsize=10, color='white')
            
            plt.tight_layout()
            fig.patch.set_facecolor('#1A1A1A')
            
            st.pyplot(fig)
        
        # Display emotion description
        st.markdown(f"""
        <div style="background-color: {emotion_color}20; border-left: 5px solid {emotion_color}; padding: 15px; border-radius: 5px;">
            <h4>About {emotion_label.capitalize()} Emotion:</h4>
            <p>{get_emotion_description(emotion_label)}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Remove the temporary file
        try:
            os.unlink(audio_path)
        except:
            pass
        
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        st.info("Please try uploading a different audio file or recording again.")

if __name__ == "__main__":
    main()
