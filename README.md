# 🎭 Emotion Voice Analyzer

A sophisticated web application that analyzes speech audio to detect emotional states using deep learning. Built with Streamlit and TensorFlow, featuring a modern dark-themed interface with real-time audio visualization and comprehensive emotion analysis.

## ✨ Features

- **🎯 Emotion Detection**: Identifies 7 different emotions from speech audio
- **🎵 Audio Visualization**: Real-time waveform and mel-spectrogram displays
- **🌐 Web Interface**: Modern, responsive Streamlit web application
- **📊 Interactive Charts**: Emotion probability distributions with custom styling
- **📈 Analysis History**: Track and export previous analysis results
- **🎨 Dark Theme**: Beautiful dark mode interface with custom CSS
- **🔄 Auto Model Loading**: Automatic download of pre-trained models from Google Drive
- **📱 Responsive Design**: Works on desktop and mobile devices

## 🎭 Supported Emotions

| Emotion | Emoji | Description |
|---------|-------|-------------|
| **Angry** | 😠 | Tension and hostility from frustration |
| **Disgust** | 🤢 | Revulsion to something unpleasant |
| **Fear** | 😨 | Distressing emotion from perceived threat |
| **Happy** | 😄 | Joy, satisfaction, and fulfillment |
| **Neutral** | 😐 | Calm, balanced emotional state |
| **Peaceful** | 😌 | Tranquility and relaxed state |
| **Sad** | 😢 | Emotional pain and sorrow |

## 🔧 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM recommended
- Internet connection (for initial model download)
- Audio file support (WAV, MP3, FLAC)

### Audio Requirements
- **Duration**: 2-5 seconds recommended
- **Format**: WAV, MP3, or FLAC
- **Content**: Clear speech audio
- **Quality**: 16kHz+ sample rate preferred

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/emotion-voice-analyzer.git
cd emotion-voice-analyzer
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install System Dependencies

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y libsndfile1 portaudio19-dev
sudo apt-get install -y ffmpeg
```

#### macOS:
```bash
brew install portaudio
brew install ffmpeg
```

#### Windows:
```bash
# Install Microsoft C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

## 📋 Dependencies

```txt
streamlit==1.34.0
tensorflow==2.16.1
numpy
pandas
librosa
scikit-learn
matplotlib
gdown
pytz
```

### Optional Dependencies
```bash
# For better audio processing
pip install soundfile
pip install pydub

# For GPU acceleration (optional)
pip install tensorflow-gpu  # If you have CUDA-compatible GPU
```

## 🚀 Usage

### Running the Application
```bash
streamlit run app.py
```

The application will automatically:
1. Download pre-trained models on first run
2. Open your web browser to `http://localhost:8501`
3. Display the emotion analyzer interface

### Using the Analyzer

#### 1. Upload Audio File
- Click "Choose an audio file" 
- Select WAV, MP3, or FLAC file
- Recommended: 2-5 seconds of clear speech

#### 2. Analyze Emotion
- Click "Analyze Uploaded Audio"
- View real-time processing progress
- See emotion detection results

#### 3. Explore Results
- **Emotion Detection**: Primary emotion with confidence score
- **Probability Distribution**: Chart showing all emotion probabilities
- **Audio Visualizations**: Waveform and spectrogram displays
- **Emotion Description**: Detailed explanation of detected emotion

#### 4. Review History
- Access the "History" tab
- View all previous analyses
- Export results as CSV

## ⚙️ Configuration

### Model Configuration
The application uses pre-trained models stored on Google Drive:
```python
MODEL_FILE_ID   = "1rNjobTEVv7iylFSPinYdohpZS-rs5YRJ"  # CNN model
SCALER_FILE_ID  = "1aEN09Avk2G9_Oso7Nl3kLit4cCFphtMo"  # Feature scaler
ENCODER_FILE_ID = "1INMy6xujSKvW__RAeVUEJRjTqvPdzeQ8"  # Label encoder
```

### Audio Processing Settings
```python
TARGET_FEATURE_LEN = 2376    # Feature vector length
AUDIO_DURATION = 2.5         # Analysis duration (seconds)
AUDIO_OFFSET = 0.6          # Start offset (seconds)
```

### Customizing the Interface
Modify the CSS in `apply_custom_css()` function to change:
- Color scheme
- Layout spacing
- Font styles
- Component styling

## 📁 Project Structure

```
emotion-voice-analyzer/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore file
├── models/               # Downloaded models (auto-created)
│   ├── CNN_full_model.h5
│   ├── scaler2.pickle
│   └── encoder2.pickle
├── assets/               # Static assets
│   └── demo_audio/       # Sample audio files
├── docs/                 # Additional documentation
│   ├── model_info.md
│   ├── audio_guide.md
│   └── deployment.md
└── tests/                # Test files
    └── test_audio_processing.py
```

## 📊 Datasets & Training Data

### Datasets Used
The model was trained on four major emotion speech datasets:

#### 1. **RAVDESS** (Ryerson Audio-Visual Database)
- **Size**: 24 actors, 1,440 audio files
- **Emotions**: 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
- **Quality**: Professional actors, controlled recording environment
- **Format**: WAV files, 48kHz sample rate

#### 2. **CREMA-D** (Crowdsourced Emotional Multimodal Actors)
- **Size**: 91 actors, 7,442 audio clips
- **Emotions**: 6 emotions (sad, angry, disgust, fear, happy, neutral)
- **Diversity**: Multi-ethnic actors, age range 20-74
- **Format**: WAV files, various durations

#### 3. **TESS** (Toronto Emotional Speech Set)
- **Size**: 2 actresses, 2,800 audio files
- **Emotions**: 7 emotions including "ps" (peaceful/surprise)
- **Content**: 200 target words spoken in different emotions
- **Quality**: High-quality studio recordings

#### 4. **SAVEE** (Surrey Audio-Visual Expressed Emotion)
- **Size**: 4 male speakers, 480 audio files
- **Emotions**: 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise)
- **Content**: Phonetically balanced sentences
- **Application**: Research-grade dataset

### Combined Dataset Statistics
- **Total Audio Files**: ~12,000+ original files
- **After Augmentation**: ~48,000+ training samples
- **Emotion Distribution**: Balanced across all classes
- **Duration**: 2.5 seconds analysis window with 0.6s offset

## 🔄 Data Preprocessing & Augmentation

### Audio Preprocessing Pipeline

#### 1. **Audio Loading**
```python
# Load audio with librosa
data, sr = librosa.load(path, duration=2.5, offset=0.6)
```

#### 2. **Data Augmentation Techniques**
To increase dataset diversity and model robustness:

**Noise Addition**
```python
def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])
```

**Time Stretching**
```python
def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)
```

**Time Shifting**
```python
def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)
```

**Pitch Shifting**
```python
def pitch(data, sr, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=pitch_factor)
```

#### 3. **Feature Extraction Process**
For each audio sample, multiple features are extracted:

**Zero Crossing Rate (ZCR)**
- Measures rate of sign changes in the signal
- Indicates voiced vs unvoiced speech segments
- Helps distinguish between different emotions

**Root Mean Square Energy (RMS)**
- Measures the power/energy of the audio signal
- Correlates with loudness and emotional intensity
- Important for detecting angry vs calm emotions

**Mel-Frequency Cepstral Coefficients (MFCCs)**
- 40 MFCC coefficients extracted
- Captures spectral characteristics of speech
- Most important features for emotion recognition

```python
def extract_features(data, sr=22050):
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=data)
    
    # RMS energy
    rms = librosa.feature.rms(y=data)
    
    # MFCCs (40 coefficients)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    
    # Combine all features
    return np.hstack([zcr, rms, mfcc.T.flatten()])
```

#### 4. **Feature Vector Specifications**
- **Total Length**: 2,376 dimensions
- **Composition**: ZCR + RMS + MFCCs (flattened)
- **Normalization**: StandardScaler applied
- **Format**: Reshaped to (samples, 2376, 1) for CNN input

## 🧠 Model Architecture & Training

### CNN Model Architecture
```python
Sequential([
    Conv1D(512, kernel_size=5, activation='relu', input_shape=(2376, 1)),
    BatchNormalization(),
    MaxPool1D(pool_size=5, strides=2),
    
    Conv1D(512, kernel_size=5, activation='relu'),
    BatchNormalization(), 
    MaxPool1D(pool_size=5, strides=2),
    Dropout(0.2),
    
    Conv1D(256, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPool1D(pool_size=5, strides=2),
    
    Conv1D(256, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPool1D(pool_size=5, strides=2),
    Dropout(0.2),
    
    Conv1D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPool1D(pool_size=3, strides=2),
    Dropout(0.2),
    
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dense(7, activation='softmax')  # 7 emotion classes
])
```

### Training Details
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Epochs**: 20 (with early stopping)
- **Batch Size**: 64
- **Train/Validation Split**: 80/20
- **Callbacks**: Early Stopping, Learning Rate Reduction, Model Checkpoint
- **Final Accuracy**: ~85% on test data

### Model Files
- `CNN_full_model.h5`: Complete trained model (TensorFlow/Keras format)
- `scaler2.pickle`: StandardScaler for feature normalization
- `encoder2.pickle`: OneHotEncoder for emotion label encoding

## 🐛 Troubleshooting

### Common Issues

#### Audio Loading Errors
```bash
# Error: "librosa or PortAudio not available"
sudo apt-get install -y libsndfile1 portaudio19-dev
pip install librosa soundfile
```

#### Model Download Issues
```bash
# Check internet connection
ping google.com

# Clear cache and restart
rm -rf models/
streamlit run app.py
```

#### Memory Issues
```bash
# Reduce TensorFlow memory usage
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Monitor memory usage
htop
```

#### Streamlit Port Issues
```bash
# Use different port
streamlit run app.py --server.port 8502

# Kill existing processes
pkill -f streamlit
```

### Error Messages

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: librosa` | Install audio dependencies |
| `Failed to load required models` | Check internet connection, restart app |
| `Error processing audio` | Verify audio file format and quality |
| `Streamlit port in use` | Use different port or kill existing process |

## 📊 Performance Optimization

### For Better Performance
```python
# Reduce model precision (optional)
model = tf.keras.models.load_model(MODEL_PATH)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Enable GPU acceleration
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

### Audio Processing Tips
- Use WAV format for best quality
- Ensure clear speech without background noise
- Optimal length: 2-5 seconds
- Sample rate: 16kHz or higher

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py --server.runOnSave true
```

### Production Deployment

#### Streamlit Cloud
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy automatically

#### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

## 🔄 Reproducing Training Results

### Prerequisites for Training
```bash
# Required datasets (download from official sources)
# 1. RAVDESS: https://zenodo.org/record/1188976
# 2. CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D
# 3. TESS: https://tspace.library.utoronto.ca/handle/1807/24487  
# 4. SAVEE: http://kahlan.eps.surrey.ac.uk/savee/

# Install training dependencies
pip install librosa tensorflow scikit-learn pandas numpy matplotlib seaborn
pip install IPython tqdm
```

### Training Reproduction Steps

#### 1. **Dataset Preparation**
```python
# Organize datasets in this structure:
datasets/
├── ravdess/audio_speech_actors_01-24/
├── crema/AudioWAV/
├── tess/TESS Toronto emotional speech set data/
└── savee/ALL/
```

#### 2. **Feature Extraction** 
```python
# Run feature extraction (takes ~30-60 minutes)
python Speech-Emotion-Recognition-using-CNN.py

# This generates:
# - emotion.csv (extracted features)
# - Data preprocessing and augmentation
# - Feature scaling and encoding
```

#### 3. **Model Training**
```python
# Train the CNN model (takes ~2-4 hours on GPU)
# Automatic callbacks:
# - Early stopping (patience=5)
# - Learning rate reduction  
# - Model checkpointing (saves best weights)

# Outputs:
# - CNN_full_model.h5 (complete model)
# - best_model1_weights.h5 (best weights only)
# - scaler2.pickle (feature scaler)
# - encoder2.pickle (label encoder)
```

#### 4. **Model Evaluation**
```python
# Automatic evaluation includes:
# - Accuracy metrics on test set
# - Confusion matrix visualization
# - Classification report
# - Training/validation loss curves
```

### Improving the Model

#### Data Enhancements
- **More Datasets**: Add IEMOCAP, Berlin EMO-DB, or custom datasets
- **Better Augmentation**: Experiment with speed changes, formant shifting
- **Longer Audio**: Increase analysis window beyond 2.5 seconds
- **Quality Filtering**: Remove low-quality or mislabeled samples

#### Architecture Improvements  
```python
# Try different architectures:
# 1. Deeper CNN with more layers
# 2. CNN + LSTM hybrid for temporal modeling
# 3. Attention mechanisms for feature importance
# 4. ResNet-style skip connections
# 5. Multi-scale convolutions
```

#### Advanced Features
```python
# Additional audio features to experiment with:
# - Chroma features
# - Spectral contrast  
# - Tonnetz (harmonic network)
# - Spectral centroid, rolloff, bandwidth
# - Delta and delta-delta MFCCs
```

#### Hyperparameter Tuning
```python
# Key parameters to optimize:
learning_rates = [0.001, 0.0005, 0.0001]
batch_sizes = [32, 64, 128]  
dropout_rates = [0.2, 0.3, 0.5]
conv_filters = [256, 512, 1024]
kernel_sizes = [3, 5, 7]
```

### Transfer Learning Opportunities
```python
# Use pre-trained models as feature extractors:
# 1. Wav2Vec 2.0 for speech representations
# 2. VGGish for audio features  
# 3. OpenL3 for general audio embeddings
# 4. YAMNet for audio classification features
```

#### Heroku Deployment
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

## 🔬 Model Development & Research

### Training Environment
- **Platform**: Kaggle Notebooks with GPU acceleration
- **Framework**: TensorFlow 2.16.1, Keras
- **Audio Processing**: librosa, soundfile
- **Data Science**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn

### Research Methodology
1. **Dataset Integration**: Combined 4 major emotion datasets
2. **Exploratory Analysis**: Audio visualization and statistical analysis
3. **Feature Engineering**: Comprehensive audio feature extraction
4. **Model Experimentation**: Tested various CNN architectures
5. **Hyperparameter Tuning**: Optimized learning rate, batch size, layers
6. **Validation**: Cross-dataset testing and confusion matrix analysis

### Model Serialization & Deployment
```python
# Save complete model for deployment
model.save('CNN_full_model.h5')

# Save preprocessing components
import pickle
with open('scaler2.pickle', 'wb') as f:
    pickle.dump(scaler, f)
    
with open('encoder2.pickle', 'wb') as f:
    pickle.dump(encoder, f)
```

### Prediction Pipeline
```python
def prediction_pipeline(audio_path):
    # 1. Load audio (2.5s, offset 0.6s)
    data, sr = librosa.load(audio_path, duration=2.5, offset=0.6)
    
    # 2. Extract features (ZCR + RMS + MFCCs)
    features = extract_features(data, sr)
    
    # 3. Normalize features
    features = scaler.transform(features.reshape(1, -1))
    
    # 4. Reshape for CNN input
    features = np.expand_dims(features, axis=2)
    
    # 5. Model prediction
    probabilities = model.predict(features)
    
    # 6. Decode prediction
    emotion = encoder.inverse_transform(probabilities)
    
    return emotion, probabilities
```

## 🔒 Privacy & Security

- **Local Processing**: All audio analysis happens locally
- **No Data Storage**: Audio files are processed and deleted immediately
- **Model Security**: Pre-trained models downloaded securely
- **Session Privacy**: Analysis history stored only in browser session

## 🎨 Customization

### Adding New Emotions
1. Retrain model with additional emotion classes
2. Update `encoder2.pickle` with new labels
3. Add emoji and color mappings in `get_emotion_emoji()` and `get_emotion_color()`
4. Update emotion descriptions in `get_emotion_description()`

### Custom Themes
Modify the CSS in `apply_custom_css()`:
```python
# Light theme example
.main {
    background-color: #FFFFFF;
    color: #000000;
}
```

### Additional Features
- Real-time microphone input
- Batch processing of multiple files
- Export detailed analysis reports
- Integration with external APIs

## 🎯 Research Contributions & Impact

### Novel Contributions
1. **Multi-Dataset Integration**: Successfully combined 4 major emotion datasets for improved generalization
2. **Comprehensive Augmentation**: 4x data augmentation strategy with noise, pitch, and temporal variations  
3. **Optimized Feature Engineering**: 2,376-dimensional feature vector combining temporal and spectral features
4. **Production-Ready Pipeline**: Complete end-to-end system from training to web deployment
5. **Cross-Cultural Validation**: Model trained on diverse speakers and recording conditions

### Research Applications
- **Psychology Research**: Analyze emotional patterns in speech therapy
- **Human-Computer Interaction**: Emotion-aware voice assistants
- **Mental Health**: Early detection of depression or anxiety through speech
- **Education Technology**: Adaptive learning systems based on student emotional state
- **Customer Service**: Automated emotion detection in call centers
- **Entertainment**: Emotion-responsive gaming and interactive media

### Academic Impact
This implementation demonstrates:
- **Reproducible Research**: Complete code and methodology available
- **Dataset Standardization**: Unified preprocessing across multiple databases
- **Performance Benchmarking**: Clear metrics for comparison with other models
- **Deployment Feasibility**: Practical application beyond academic settings

### Future Research Directions
1. **Real-Time Processing**: Optimize for live audio stream analysis
2. **Multi-Modal Fusion**: Combine audio with facial expressions and text
3. **Personalization**: Adapt models to individual speaker characteristics
4. **Emotion Intensity**: Predict not just emotion type but intensity levels
5. **Cultural Adaptation**: Train specialized models for different languages/cultures

## 🤝 Contributing

We welcome contributions! Here's how to help:

### Development Setup
```bash
# Fork the repository
git clone https://github.com/yourusername/emotion-voice-analyzer.git
cd emotion-voice-analyzer

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black app.py
flake8 app.py
```

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Areas for Contribution
- Additional emotion categories
- Improved audio preprocessing
- Better visualization options
- Mobile responsiveness
- Performance optimizations
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow Team** for the deep learning framework
- **Streamlit Team** for the amazing web app framework
- **Librosa Developers** for audio processing capabilities
- **Research Community** for emotion recognition datasets and methodologies

## 👥 Project Contributors

- **Abdelrahman Elsayyad** - Lead Developer
- **Mostafa Walid** - Data Scientist & Model Development

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/emotion-voice-analyzer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/emotion-voice-analyzer/discussions)
- **Email**: support@emotionanalyzer.com

## 📚 Resources

- [Emotion Recognition Research Papers](docs/research.md)
- [Audio Processing Guide](docs/audio_guide.md)
- [Model Training Documentation](docs/model_training.md)
- [API Documentation](docs/api.md)

---

**Made with ❤️ for emotion AI research and applications**
