# 🎭 Emotion Voice Analyzer

Analyze the emotional tone of short speech snippets right in your browser using a lightweight CNN model, Streamlit UI, and rich dark-themed visualizations  

---

## Table of Contents
1. [Live Demo](#live-demo)  
2. [Features](#features)  
3. [Quick Start](#quick-start)  
4. [Installation](#installation)  
5. [Folder Structure](#folder-structure)  
6. [How It Works](#how-it-works)  
7. [Customization](#customization)  
8. [Troubleshooting](#troubleshooting)  
9. [Contributing](#contributing)  
10. [License](#license)  

---

## Live Demo
Spin up the app locally with one command and open `http://localhost:8501` to try it out  
*(A public demo link will appear here once deployed)*  

---

## Features
- **Seven-class emotion recognition**: angry, disgust, fear, happy, neutral, peaceful / calm, sad  
- **Auto-downloaded assets** from Google Drive (model, scaler, label encoder) so no manual setup is needed  
- **Dark UI theme** with custom CSS for a modern look  
- **Waveform & Mel-spectrogram visualizations** rendered on demand  
- **History tab** storing every analysis in session memory with CSV export  
- **Graceful fallback mode** if PortAudio or librosa are missing, letting the interface remain usable  
- **Single-file deployment** — all logic lives in `app.py` for quick prototyping  

---

## Quick Start
```bash
# 1 Clone
git clone https://github.com/your-user/emotion-voice-analyzer.git
cd emotion-voice-analyzer

# 2 Create virtual env (optional)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3 Install requirements
pip install -r requirements.txt

# 4 Run
streamlit run app.py
Open the printed URL in your browser, upload a 2–5 s speech clip (WAV, MP3, FLAC) and view the results immediately

Installation
Step	Command	Notes
System deps	sudo apt update && sudo apt install -y libsndfile1 portaudio19-dev	Enables full audio pipeline
Python deps	pip install -r requirements.txt	TensorFlow 2.15+, Streamlit 1.33+, librosa 0.10+
Launch app	streamlit run app.py	Starts local dev server on port 8501

The first run downloads three files (~12 MB) from Google Drive and caches them in the project root

Folder Structure
text
Copy
Edit
emotion-voice-analyzer/
│
├─ app.py                 # main Streamlit application
├─ requirements.txt       # pip dependencies
├─ README.md
└─ assets/                # created at runtime (model + pickles)
How It Works
Audio Load — librosa loads 2.5 s of speech starting at a configurable offset

Feature Extraction — zero-crossing rate, RMS, 40 MFCCs flattened into a vector of length 2376

Pre-processing — vector scaled with a sklearn StandardScaler fitted during training

Inference — TensorFlow CNN predicts a 7-element probability distribution

Post-processing — top class converted to emoji, color, description, and plotted with Matplotlib

Session History — results appended to an in-memory pandas DataFrame and optionally downloaded as CSV

Customization
What you want	Where to tweak
Replace model	Update the three *_FILE_ID constants with new Google Drive file IDs
Add emotions	Retrain the model, extend emoji_map, color_map, and descriptions dictionaries
UI palette	Edit apply_custom_css() in app.py
Default offset	Change the audio_offset slider’s default value

Troubleshooting
PortAudio / librosa errors — make sure libsndfile1 and portaudio19-dev are installed and Python wheels match your OS architecture

Black screen on first launch — wait for Google Drive downloads to finish, then refresh the page

Memory errors on ARM boards — install TensorFlow Lite or use the fallback random-feature mode for demos only

Contributing
Pull requests are welcome

Fork the repo

Create a feature branch

Commit changes with descriptive messages

Open a PR against main

License
This project is released under the MIT License
