import streamlit as st
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import os 
# Streamlit Page Config
st.set_page_config(
    page_title="EmotionWave - Speech Emotion Detector",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom CSS Styling----------------------------------------------------------------------------------------------------------------------------------------------------------------
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
            background-color: #0f172a;
            color: #f1f5f9;
        }
        .main-title {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #38bdf8;
            padding-top: 1rem;
        }
        .upload-section {
            border: 2px dashed #38bdf8;
            border-radius: 15px;
            padding: 2rem;
            background-color: #1e293b;
            text-align: center;
        }
        .result-box {
            padding: 1.5rem;
            border-radius: 15px;
            background-color: #1e293b;
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
        }
        .emotion-badge {
            font-size: 2rem;
            font-weight: bold;
            padding: 0.75rem 2rem;
            border-radius: 40px;
            display: inline-block;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)


# --- Model Definition ---------------------------------------------------------------------------------------
class CNNLSTM(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()

        self.conv = torch.nn.Sequential(
            # Conv Block 1
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),  
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 2)),
            torch.nn.Dropout(0.3),

            # Conv Block 2
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 2)),
            torch.nn.Dropout(0.3),

            # Conv Block 3
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 2)),
            torch.nn.Dropout(0.4),

            # Conv Block 4
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 1)),
            torch.nn.Dropout(0.4)
        )

        self.lstm_input_size = 256 * 9  # From conv output shape
        self.lstm_hidden = 128
        self.lstm = torch.nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.lstm_hidden * 2, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)                                   # [B, 256, 9, W]
        x = x.permute(0, 3, 1, 2)                          # [B, W, 256, 9]
        x = x.contiguous().view(x.size(0), x.size(1), -1)  # [B, W, 256*9]
        lstm_out, _ = self.lstm(x)                         # [B, W, 256]
        out = lstm_out[:, -1, :]                           
        out = self.classifier(out)
        return torch.nn.functional.log_softmax(out, dim=1)





# Emotion mapping
EMOTION_LABELS = ['calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
EMOTION_EMOJIS = ['😌', '😄', '😔', '😠', '😨', '🤢', '😲']

# --- Feature Extraction -------------------------------------------------------------------------------------------------------------------------------------------------------------------
#defining extract_features to extract various features
def extract_features(audio, sample_rate):
    """
    Features extracted:
    - MFCCs
    - Zero Crossing Rate
    - Mel-spectrogram
    - Chroma STFT
    - Root Mean Square Energy
    """
    features = []

    # STFT magnitude
    stft_mag = np.abs(librosa.stft(audio))

    # MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate)
    features.extend(np.mean(mfcc.T, axis=0))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    features.extend(np.mean(zcr.T, axis=0))

    # Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    features.extend(np.mean(mel_spec.T, axis=0))

    # Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(S=stft_mag, sr=sample_rate)
    features.extend(np.mean(chroma_stft.T, axis=0))

    # Root Mean Square Energy
    rms_energy = librosa.feature.rms(y=audio)
    features.extend(np.mean(rms_energy.T, axis=0))

    return np.array(features)


# --- Streamlit App -------------------------------------------------------------------------------------------------------------------------------------------------------------------
def main():
    # App header
    st.markdown('<h1 class="title">🎤 Speech Emotion Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar for additional info
    with st.sidebar:
        st.header("About")
        st.markdown("""
            Welcome to EmotionWave – an advanced speech emotion recognition app powered by deep learning.

            🎧 Upload your WAV audio file  
            🧠 Experience real-time emotion detection  
            📊 Explore visual insights from audio features  

            Uncover the emotion behind every voice.
            """)
        st.divider()
        st.subheader("Model Information")
        st.caption("**Architecture:** HYBRID CNN-LSTM")
        st.caption("**Accuracy:** 74.61% (Validation Set)")
        st.divider()
        st.markdown("[GitHub Repository](https://github.com/Alien0nEarth/Emotional-classification-speech-processing.git)")
    
    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        # Upload section with custom styling
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file:
            # Audio player
            st.audio(uploaded_file)
            
            # Visualization section
            with st.spinner('Analyzing audio...'):
                # Save temporary file
                with open("temp.wav", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load audio
                data, sr = librosa.load("temp.wav", duration=2.5, offset=0.6)
                
                # Create waveform visualization
                fig, ax = plt.subplots(figsize=(10, 3))
                librosa.display.waveshow(data, sr=sr, ax=ax)
                ax.set_title("Audio Waveform", fontsize=14)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                st.pyplot(fig, use_container_width=True)  # Added use_container_width
    
    with col2:
        if uploaded_file:
            # Display analysis results
            with st.spinner('Processing emotion...'):
                # Feature extraction
                features = extract_features(data, sr)
                
                # Validate feature length
                if len(features) != 162:
                    st.error(f"Feature extraction error: Expected 162 features, got {len(features)}")
                else:
                    # Load model
                    device = torch.device('cpu')
                    model = CNNLSTM(num_classes=7).to(device)
                    model_path = os.path.join('models','best_model.pth')
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.eval()
                    
                    # Prepare input tensor
                    features_tensor = torch.tensor(features, dtype=torch.float32).view(1, 1, 9, 18)
                    
                    # Predict
                    with torch.no_grad():
                        output = model(features_tensor)
                        pred = torch.argmax(output, dim=1).item()
                    
                    # Add artificial delay for better UX
                    time.sleep(1.5)
                    
                # Display results with animation
                emotion = EMOTION_LABELS[pred]
                emoji = EMOTION_EMOJIS[pred]
                
                # Color coding for emotions
                emotion_colors = {
                    'calm': '#60a5fa',
                    'happy': '#fbbf24',
                    'sad': '#60a5fa',
                    'angry': '#f87171',
                    'fear': '#c084fc',
                    'disgust': '#34d399',
                    'surprise': '#f472b6'
                }
                
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.subheader("Analysis Results")
                
                # Emotion badge with pulsing animation
                st.markdown(
                    f'<div class="emotion-badge pulsing" style="background-color: {emotion_colors[emotion]}; color: white;">'
                    f'{emoji} {emotion.capitalize()}'
                    '</div>',
                    unsafe_allow_html=True
                )
                
                # Confidence meter
                confidence = torch.softmax(output, dim=1)[0][pred].item() * 100
                st.progress(int(confidence), text=f"Confidence: {confidence:.1f}%")
                
                # Feature visualization
                st.divider()
                st.subheader("Audio Features")
                
                # Create feature importance chart
                feature_types = ['Chroma', 'MFCC', 'ZCR', 'Mel', 'RMS']
                feature_lengths = [12, 20, 1, 128, 1]  # Adjust to match your feature extractor
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(feature_types, feature_lengths, color='#4f46e5')
                ax.set_title("Feature Composition", fontsize=14)
                st.pyplot(fig)
                
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()