import streamlit as st
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import os

# Page Configuration
st.set_page_config(
    page_title="EmotionSense",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
        }
        .title-text {
            font-size: 48px;
            font-weight: 800;
            color: #3b82f6;
            text-align: center;
            margin-bottom: 30px;
        }
        .section {
            padding: 20px;
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 0 10px rgba(0,0,0,0.05);
            margin-bottom: 30px;
        }
        .emotion-label {
            font-size: 32px;
            font-weight: 600;
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            display: inline-block;
        }
    </style>
""", unsafe_allow_html=True)

# Emotion mapping
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
EMOTION_EMOJIS = ['😐', '😌', '😄', '😔', '😠', '😨', '🤢', '😲']

# Feature extraction
def extract_features(data, sample_rate):
    result = []
    stft_magnitude = np.abs(librosa.stft(data))
    chroma = librosa.feature.chroma_stft(S=stft_magnitude, sr=sample_rate)
    result.extend(np.mean(chroma.T, axis=0))
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate)
    result.extend(np.mean(mfccs.T, axis=0))
    zcr = librosa.feature.zero_crossing_rate(y=data)
    result.extend(np.mean(zcr.T, axis=0))
    mel_spec = librosa.feature.melspectrogram(y=data, sr=sample_rate)
    result.extend(np.mean(mel_spec.T, axis=0))
    rms = librosa.feature.rms(y=data)
    result.extend(np.mean(rms.T, axis=0))
    return np.array(result)

# Model
class CNNLSTM(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 2)),
            torch.nn.Dropout(0.25),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 2)),
            torch.nn.Dropout(0.25),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 2)),
            torch.nn.Dropout(0.3),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 1)),
            torch.nn.Dropout(0.3)
        )
        self.lstm_input_size = 256 * 9
        self.lstm = torch.nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(x.size(0), x.size(1), -1)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return torch.nn.functional.log_softmax(out, dim=1)

# --- Streamlit UI ---
st.markdown('<div class="title-text">EmotionSense 🎧</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a WAV file to detect emotion", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    data, sr = librosa.load("temp.wav", duration=2.5, offset=0.6)
    features = extract_features(data, sr)

    if len(features) != 162:
        st.error("Error: Feature vector length mismatch.")
    else:
        device = torch.device('cpu')
        model = CNNLSTM(num_classes=8).to(device)
        model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
        model.eval()

        input_tensor = torch.tensor(features, dtype=torch.float32).view(1, 1, 9, 18)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()

        emotion = EMOTION_LABELS[pred]
        emoji = EMOTION_EMOJIS[pred]
        confidence = torch.softmax(output, dim=1)[0][pred].item() * 100

        color_map = {
            'neutral': '#64748b', 'calm': '#38bdf8', 'happy': '#facc15', 'sad': '#60a5fa',
            'angry': '#ef4444', 'fear': '#a855f7', 'disgust': '#10b981', 'surprise': '#ec4899'
        }

        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("Detected Emotion")
        st.markdown(f"<div class='emotion-label' style='background-color:{color_map[emotion]};'>{emoji} {emotion.capitalize()}</div>", unsafe_allow_html=True)
        st.progress(int(confidence), text=f"Confidence: {confidence:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("🔍 Feature Summary"):
            feature_types = ['Chroma', 'MFCC', 'ZCR', 'Mel', 'RMS']
            feature_counts = [12, 20, 1, 128, 1]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(feature_types, feature_counts, color="#3b82f6")
            ax.set_title("Feature Contributions")
            st.pyplot(fig)

else:
    st.info("Please upload a .wav audio file to begin analysis.")
