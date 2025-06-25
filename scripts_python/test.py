import torch
import numpy as np
import librosa
from pathlib import Path

# Updated Emotion mappings after dropping 'disgust'
EMOTION_TO_INDEX = {
    'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3,
    'angry': 4, 'fear': 5, 'surprise': 6
}
INDEX_TO_EMOTION = {v: k for k, v in EMOTION_TO_INDEX.items()}

# CNN-LSTM Model Definition
class EmotionClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(EmotionClassifier, self).__init__()
        self.cnn_layers = torch.nn.Sequential(
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

        self.classifier = torch.nn.Sequential(
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
        x = self.cnn_layers(x)                            # Shape: [B, 256, 9, 3]
        x = x.permute(0, 3, 1, 2)                         # Shape: [B, 3, 256, 9]
        x = x.contiguous().view(x.size(0), x.size(1), -1) # Shape: [B, 3, 256*9]
        lstm_out, _ = self.lstm(x)                        # Shape: [B, 3, 256]
        x = lstm_out[:, -1, :]                            # Take last output
        x = self.classifier(x)                            # Fully connected layers
        return torch.nn.functional.log_softmax(x, dim=1)

# Feature Extraction
def extract_audio_features(file_path):
    waveform, sr = librosa.load(file_path, duration=2.5, offset=0.6)

    stft_mag = np.abs(librosa.stft(waveform))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft_mag, sr=sr).T, axis=0)
    mfcc = np.mean(librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=20).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=waveform))
    mel = np.mean(librosa.feature.melspectrogram(y=waveform, sr=sr).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=waveform))

    feature_vector = np.hstack([chroma, mfcc, zcr, mel, rms])
    return feature_vector

# Prediction Function
def predict_emotion_from_audio(audio_file, model_checkpoint='models/best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EmotionClassifier(num_classes=7).to(device)  # updated to 7
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.eval()

    features = extract_audio_features(audio_file)
    input_tensor = torch.tensor(features.reshape(1, 1, 9, 18), dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_index = torch.argmax(output, dim=1).item()

    return INDEX_TO_EMOTION[predicted_index]

# Entry Point
if __name__ == "__main__":
    input_path = input("Enter the path to the audio file: ").strip()
    model_file = "models/best_model.pth"

    try:
        path_obj = Path(input_path)
        if not path_obj.exists() or not path_obj.is_file():
            raise FileNotFoundError(f"File not found: {input_path}")

        emotion_result = predict_emotion_from_audio(input_path, model_file)
        print(f"\nPredicted Emotion: **{emotion_result.upper()}**")

    except FileNotFoundError as fnf_err:
        print(fnf_err)
    except Exception as err:
        print(f"An error occurred during prediction: {err}")
