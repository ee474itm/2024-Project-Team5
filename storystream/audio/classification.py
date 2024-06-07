import librosa
import numpy as np
import torch

from storystream.audio.model import M2EClassifier


class AudioClassifier:
    default_config = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    def __init__(self, model_path, **kwargs):
        """
        Initialize the AudioClassifier class.

        :param model_path: Path to the model weights file.
        :param kwargs: Additional configurations such as device.
        """
        config = self.default_config.copy()
        config.update(kwargs)
        self.device = config["device"]

        self.model = M2EClassifier().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.emotions = [
            "exciting/aggressive",
            "dramatic",
            "happy",
            "romantic",
            "sad",
            "angry",
        ]

    def extract_features(self, data, sample_rate):
        zcr = np.mean(
            librosa.feature.zero_crossing_rate(y=data, hop_length=20).T, axis=0
        )
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(
            librosa.feature.chroma_stft(
                S=stft, sr=sample_rate, n_fft=20, hop_length=20
            ).T,
            axis=0,
        )
        mfcc = np.mean(
            librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=20).T, axis=0
        )
        rms = np.mean(librosa.feature.rms(y=data, frame_length=100).T, axis=0)
        mel = np.mean(
            librosa.feature.melspectrogram(y=data, sr=sample_rate, hop_length=20).T,
            axis=0,
        )

        features = np.hstack((zcr, chroma_stft, mfcc, rms, mel))
        return features

    def feature_extractor(self, path):
        data, sample_rate = librosa.load(path)
        features = self.extract_features(data, sample_rate)
        return np.array(features)

    def classify(self, audio_file_path):
        feature = self.feature_extractor(audio_file_path)
        feature = np.array(feature).reshape((1, feature.shape[0]))
        feature_tensor = torch.from_numpy(feature[:, 1:]).to(self.device, torch.float32)
        _, predicted_idx = torch.max(self.model(feature_tensor), 1)
        return self.emotions[predicted_idx]
