from storystream.audio.model import M2EClassifier
import torch
import numpy as np
import librosa

class AudioClassifier:
    def __init__(self, model_path, device):
        self.model = M2EClassifier()
        self.device = device
        self.emotions = ['romantic', 'aggressive', 'sad', 'happy', 'dramatic']
        
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.to(self.device)        
        pass
    
    def extract_features(self, data, sample_rate):
        result = np.array([])
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data, hop_length=20).T, axis=0)
        result = np.hstack((result, zcr))
        
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate, n_fft=20, hop_length=20).T, axis=0)
        result = np.hstack((result, chroma_stft))
        
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=20).T, axis=0)
        result = np.hstack((result, mfcc))
        
        rms = np.mean(librosa.feature.rms(y=data, frame_length=100).T, axis=0)
        result = np.hstack((result, rms))
        
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate, hop_length=20).T, axis=0)
        result = np.hstack((result, mel))
        
        return result
    
    def feature_extractor(self, path):
        data, sample_rate = librosa.load(path)
        res1 = self.extract_features(data, sample_rate)
        result = np.array(res1)
        return result

    def classify(self, audio_file_path):
        feature = self.feature_extractor(audio_file_path)
        feature = torch.tensor(feature, dtype=torch.float32).to(self.device)
        _, predicted_idx = torch.max(self.model(feature), 1)
        return self.emotions[predicted_idx]
