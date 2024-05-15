from storystream.audio.classification import AudioClassifier


class AudioContextExtractor:
    def __init__(self, model_path, device):
        self.classifier = AudioClassifier(model_path=model_path, device=device)

    def extract_context(self, audio_file_path):
        classified_info = self.classifier.classify(audio_file_path)
        # Convert classified info to text context
        context = f"Audio context: {classified_info}"
        return context
