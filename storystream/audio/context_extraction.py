from storystream.audio.classification import AudioClassifier


class AudioContextExtractor:
    def __init__(self):
        self.classifier = AudioClassifier()

    def extract_context(self, audio_file_path):
        classified_info = self.classifier.classify(audio_file_path)
        # Convert classified info to text context
        context = f"Audio context: {classified_info}"
        return context
