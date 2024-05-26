from storystream.audio.classification import AudioClassifier


class AudioContextExtractor:
    def __init__(self, classifier_params=None):
        self.classifier = AudioClassifier(**(classifier_params or {}))

    def extract_context(self, audio_file_path, **kwargs):
        classified_info = self.classifier.classify(audio_file_path, **kwargs)
        # Convert classified info to text context
        context = f"Audio context: {classified_info}"
        return context
