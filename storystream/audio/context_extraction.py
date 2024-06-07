from storystream.audio.classification import AudioClassifier


class AudioContextExtractor:
    def __init__(self, classifier_params=None):
        self.classifier = AudioClassifier(**(classifier_params or {}))

    def extract_context(self, audio_file_path, **kwargs):
        return self.classifier.classify(audio_file_path, **kwargs)
