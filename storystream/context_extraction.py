import os

from storystream.audio.context_extraction import AudioContextExtractor
from storystream.image.context_extraction import ImageContextExtractor
from storystream.utils import is_url


class ContextExtractor:
    def __init__(self):
        self.audio_extractor = AudioContextExtractor()
        self.image_extractor = ImageContextExtractor()

    def extract_audio_context(self, audio_file_path):
        return self.audio_extractor.extract_context(audio_file_path)

    def extract_image_context(self, image_path):
        if is_url(image_path):
            return self.image_extractor.extract_context_from_url(image_path)
        elif os.path.isfile(image_path):
            return self.image_extractor.extract_context_from_file(image_path)
        else:
            raise ValueError(f"Invalid image path: {image_path}")

    def extract_all_contexts(self, audio_file_path=None, image_file_path=None):
        contexts = []

        if audio_file_path:
            contexts.append(self.extract_audio_context(audio_file_path))

        if image_file_path:
            contexts.append(self.extract_image_context(image_file_path))

        return " ".join(contexts)


# Example usage
if __name__ == "__main__":
    extractor = ContextExtractor()
    audio_context = extractor.extract_audio_context("path/to/audio/file")
    image_context = extractor.extract_image_context("path/to/image/file")

    print("Audio Context:", audio_context)
    print("Image Context:", image_context)

    all_contexts = extractor.extract_all_contexts(
        audio_file_path="path/to/audio/file",
        image_file_path="path/to/image/file",
    )
    print("All Contexts Combined:", all_contexts)
