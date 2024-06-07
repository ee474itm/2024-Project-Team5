import os

from storystream.audio.context_extraction import AudioContextExtractor
from storystream.image.context_extraction import ImageContextExtractor
from storystream.utils import is_url


class ContextExtractor:
    def __init__(self, audio_params=None, image_params=None):
        """
        Initialize the ContextExtractor class.

        :param audio_params: Dictionary of parameters to pass to AudioContextExtractor.
        :param image_params: Dictionary of parameters to pass to ImageContextExtractor.
        """
        self.audio_extractor = AudioContextExtractor(**(audio_params or {}))
        self.image_extractor = ImageContextExtractor(**(image_params or {}))

    def extract_audio_context(self, audio_file_path, **kwargs):
        return self.audio_extractor.extract_context(audio_file_path, **kwargs)

    def extract_image_context(self, image_path, **kwargs):
        if is_url(image_path):
            return self.image_extractor.extract_context_from_url(image_path, **kwargs)
        elif os.path.isfile(image_path):
            return self.image_extractor.extract_context_from_file(image_path, **kwargs)
        else:
            raise ValueError(f"Invalid image path: {image_path}")

    def extract_all_contexts(
        self,
        user_input: str = None,
        audio_file_path=None,
        image_file_path=None,
        **kwargs,
    ):
        contexts = {}
        if user_input:
            contexts["text"] = user_input
        if audio_file_path:
            contexts["audio"] = self.extract_audio_context(audio_file_path, **kwargs)
        if image_file_path:
            contexts["image"] = self.extract_image_context(image_file_path, **kwargs)
        return contexts


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
