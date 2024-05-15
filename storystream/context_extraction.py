from storystream.audio.context_extraction import AudioContextExtractor
from storystream.image.context_extraction import ImageContextExtractor
from storystream.motion.context_extraction import MotionContextExtractor


class ContextExtractor:
    def __init__(self):
        self.audio_extractor = AudioContextExtractor()
        self.image_extractor = ImageContextExtractor()
        self.motion_extractor = MotionContextExtractor()

    def extract_audio_context(self, audio_file_path):
        return self.audio_extractor.extract_context(audio_file_path)

    def extract_image_context(self, image_file_path):
        return self.image_extractor.extract_context(image_file_path)

    def extract_motion_context(self, motion_file_path):
        return self.motion_extractor.extract_context(motion_file_path)

    def extract_all_contexts(
        self, audio_file_path=None, image_file_path=None, motion_file_path=None
    ):
        contexts = []

        if audio_file_path:
            contexts.append(self.extract_audio_context(audio_file_path))

        if image_file_path:
            contexts.append(self.extract_image_context(image_file_path))

        if motion_file_path:
            contexts.append(self.extract_motion_context(motion_file_path))

        return " ".join(contexts)


# Example usage
if __name__ == "__main__":
    extractor = ContextExtractor()
    audio_context = extractor.extract_audio_context("path/to/audio/file")
    image_context = extractor.extract_image_context("path/to/image/file")
    motion_context = extractor.extract_motion_context("path/to/motion/file")

    print("Audio Context:", audio_context)
    print("Image Context:", image_context)
    print("Motion Context:", motion_context)

    all_contexts = extractor.extract_all_contexts(
        audio_file_path="path/to/audio/file",
        image_file_path="path/to/image/file",
        motion_file_path="path/to/motion/file",
    )
    print("All Contexts Combined:", all_contexts)
