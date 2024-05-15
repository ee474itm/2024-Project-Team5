from storystream.motion.classification import MotionClassifier


class MotionContextExtractor:
    def __init__(self):
        self.classifier = MotionClassifier()

    def extract_context(self, motion_file_path):
        classified_info = self.classifier.classify(motion_file_path)
        # Convert classified info to text context
        context = f"Motion context: {classified_info}"
        return context
