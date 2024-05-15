from storystream.image.classification import ImageClassifier


class ImageContextExtractor:
    def __init__(self):
        self.classifier = ImageClassifier()

    def extract_context(self, image_file_path):
        classified_info = self.classifier.classify(image_file_path)
        # Convert classified info to text context
        context = f"Image context: {classified_info}"
        return context
