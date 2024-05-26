from storystream.context_extraction import ContextExtractor
from storystream.llm.conversation import LLMConversation


class StoryGenerator:
    def __init__(self, audio_params=None, image_params=None, llm_params=None):
        """
        Initialize the StoryGenerator class.

        :param audio_params: Dictionary of parameters to pass to AudioContextExtractor.
        :param image_params: Dictionary of parameters to pass to ImageContextExtractor.
        :param llm_params: Dictionary of parameters to pass to LLMConversation.
        """
        self.context_extractor = ContextExtractor(
            audio_params=audio_params, image_params=image_params
        )
        self.llm_conversation = LLMConversation(**(llm_params or {}))

    def generate_story(self, audio_file_path=None, image_file_path=None, **kwargs):
        """
        Generate a story from the provided multimedia files.

        :param audio_file_path: The path to the audio file.
        :param image_file_path: The path to the image file.
        :param kwargs: Additional parameters for context extraction.
        :return: The generated story as a string.
        """
        # Extract context from the provided multimedia files
        context = self.context_extractor.extract_all_contexts(
            audio_file_path=audio_file_path, image_file_path=image_file_path, **kwargs
        )

        # Generate story using the LLM with the extracted context
        story = self.llm_conversation.get_response(context)
        return story


# Example usage
if __name__ == "__main__":
    story_generator = StoryGenerator(summarizer=True)
    story = story_generator.generate_story(
        audio_file_path="path/to/audio/file",
        image_file_path="path/to/image/file",
    )
    print("Generated Story:", story)
