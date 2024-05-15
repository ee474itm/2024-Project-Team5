from storystream.context_extraction import ContextExtractor
from storystream.llm.conversation import LLMConversation


class StoryGenerator:
    def __init__(self, summarizer: bool = False):
        self.context_extractor = ContextExtractor()
        self.llm_conversation = LLMConversation(summarizer=summarizer)

    def generate_story(
        self, audio_file_path=None, image_file_path=None, motion_file_path=None
    ):
        # Extract context from the provided multimedia files
        context = self.context_extractor.extract_all_contexts(
            audio_file_path=audio_file_path,
            image_file_path=image_file_path,
            motion_file_path=motion_file_path,
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
        motion_file_path="path/to/motion/file",
    )
    print("Generated Story:", story)
