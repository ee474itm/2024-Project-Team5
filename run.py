from storystream.story_generation import StoryGenerator

# Instantiate the StoryGenerator
story_generator = StoryGenerator(summarizer=True)

# Generate a story
story = story_generator.generate_story(
    audio_file_path="path/to/audio/file",
    image_file_path="path/to/image/file",
)

# Print the generated story
print("Generated Story:", story)
