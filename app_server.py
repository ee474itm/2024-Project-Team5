import argparse
import os
import pathlib
import tempfile
import threading

import rpyc
from rpyc.utils.server import ThreadedServer

from storystream.story_generation import StoryGenerator


class StoryGenerationService(rpyc.Service):
    lock = threading.Lock()
    story_generator = None

    def on_connect(self, conn):
        with self.lock:
            if StoryGenerationService.story_generator is None:
                self.initialize_story_generator()
        print("Client connected")

    @classmethod
    def initialize_story_generator(cls):
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(
            base_dir, "storystream", "audio", "m2e_classifier_jhs.pth"
        )
        image_params = {}
        classifier_params = {"model_path": model_path}
        audio_params = {"classifier_params": classifier_params}
        llm_params = {"use_summarize": True}
        cls.story_generator = StoryGenerator(audio_params, image_params, llm_params)

    def on_disconnect(self, conn):
        print("Client disconnected")

    def save_temp_file(self, name, data):
        """Save data to a temporary file and return the file path."""
        suffix = pathlib.Path(name).suffix
        temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=suffix)
        temp_file.write(data)
        temp_file.flush()
        return temp_file

    def exposed_generate_story(self, session_id, audio, image, text, realtime):
        """Generate a story using provided audio, image, and text data."""
        audio_files = [self.save_temp_file(x["name"], x["data"]) for x in audio]
        image_files = [self.save_temp_file(x["name"], x["data"]) for x in image]

        return StoryGenerationService.story_generator.generate_story(
            session_id=session_id,
            audio_file_path=audio_files[0].name if audio_files else None,
            image_file_path=image_files[0].name if image_files else None,
            text=text,
            realtime=realtime,
        )

        return StoryGenerationService.story_generator.generate_story(
            session_id=session_id,
            audio_file_paths=[fp.name for fp in audio_files],
            image_file_paths=[fp.name for fp in image_files],
            text=text,
            realtime=realtime,
        )


def start_server(hostname, port):
    """Start the RPyC server on the given hostname and port."""
    server = ThreadedServer(StoryGenerationService, hostname=hostname, port=port)
    print(f"Starting server on {hostname}:{port}")
    server.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the story generation server.")
    parser.add_argument(
        "--hostname",
        type=str,
        default="localhost",
        help="Hostname to run the server on",
    )
    parser.add_argument(
        "--port", type=int, default=18861, help="Port number to run the server on"
    )
    args = parser.parse_args()

    start_server(args.hostname, args.port)
