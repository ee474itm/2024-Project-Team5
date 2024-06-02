import argparse
import time

import rpyc
from rpyc.utils.server import ThreadedServer


class StoryGenerationService(rpyc.Service):
    def on_connect(self, conn):
        print("Client connected")

    def on_disconnect(self, conn):
        print("Client disconnected")

    def exposed_generate_story(
        self, session_id=None, audio=None, image=None, text=None
    ):
        """Generate a story in parts, simulating a streaming response."""
        for i in range(5):
            yield f"Part {i + 1} of the story. Session ID: {session_id}"
            time.sleep(1)


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
