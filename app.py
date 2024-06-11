import argparse
import os
import re
import typing
import uuid
from datetime import datetime

import rpyc
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile


def clean_up_text(text: str) -> str:
    if match := re.search(r'(.*[.!?]["\')\]]*(\s|\s?$))', text, re.DOTALL):
        cleaned_text = match.group(1).rstrip()
        if text[len(cleaned_text) :].strip():
            if paragraphs := "\n\n".join(text.split("\n\n")[:-1]):
                return paragraphs
        return cleaned_text
    return ""


class StreamCache:
    def __init__(
        self,
        iterable: typing.Iterable[str],
        callbacks_on_finish: typing.Optional[typing.List[typing.Callable]] = None,
    ):
        self.iterable = iterable
        self.cache = []
        self.iterator = iter(self.iterable)
        self.completed = False
        self.callbacks_on_finish = callbacks_on_finish or []

    def __iter__(self):
        yield from self.cache
        if not self.completed:
            for item in self.iterator:
                self.cache.append(item)
                yield item
            self.clean()

    def clean(self):
        self.cache = clean_up_text("".join(self.cache))
        self.completed = True
        for callback in self.callbacks_on_finish:
            callback(str(self.cache))


class FileHandler:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    @staticmethod
    def process_uploaded_files(
        uploaded_files: list[UploadedFile],
    ) -> dict[str, list[UploadedFile]]:
        """Process uploaded files and categorize them by type."""
        contents = {"audio": [], "image": []}
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.type.split("/")[0]
            if file_type in contents:
                contents[file_type].append(uploaded_file)
        return contents

    def save_file(self, file):
        with open(os.path.join(self.save_dir, file.name), "wb") as f:
            f.write(file.getvalue())

    def save_uploaded_files(self, contents: dict[str, list[UploadedFile]]):
        """Save uploaded files to the local filesystem."""
        os.makedirs(self.save_dir, exist_ok=True)

        for media in contents["audio"] + contents["image"]:
            self.save_file(media)

        # Save text input to a file
        for index, text in enumerate(contents.get("text", [])):
            with open(os.path.join(self.save_dir, f"userinput_{index}.txt"), "w+") as f:
                f.write(text)

    def log_response(self, response: typing.Iterable[str]):
        """Log the model's response to a file."""
        with open(os.path.join(self.save_dir, "response.txt"), "w+") as f:
            for line in response:
                f.write(line)


class StoryGeneratorApp:
    def __init__(self, host: str, port: int, save_dir: str):
        self.title = "MUSE"
        self.host = host
        self.port = port
        self.save_dir = save_dir
        self.initialize_session_state()
        self.story_generator = rpyc.connect(
            host, port, config={"sync_request_timeout": 120}
        )

    def initialize_session_state(self):
        """Initialize session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        if "chat_index" not in st.session_state:
            st.session_state.chat_index = 0

    def display_title(self):
        """Display the app title."""
        st.title(self.title)

    def handle_file_upload(self) -> list[UploadedFile]:
        """Handle file uploads and return a list of uploaded files."""
        return st.file_uploader(
            "Upload files",
            type=["png", "jpg", "jpeg", "mp3", "wav"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.chat_index}",
        )

    def handle_text_input(self) -> str:
        """Handle text input from the user."""
        return st.chat_input(
            "Say something", key=f"textbox_{st.session_state.chat_index}"
        )

    def append_message(self, role: str, contents: dict[str, typing.Any]):
        """Append a message to the session state."""
        st.session_state.messages.append({"role": role, **contents})

    def get_save_dir(self) -> str:
        session_id = st.session_state.session_id
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.save_dir, f"{datetime_str}_{session_id}")

    def display_messages(self):
        """Display all messages from the session state."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content_type in ["image", "audio", "text"]:
                    for content in message.get(content_type, []):
                        if content_type == "image":
                            st.image(content)
                        elif content_type == "audio":
                            st.audio(content)
                        elif content_type == "text":
                            if isinstance(content, str):
                                st.write(content)
                            elif isinstance(content, StreamCache):
                                if content.completed:
                                    st.write(content.cache)
                                else:
                                    st.write_stream(content)
                            else:
                                raise ValueError(f"Unknown type: {type(content)}")

    def run(self):
        """Run the main app logic."""
        self.display_title()

        text_input = self.handle_text_input()
        self.display_messages()
        uploaded_files = self.handle_file_upload()

        if text_input:
            contents = {
                "audio": [],
                "image": [],
                "text": [text_input] if text_input else [],
            }

            if uploaded_files:
                file_contents = FileHandler.process_uploaded_files(uploaded_files)
                contents["audio"].extend(file_contents["audio"])
                contents["image"].extend(file_contents["image"])

            self.append_message("user", contents)

            # Save uploaded files if save_dir is provided
            callbacks_on_finish = []
            if self.save_dir:
                save_dir = self.get_save_dir()
                file_handler = FileHandler(save_dir)
                file_handler.save_uploaded_files(contents)
                callbacks_on_finish.append(file_handler.log_response)
            callbacks_on_finish.append(lambda _: st.rerun())

            # Generate story and wrap in StreamCache
            response = self.story_generator.root.generate_story(
                session_id=st.session_state.session_id,
                audio=[
                    {"name": a.name, "data": a.getvalue()} for a in contents["audio"]
                ],
                image=[
                    {"name": i.name, "data": i.getvalue()} for i in contents["image"]
                ],
                text=contents.get("text", []),
                realtime=True,
            )

            story = StreamCache(response, callbacks_on_finish=callbacks_on_finish)

            self.append_message(
                "assistant", {"audio": [], "image": [], "text": [story]}
            )
            st.session_state.chat_index += 1
            st.rerun()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Streamlit client.")
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host of the rpyc server"
    )
    parser.add_argument(
        "--port", type=int, default=18861, help="Port number of the rpyc server"
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help="Directory to save uploaded files"
    )
    args = parser.parse_args()

    if "app" not in st.session_state:
        st.session_state.app = StoryGeneratorApp(
            host=args.host, port=args.port, save_dir=args.save_dir
        )

    st.session_state.app.run()
