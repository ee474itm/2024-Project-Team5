import argparse
import copy
import typing
import uuid

import rpyc
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile


class StreamCache:
    def __init__(self, iterable):
        self.iterable = iterable
        self.cache = []
        self.iterator = iter(self.iterable)
        self.completed = False

    def __iter__(self):
        for item in self.cache:
            yield item
        if not self.completed:
            for item in self.iterator:
                self.cache.append(item)
                yield item
            self.completed = True


class StoryGeneratorApp:
    def __init__(self, host: str, port: int):
        self.title = "MUSE"
        self.host = host
        self.port = port
        self.initialize_session_state()
        self.story_generator = rpyc.connect(
            host, port, config={"sync_request_timeout": 120}
        )

    def story_generation_fn(
        self,
        audio: list[UploadedFile] = None,
        image: list[UploadedFile] = None,
        text: list[str] = None,
    ):
        audio_info = (
            [{"name": a.name, "data": a.getvalue()} for a in audio] if audio else []
        )
        image_info = (
            [{"name": i.name, "data": i.getvalue()} for i in image] if image else []
        )

        return self.story_generator.root.generate_story(
            session_id=st.session_state.session_id,
            audio=audio_info,
            image=image_info,
            text=text,
            realtime=True,
        )

    def initialize_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        if "chat_index" not in st.session_state:
            st.session_state.chat_index = 0

    def display_title(self):
        st.title(self.title)

    def handle_file_upload(self) -> list[UploadedFile]:
        return st.file_uploader(
            "Upload files",
            type=["png", "jpg", "jpeg", "mp3", "wav"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.chat_index}",
        )

    def handle_text_input(self) -> str:
        return st.chat_input("Say something")

    def process_uploaded_files(
        self, uploaded_files: list[UploadedFile]
    ) -> dict[str, list[UploadedFile]]:
        contents = {"audio": [], "image": []}
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.type.split("/")[0]
            if file_type in contents:
                contents[file_type].append(uploaded_file)
        return contents

    def append_message(self, role: str, contents: dict[str, typing.Any]):
        st.session_state.messages.append({"role": role, **contents})

    def generate_story(self, contents: dict[str, list[UploadedFile]]):
        return self.story_generation_fn(
            audio=contents["audio"], image=contents["image"], text=contents["text"]
        )

    def display_messages(self):
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
                            elif isinstance(content, typing.Iterable):
                                st.write_stream(content)

    def run(self):
        self.display_title()

        uploaded_files = self.handle_file_upload()
        text_input = self.handle_text_input()

        if text_input:
            contents = {
                "audio": [],
                "image": [],
                "text": [text_input] if text_input else [],
            }

            if uploaded_files:
                file_contents = self.process_uploaded_files(uploaded_files)
                contents["audio"].extend(file_contents["audio"])
                contents["image"].extend(file_contents["image"])

            self.append_message("user", contents)

            story = self.generate_story(copy.deepcopy(contents))
            self.append_message(
                "assistant", {"audio": [], "image": [], "text": [StreamCache(story)]}
            )

            st.session_state.chat_index += 1
            st.rerun()

        self.display_messages()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Streamlit client.")
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host of the rpyc server"
    )
    parser.add_argument(
        "--port", type=int, default=18861, help="Port number of the rpyc server"
    )
    args = parser.parse_args()

    if "app" not in st.session_state:
        st.session_state.app = StoryGeneratorApp(host=args.host, port=args.port)

    st.session_state.app.run()
