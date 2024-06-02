import argparse
import copy
import typing
import uuid

import rpyc
import streamlit as st


class StoryGeneratorApp:
    def __init__(self, host, port):
        self.title = "MUSE"
        self.host = host
        self.port = port
        self.initialize_session_state()
        self.story_generator = rpyc.connect(host, port)

    def story_generation_fn(self, audio=None, image=None, text=None):
        return self.story_generator.root.generate_story(
            session_id=st.session_state.session_id, audio=audio, image=image, text=text
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

    def handle_file_upload(self):
        return st.file_uploader(
            "Upload files",
            type=["png", "jpg", "jpeg", "mp3", "wav"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.chat_index}",
        )

    def handle_text_input(self):
        return st.chat_input("Say something")

    def process_uploaded_files(self, uploaded_files):
        contents = {"audio": [], "image": []}
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.type.split("/")[0]
            if file_type in contents:
                contents[file_type].append(uploaded_file)
        return contents

    def append_message(self, role, contents):
        st.session_state.messages.append({"role": role, **contents})

    def generate_story(self, contents):
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
            contents = {"audio": [], "image": [], "text": []}
            if text_input:
                contents["text"].append(text_input)

            if uploaded_files:
                file_contents = self.process_uploaded_files(uploaded_files)
                contents["audio"].extend(file_contents["audio"])
                contents["image"].extend(file_contents["image"])

            self.append_message("user", contents)

            story = self.generate_story(copy.deepcopy(contents))
            self.append_message(
                "assistant", {"audio": [], "image": [], "text": [story]}
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
