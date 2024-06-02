import copy
import typing

import streamlit as st


class StoryGeneratorApp:
    _story_generation_instance = None

    def __init__(self):
        self.title = "MUSE"
        self.initialize_session_state()

    @staticmethod
    def story_generation_fn(audio=None, image=None, text=None):
        return "Lorem Ipsum"

    def initialize_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.session_id = 0

    def display_title(self):
        st.title(self.title)

    def handle_file_upload(self):
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["png", "jpg", "jpeg", "mp3", "wav"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.session_id}",
        )
        return uploaded_files

    def handle_text_input(self):
        return st.chat_input("Say something")

    def process_uploaded_files(self, uploaded_files):
        contents = {"audio": [], "image": []}
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.type.split("/")[0]
            if file_type == "image":
                contents["image"].append(uploaded_file)
            elif file_type == "audio":
                contents["audio"].append(uploaded_file)
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
                    for content in message[content_type]:
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

            st.session_state.session_id += 1
            st.rerun()

        self.display_messages()


if "app" not in st.session_state:
    st.session_state.app = StoryGeneratorApp()

st.session_state.app.run()
