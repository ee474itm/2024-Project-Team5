import itertools
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


@dataclass
class SessionData:
    summary: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    summary_future: Optional[Future] = None


class LLMConversation:
    default_config = {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "use_summarize": False,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    base_prompt = (
        "You must take the role of an author writing a short story. "
        "Make a short story based on the following details:\n"
    )
    summary_prompt = "Summarize the following story:\n"

    def __init__(self, **kwargs):
        """
        Initialize the LLMConversation class.

        :param model_id: The model ID to load.
        :param use_summarize: Boolean flag to determine if summarizer should be used.
        :param device: The device to load the model on.
        :param low_cpu_mem_usage: Boolean flag to use low CPU memory usage.
        :param kwargs: Additional parameters for model loading.
        """
        self.sessions: Dict[str, SessionData] = {}

        config = self.default_config.copy()
        config.update(kwargs)

        self.model_id = config.pop("model_id")
        self.use_summarize = config.pop("use_summarize")
        self.device = config.pop("device")
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        self.executor = ThreadPoolExecutor()

    def get_response(
        self, input_text: dict, session_id: str = None, realtime: bool = False
    ) -> Union[str, Tuple[TextIteratorStreamer, Future]]:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionData()

        session_data = self.sessions[session_id]

        with session_data.lock:
            if session_data.summary_future and session_data.summary_future.done():
                session_data.summary = session_data.summary_future.result()
                session_data.summary_future = None

        storystream = self.generate_story(input_text, session_id, realtime)
        if realtime:
            output_story, future = storystream
            os1, os2 = itertools.tee(output_story, 2)
        else:
            future = None
            os1, os2 = [storystream] * 2

        if self.use_summarize:
            with session_data.lock:
                session_data.summary_future = self.executor.submit(
                    self.generate_summary, session_id, os2, future
                )

        return os1

    def generate_story(
        self, input_text: dict, session_id: str, realtime: bool = False
    ) -> Union[str, Tuple[TextIteratorStreamer, Future]]:
        prompt = self.build_story_prompt(input_text, session_id)
        return self.generate_text(
            prompt,
            max_length=256,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            realtime=realtime,
        )

    def generate_summary(
        self, session_id: str, story: str, future: Future = None
    ) -> str:
        if future and not future.done():
            future.result()
        if not isinstance(story, str):
            story = "".join(story)
        prompt = self.build_summary_prompt(session_id, story)
        return self.generate_text(prompt, max_length=128, do_sample=False)

    def build_story_prompt(self, input_text: dict, session_id: str) -> str:
        mood = input_text.get("audio")
        image_context = input_text.get("image", {})
        subject = image_context.get("subject")
        action = image_context.get("action")
        location = image_context.get("location")
        context = input_text.get("text")

        prompt = str(self.base_prompt)

        if self.use_summarize and session_id in self.sessions:
            prompt += f"Previous context: {self.sessions[session_id].summary}\n"

        if context:
            prompt += f"Additional context: {context}\n"

        if subject or action or location:
            prompt += f"Subject: {subject}\nAction: {action}\nLocation: {location}\n"

        if mood:
            prompt += f"Atmosphere: {mood}\n"

        prompt += "\nIncorporate all of the details provided.\nStory:\n"
        return prompt

    def build_summary_prompt(self, session_id: str, story: str) -> str:
        prompt = self.summary_prompt

        if session_id in self.sessions:
            prompt += f"{self.sessions[session_id].summary}\n"

        prompt += f"{story}\n"
        prompt += "\nOnly summarize the events. Don't add anything else.\n\nSummary:\n"
        return prompt

    def generate_text(
        self,
        prompt: str,
        max_length: int,
        do_sample: bool,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        realtime: bool = False,
    ) -> Union[str, Tuple[TextIteratorStreamer, Future]]:
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )
        generate_kwargs = {
            "input_ids": input_tokens,
            "eos_token_id": self.terminators,
            "max_length": input_tokens.size(1) + max_length,
            "num_return_sequences": 1,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if top_k is not None:
            generate_kwargs["top_k"] = top_k
        if top_p is not None:
            generate_kwargs["top_p"] = top_p
        if temperature is not None:
            generate_kwargs["temperature"] = temperature

        if realtime:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
            return streamer, self.executor.submit(
                self.handle_realtime_generate, generate_kwargs, streamer
            )
        else:
            output_tokens = self.llm.generate(**generate_kwargs)
            return self.tokenizer.decode(
                output_tokens[0][input_tokens.shape[-1] :], skip_special_tokens=True
            )

    def handle_realtime_generate(self, generate_kwargs, streamer):
        self.llm.generate(**generate_kwargs, streamer=streamer)
        streamer.end()
