import torch
from langchain_core.prompts.prompt import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class LLMConversation:
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_template = """
<|begin_of_text|>
<|start_header_id|>user<|end_header_id|>
{input}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    summary_template = """
<|begin_of_text|>
<|start_header_id|>assistant<|end_header_id|>
{summary}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
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
        :param quantization_config: Configuration for quantization.
        :param kwargs: Additional parameters for model loading.
        """
        self.summaries = {}
        self.system_prompt = PromptTemplate(
            input_variables=["input"], template=self.prompt_template
        )
        self.summarizer_prompt = PromptTemplate(
            input_variables=["summary", "new_lines"], template=self.summary_template
        )

        self.model_id = kwargs.pop("model_id", self.model_id)
        self.use_summarize = kwargs.pop("use_summarize", False)
        self.device = kwargs.pop("device", self.device)
        self.quantization_config = kwargs.pop(
            "quantization_config", self.quantization_config
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=self.quantization_config,
            **kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    def get_response(self, input_text: dict, session_id: str = None) -> str:
        """
        Get a response from the conversation chain.

        :param input_text: The input text to get a response for.
        :param session_id: The session ID for handling session-based processing.
        :return: The response from the conversation chain.
        """
        mood = input_text.get("audio")
        subject, action, location = input_text.get("image", (None, None, None))
        context = input_text.get("text")

        prompt = str(self.base_prompt)

        if self.use_summarize and session_id in self.summaries:
            prompt += f"Previous context: {self.summaries[session_id]}\n"

        if context is not None:
            prompt += f"Additional context: {context}\n"

        if (subject, action, location) != (None, None, None) or mood:
            prompt += f"Subject: {subject}\nAction: {action}\nLocation: {location}\n"

        if mood:
            prompt += f"Atmosphere: {mood}\n"

        prompt += "\nIncorporate all of the details provided.\nStory:\n"

        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt")
        input_tokens = input_tokens.to(self.device)
        output_tokens = self.llm.generate(
            input_tokens,
            eos_token_id=self.terminators,
            max_length=input_tokens.size(1) + 256,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        output_story = self.tokenizer.decode(
            output_tokens[0][input_tokens.shape[-1] :], skip_special_tokens=True
        )

        if self.use_summarize:
            prompt = str(self.summary_prompt)

            if session_id in self.summaries:
                prompt += f"{self.summaries[session_id]}\n"

            prompt += f"{output_story}\n"
            prompt += (
                "\nOnly summarize the events. Don't add anything else.\n\nSummary:\n"
            )

            input_tokens = self.tokenizer.encode(prompt, return_tensors="pt")
            input_tokens = input_tokens.to(self.device)
            output_tokens = self.llm.generate(
                input_tokens,
                eos_token_id=self.terminators,
                max_length=input_tokens.size(1) + 128,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            self.summaries[session_id] = self.tokenizer.decode(
                output_tokens[0][input_tokens.size(1) :], skip_special_tokens=True
            )

        return output_story
