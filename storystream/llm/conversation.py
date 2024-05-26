import torch
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain_community.llms import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


class LLMConversation:
    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        summarizer: bool = False,
        device: torch.device = None,
        quantization_config: BitsAndBytesConfig = BitsAndBytesConfig(load_in_8bit=True),
        **kwargs,
    ):
        """
        Initialize the LLMConversation class.

        :param model_id: The model ID to load.
        :param summarizer: Boolean flag to determine if summarizer should be used.
        :param device: The device to load the model on.
        :param quantization_config: Configuration for quantization.
        :param kwargs: Additional parameters for model loading.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            quantization_config=quantization_config,
            **kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.llm = HuggingFacePipeline(
            pipeline=pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.3,
                repetition_penalty=1.03,
                do_sample=True,
            )
        )
        self.summarizer = HuggingFacePipeline(
            pipeline=pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                repetition_penalty=1.03,
                do_sample=False,
            )
        )
        if summarizer:
            self.conversation_chain = ConversationChain(
                llm=self.llm,
                memory=ConversationSummaryMemory(llm=self.summarizer),
                verbose=True,
            )
        else:
            self.conversation_chain = ConversationChain(llm=self.llm, verbose=True)

    def get_response(self, input_text: str) -> str:
        """
        Get a response from the conversation chain.

        :param input_text: The input text to get a response for.
        :return: The response from the conversation chain.
        """
        return self.conversation_chain.predict(input=input_text)
