import torch
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LLMConversation:
    def __init__(
        self, model_id="meta-llama/Meta-Llama-3-8B-Instruct", summarizer=False
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.llm = HuggingFacePipeline(
            pipeline=pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=512,
                temperature=0.7,
                repetition_penalty=1.03,
                do_sample=True,
            )
        )
        self.summarizer = HuggingFacePipeline(
            pipeline=pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=512,
                temperature=0.1,
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

    def get_response(self, input_text):
        return self.conversation_chain.predict(input=input_text)
