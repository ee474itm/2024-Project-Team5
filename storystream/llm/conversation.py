from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain_community.llms import HuggingFacePipeline


class LLMConversation:
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B", summarizer=False):
        self.llm = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 512,
                "top_k": 30,
                "temperature": 0.7,
                "repetition_penalty": 1.03,
            },
        )
        self.summarizer = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 512,
                "temperature": 0.1,
                "repetition_penalty": 1.03,
            },
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
