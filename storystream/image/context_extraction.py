import torch
from PIL import Image
from transformers import (
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

from storystream.utils import fetch_from_url


class ImageContextExtractor:
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt = """[INST] <image>
Fill in the following form for 'Subject', 'Action', and 'Location'.
Be descriptive. Output only what goes into <Fill in Here>.
1. Describe what the main subject in the image looks like. : Main subject of the image is <Fill in Here>.
2. Describe the action taken by that subject in detail. : Main subject is doing <Fill in Here>.
3. Describe the location in detail. : Main subject is located in <Fill in Here>. 
[/INST]"""

    def __init__(self, **kwargs):
        """
        Initialize the ImageContextExtractor class.

        :param model_id: The model ID to load.
        :param device: The device to load the model on.
        :param quantization_config: Configuration for quantization.
        :param kwargs: Additional parameters for model loading.
        """
        self.model_id = kwargs.pop("model_id", self.model_id)
        self.device = kwargs.pop("device", self.device)
        self.quantization_config = kwargs.pop(
            "quantization_config", self.quantization_config
        )

        self.processor = LlavaNextProcessor.from_pretrained(
            self.model_id, device_map=self.device
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=True,
            quantization_config=self.quantization_config,
            device_map="auto",
            **kwargs,
        )

    def extract_context_from_url(
        self, image_url: str, max_new_tokens: int = 512, **kwargs
    ) -> dict:
        """
        Extract context from an image URL.

        :param image_url: The URL of the image.
        :param max_new_tokens: The maximum number of new tokens to generate. Default is 512.
        :param kwargs: Additional parameters for context extraction.
        :return: The extracted context as a dictionary.
        """
        content = fetch_from_url(image_url)
        image = Image.open(content)

        return self.extract_context(image, max_new_tokens, **kwargs)

    def extract_context_from_file(
        self, image_file_path: str, max_new_tokens: int = 512, **kwargs
    ) -> dict:
        """
        Extract context from an image file.

        :param image_file_path: The path to the image file.
        :param max_new_tokens: The maximum number of new tokens to generate. Default is 512.
        :param kwargs: Additional parameters for context extraction.
        :return: The extracted context as a dictionary.
        """
        image = Image.open(image_file_path)

        return self.extract_context(image, max_new_tokens, **kwargs)

    def extract_context(
        self, image: Image, max_new_tokens: int = 512, **kwargs
    ) -> dict:
        """
        Extract context from an image.

        :param image: The PIL Image object.
        :param max_new_tokens: The maximum number of new tokens to generate. Default is 512.
        :param kwargs: Additional parameters for context extraction.
        :return: The extracted context as a dictionary with keys 'subject', 'action', and 'location'.
        """
        inputs = self.processor(self.prompt, image, return_tensors="pt")
        inputs.to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        context = self.processor.decode(generated_ids[0], skip_special_tokens=True)

        context_list = context.splitlines()
        feature_list = [element for element in context_list if element]
        feature_list = [
            element.split("is", 1)[1].strip()
            for element in feature_list
            if "is" in element
        ]

        return {
            "subject": feature_list[0] if len(feature_list) > 0 else None,
            "action": feature_list[1] if len(feature_list) > 1 else None,
            "location": feature_list[2] if len(feature_list) > 2 else None,
        }
