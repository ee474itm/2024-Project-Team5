import torch
from PIL import Image
from transformers import (
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

from storystream.utils import fetch_from_url


class ImageContextExtractor:
    def __init__(
        self,
        model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        device: torch.device = None,
        quantization_config: BitsAndBytesConfig = BitsAndBytesConfig(load_in_8bit=True),
        **kwargs,
    ):
        """
        Initialize the ImageContextExtractor class.

        :param model_id: The model ID to load.
        :param device: The device to load the model on.
        :param quantization_config: Configuration for quantization.
        :param kwargs: Additional parameters for model loading.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
            device_map=device,
            **kwargs,
        )

    def extract_context_from_url(
        self, image_url: str, max_new_tokens: int = 512, **kwargs
    ) -> str:
        """
        Extract context from an image URL.

        :param image_url: The URL of the image.
        :param max_new_tokens: The maximum number of new tokens to generate. Default is 512.
        :param kwargs: Additional parameters for context extraction.
        :return: The extracted context as a string.
        """
        content = fetch_from_url(image_url)
        image = Image.open(content)

        return self.extract_context(image, max_new_tokens, **kwargs)

    def extract_context_from_file(
        self, image_file_path: str, max_new_tokens: int = 512, **kwargs
    ) -> str:
        """
        Extract context from an image file.

        :param image_file_path: The path to the image file.
        :param max_new_tokens: The maximum number of new tokens to generate. Default is 512.
        :param kwargs: Additional parameters for context extraction.
        :return: The extracted context as a string.
        """
        image = Image.open(image_file_path)

        return self.extract_context(image, max_new_tokens, **kwargs)

    def extract_context(self, image: Image, max_new_tokens: int = 512, **kwargs) -> str:
        """
        Extract context from an image.

        :param image: The PIL Image object.
        :param max_new_tokens: The maximum number of new tokens to generate. Default is 512.
        :param kwargs: Additional parameters for context extraction.
        :return: The extracted context as a string.
        """
        prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
        inputs = self.processor(prompt, image, return_tensors="pt")
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
        context = self.processor.decode(output[0], skip_special_tokens=True)

        return context
