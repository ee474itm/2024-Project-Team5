from setuptools import find_packages, setup

setup(
    name="storystream",
    version="0.0.0",
    url="https://github.com/klae01/ee474_stroytelling_ai",
    author="Hosu Lee",
    author_email="tspt2479@gmail.com",
    description="",
    packages=find_packages(),
    install_requires=["langchain", "huggingface_hub", "transformers"],
)
