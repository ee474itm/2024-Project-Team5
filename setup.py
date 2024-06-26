from setuptools import find_packages, setup

setup(
    name="storystream",
    version="0.1.0",
    url="https://github.com/klae01/ee474_stroytelling_ai",
    author="Hosu Lee",
    author_email="tspt2479@gmail.com",
    description="Story generation using multimodal inputs.",
    packages=find_packages(),
    install_requires=[
        "accelerate",
        "librosa==0.10.2.post1",
        "pillow",
        "requests",
        "rpyc",
        "streamlit>=1.35.0",
        "torch",
        "transformers>=4.41.0,<=4.41.2",
    ],
)
