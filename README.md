# MUSE: Multimodal Utility for Story Enhancement

## Overview

MUSE (Multimodal Utility for Story Enhancement) is a system designed to enhance storytelling by incorporating multimodal inputs such as audio, images, and text. By leveraging sentiment analysis, object recognition, and action recognition, MUSE aims to reduce the burden of the prewriting process and enrich the storytelling experience through interactivity and multimodality.

## Table of Contents
1. [Motivation](#motivation)
2. [System Design Overview](#system-design-overview)
3. [Components](#components)
4. [Installation](#installation)
5. [Usage](#usage)
6. [License](#license)
7. [Acknowledgments](#acknowledgments)

## Motivation

### Value of Storytelling
- **Creative, Educational, Healing**: Storytelling is a powerful tool for creativity, education, and healing.
- **Challenges**: The high cost and complexity of using user-generated text for storytelling can be prohibitive.
- **Interactive Stories**: Incorporating images and audio can enrich storytelling, making it more engaging and accessible.

### Commitment in the Writing Process
- **Time-Consuming Prewriting**: Prewriting can take up to 85% of a writer's time, often making it a daunting task.
- **Fear of Commitment**: Writers may find the initial stages of writing intimidating due to the commitment required to their ideas.
- **Goal**: MUSE aims to alleviate these burdens by introducing interactivity and multimodality, making the writing process more flexible and less intimidating.

### Our Approach with MUSE
- **Interactivity**: By incorporating interactive elements, MUSE helps reduce the fear of commitment, allowing writers to explore and develop ideas more freely.
- **Multimodal Inputs**: Utilizing images and audio provides flexibility and enhances the storytelling experience, making it more dynamic and engaging.

## System Design Overview

The system is composed of several components:
- **Audio Sentiment Analysis**: Analyzes the sentiment of audio files.
- **Image Description**: Generates descriptions for the images related to the story.
- **Language Model (LLM)**: Generates stories based on the provided context.

### Workflow
1. User inputs audio, image, and text.
2. Audio sentiment is classified.
3. LLaVA processes the image and generates context.
4. Memory integrates the context and generates a story.
5. LLaVA continues the story based on the generated context.

## Components

### Audio Sentiment Classifier
- Analyzes the sentiment of audio inputs to understand the emotional tone.
- Uses a pretrained model (`M2EClassifier`) to classify the sentiment of audio inputs.
- Extracts features such as zero-crossing rate, chroma, MFCC, RMS, and mel-spectrogram using `librosa`.

### Image Context Extraction
- Uses image analysis to describe the main subject, action, and location within an image.
- LLaVA (`llava-v1.6-mistral-7b-hf`) is used for object and action recognition in images.
- Generates descriptive context about the main subject, actions, and location in the images.

### Story Generation
- Utilizes the extracted audio and image contexts to generate and enhance stories interactively.
- Uses a language model LLaMA3 (`Meta-Llama-3-8B-Instruct`) for generating and summarizing stories.
- Integrates audio and visual contexts to create coherent and creative narratives.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/klae01/ee474_stroytelling_ai.git
   cd ee474_stroytelling_ai
   ```

2. **Install Dependencies**:
   ```bash
   pip install .
   ```

## Usage

You can run the MUSE web demo in two configurations:

### Basic Configuration (Single Machine)

1. **Start the Server**:
   ```bash
   python app_server.py
   ```

2. **Start the Client**:
   Open a new terminal and run:
   ```bash
   streamlit run app.py
   ```

### Advanced Configuration (Separate Machines)

1. **Start the Server**:
   On the backend machine:
   ```bash
   python app_server.py --hostname <hostname> --port <port>
   ```

2. **Start the Client**:
   On the frontend machine:
   ```bash
   streamlit run app.py -- --host <hostname> --port <port> --save_dir <save_directory>
   ```

- Ensure `<hostname>` and `<port>` match on both server and client.
- Specify `<save_directory>` to store uploaded files and responses.

### Interacting with the Application

1. Open the Streamlit application in your web browser.
2. Upload audio, image, or text files using the file uploader.
3. Enter text input in the chat box and interact with the system to generate and enhance stories.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
Special thanks to the project team for their contributions and efforts in developing MUSE.
