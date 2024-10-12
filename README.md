# Voice-Driven Emotion Separation for 3D Facial Expression using an Emotion-Separating Encoder

<img src="https://camo.githubusercontent.com/2722992d519a722218f896d5f5231d49f337aaff4514e78bd59ac935334e916a/68747470733a2f2f692e696d6775722e636f6d2f77617856496d762e706e67" alt="Oryx Video-ChatGPT" data-canonical-src="https://i.imgur.com/waxVImv.png" style="max-width: 100%;">

## Overview

Our paper introduces a novel voice-driven 3D facial expression system that adeptly separates emotional cues from vocal content, generating distinct facial animations for various emotional states. Through the use of an emotion-separating encoder and a voice-to-face morph synthesizer, we enhance emotional expression in virtual characters, with significant improvements over existing methods as validated by experiments on multiple datasets. This advancement has broad applications in education, marketing, healthcare, and entertainment, enriching user experiences through more emotionally resonant virtual interactions.

# 👁️💬 Architecture

The proposed model undertakes the task of separating the emotional and content aspects inherent in the voice data of a patient utilizing two embedding spaces from a given voice input. These extracted features from the embedding spaces are combined and subsequently passed through the voice-to-face morph synthesizer. This morph synthesizer then produces morph coefficients enhanced with emotional cues. These coefficients serve as valuable inputs to dense layers for emotion classification.

<img style="max-width: 100%;" src="https://github.com/swerizwan/verhm/blob/main/resources/overview.png" alt="VERHM Overview">

# Demo

```
python3 run_demo.py --input_voice "./newinput/angry.wav"
```

<div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
    <div style="text-align: center;">
        <p>Frustrated</p>
        <img style="width: 20%;" src="https://github.com/swerizwan/voiceemo/blob/main/resources/image1.gif" alt="Frustrated">
    </div>
    <div style="text-align: center;">
        <p>Sad</p>
        <img style="width: 20%;" src="https://github.com/swerizwan/voiceemo/blob/main/resources/image2.gif" alt="Sad">
    </div>
    <div style="text-align: center;">
        <p>Angry</p>
        <img style="width: 20%;" src="https://github.com/swerizwan/voiceemo/blob/main/resources/image3.gif" alt="Angry">
    </div>
</div>

## Installation

Follow these steps to set up the project:

1. Download the repository.
2. Create a conda virtual environment using: `conda create -n verhm python=3.7`.
3. Activate the environment: `conda activate verhm`.
4. Install all required dependencies mentioned in the Workflow section.

To begin, ensure you have the following essential libraries installed on your system:

- Pytorch 1.9.0
- CUDA 11.3
- Blender 3.4.1
- ffmpeg 4.4.1
- torch==1.9.0
- torchvision==0.10.0
- torchaudio==0.9.0
- numpy
- scipy==1.7.1
- librosa==0.8.1
- tqdm
- pickle
- transformers==4.6.1
- trimesh==3.9.27
- pyrender==0.1.45
- opencv-python

## Datasets

Our project utilizes several datasets for training and testing purposes:

- **VOCASET**: Offers audio-4D scan pairs for emotion analysis. [Dataset Link](https://voca.is.tue.mpg.de/download.php) `python main.py --dataset vocaset`
- **RAVDESS**: The RAVDESS comprises 7,356 files, totaling 24.8 GB in size. It features recordings from 24 professional actors, evenly split between genders. [Dataset Link](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) `python main.py --dataset ravdess`
- **MEAD**: The MEAD \cite{Mead} features 60 actors and actresses conversing with eight distinct emotions at varying intensity levels. [Dataset Link](https://wywu.github.io/projects/MEAD/MEAD.html/) `python main.py --dataset mead`

## Running the Demo

To run the demo, follow these steps:

1. Download Blender from [Blender Official Website](https://www.blender.org/download/), and place it in the Blender folder within the root directory.
2. Download the pre-trained model from [Pre-Trained Model Link](https://drive.google.com/file/d/1ywEYhMWdxWk9Bqt0UIOdAyYM6v8JUF-K/view?usp=sharing) and put it in the `pre-trained` folder in the root directory.
3. Run the demo by executing `run_demo.py` with the desired input voice. 
