Voice-Driven 3D Facial Emotion Recognition For Mental Health Monitoring
About the Project
Vocal expressions are significant indicators of emotional states, crucial for mental health monitoring. However, existing methods often struggle to accurately separate the emotions and contents conveyed through speech alone. Our solution introduces a neural network that effectively disentangles various emotions from voice signals, enabling the generation of precise 3D facial expressions.

Workflow
To begin, ensure you have the following essential libraries installed on your system:

Pytorch 1.9.0
CUDA 11.3
Blender 3.4.1
ffmpeg 4.4.1
torch==1.9.0
torchvision==0.10.0
torchaudio==0.9.0
numpy
scipy==1.7.1
librosa==0.8.1
tqdm
pickle
transformers==4.6.1
trimesh==3.9.27
pyrender==0.1.45
opencv-python
Installation
Follow these steps to set up the project:

Download the repository.
Create a conda virtual environment using: conda create -n verhm python=3.7.
Activate the environment: conda activate verhm.
Install all required dependencies mentioned in the Workflow section.

Baselines: 
We performed a comparative analysis of the proposed research with four baseline methods:
VOCA: This is a neural network model trained on a 4D face dataset synchronized with voice recordings from multiple speakers. VOCA animates faces in response to spoken language, accommodating any language and speaking style. Additionally, VOCA offers animator controls like jaw, eyeball, and head rotations during animation for adjusting voice style, facial features, and pose dynamics during animation.

MeshTalk: This study introduces a technique for generating facial animations synchronized with voice input. In contrast to previous methodologies that may exhibit drawbacks such as artificial upper-face movements or dependency on particular subjects, it prioritizes achieving facial expressions. This method is the utilization of a distinct categorical latent space for facial animation. 

FaceFormer: A transformer-based model has been developed to generate facial animations based on voice input. Unlike previous methods, FaceFormer takes into consideration voice context and predicts sequences of animated facial emotions. It incorporates self-supervised pre-trained voice representations to overcome the challenge of limited data availability. 

EmoTalk: This study aims to improve the reflection of both speech content and emotion. The network includes an Emotion Disentangling Encoder (EDE) to separate speech emotion from content.  an Emotion-Guided Feature Fusion Decoder generates expressive 3D facial animations. This facilitates the reconstruction of plausible 3D faces from 2D emotional data. Additionally, they contribute a comprehensive dataset, 3D Emotional Talking Face (3D-ETF), to train the network.

Datasets: 
We used VOCASET, RAVDESS, and MEAD datasets for our experiments.

The VOCASET consists of 480 facial expressions captured from 12 individuals (6 male, 6 female). Each expression, recorded at 60fps, ranges between 3 and 4 seconds in duration. The 3D face meshes within each sequence contain 5023 vertices. With its variety of expressions and linguistic features, VOCASET serves as a resource for research in speech-driven animation and virtual character interaction. In this research, we utilized the VOCASET dataset to train and evaluate our model for voice-driven 3D facial emotion recognition. The dataset implementation began with loading and processing audio and corresponding 3D facial vertices data. Audio files were preprocessed using the Wav2Vec2Processor to extract features suitable for training. Each audio sample, associated vertices, and template data were read and stored in a structured format. The data was then divided into training, validation, and test sets based on predefined subject splits, ensuring a comprehensive evaluation of the model's performance.
To run the code using VOCASET, python main.py --dataset vocaset

The RAVDESS comprises 7,356 files, totaling 24.8 GB in size. It features recordings from 24 professional actors, evenly split between genders. Each actor delivers two statements in a neutral North American accent, covering a spectrum of emotions including calm, happy, sad, angry, fearful, surprise, and disgust for speech, and calm, happy, sad, angry, and fearful for song. These expressions are presented at two levels of emotional intensity (normal and strong), along with a neutral expression. We have utilized only the speech data for our experiments. The dataset comprises emotional speech recordings, which were processed to extract Mel-frequency cepstral coefficients (MFCCs) for use as input features. Each audio file was loaded using the librosa library, and MFCCs were computed and padded to a fixed length of 400 to ensure uniform input dimensions for the model. The emotional labels corresponding to the audio files were mapped from the filenames, which include a specific code representing the emotion.
To run the code using RAVDESS, python main.py --dataset ravdess

The MEAD features 60 actors and actresses conversing with eight distinct emotions at varying intensity levels. This work captures high-quality audio-visual clips from seven different viewing angles in a controlled setting. Mead dataset has wide-ranging applications across various research domains, including conditional generation, cross-modal understanding, and expression recognition. Implementing the MEAD dataset in our research involved a meticulous process of data preparation, feature extraction, and dataset creation for effective model training and evaluation. Initially, we collected audio file paths from the specified directory, ensuring that only .wav files were included. The file paths were shuffled to introduce randomness and subsequently divided into training, validation, and testing sets. Feature extraction was performed on the audio files using the librosa library, where Mel-frequency cepstral coefficients (MFCCs) were computed. Each audio sample's MFCCs were padded or truncated to ensure a uniform length of 100 frames. This preprocessing step was crucial for maintaining consistency across the dataset, facilitating efficient model training. Labels were generated randomly for demonstration purposes, representing seven different classes. The dataset was then wrapped in DataLoader instances for batched and parallelized processing during the training and evaluation phases.
To run the code using MEAD, python main.py --dataset mead


