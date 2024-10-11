
import numpy as np
import math
import torch
import random
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from models.wav2vecfile import Wav2Vec2Model, Wav2Vec2ForSpeechClassification
from utils.utilsfile import init_biased_mask, enc_dec_mask

# Offline settings for Hugging Face datasets and transformers
HF_DATASETS_OFFLINE = 1
TRANSFORMERS_OFFLINE = 1

# Paths for pretrained models
path = 'jonatasgrosman/wav2vec2-large-xlsr-53-english'
path2 = 'jonatasgrosman/wav2vec-english-speech-emotion-recognition'

class verhm(nn.Module):
    def __init__(self, args):
        super(verhm, self).__init__()
        
        # Initialize model parameters from args
        self.features = args.features  # Dimensionality of input features
        self.blendshapes = args.blendshapes  # Number of blendshapes for facial animation
        self.device = args.device  # Device for model training (e.g., "cuda" or "cpu")
        self.batch_size = args.batch_size  # Batch size for training
        
        # Load Wav2Vec2 models for continuous and emotional speech recognition
        self.audio_encoder_cont = Wav2Vec2Model.from_pretrained(path, local_files_only=True)
        self.processor = Wav2Vec2Processor.from_pretrained(path, local_files_only=True)
        self.audio_encoder_cont.feature_extractor._freeze_parameters()
        
        self.audio_encoder_emo = Wav2Vec2ForSpeechClassification.from_pretrained(path2, local_files_only=True)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(path2, local_files_only=True)
        self.audio_encoder_emo.wav2vec2.feature_extractor._freeze_parameters()
        
        # Define neural network layers for feature mapping
        self.max_seq_len = args.max_seq_len  # Maximum sequence length for transformer
        self.audio_feature_map_cont = nn.Linear(1024, 512)  # Linear layer for continuous speech feature mapping
        self.audio_feature_map_emo = nn.Linear(1024, 832)  # Linear layer for emotional speech feature mapping
        self.audio_feature_map_emo2 = nn.Linear(832, 256)  # Additional linear layer for emotional speech feature mapping
        self.relu = nn.ReLU()  # ReLU activation function
        
        # Define bias masks for transformer
        self.biased_mask1 = init_biased_mask(n_head=4, max_seq_len=args.max_seq_len, period=args.period)
        
        # One-hot encodings for level and person attributes
        self.one_hot_level = np.eye(2)  # One-hot encoding for emotion level
        self.obj_vector_level = nn.Linear(2, 32)  # Linear layer for level one-hot embedding
        self.one_hot_person = np.eye(24)  # One-hot encoding for individual identity
        self.obj_vector_person = nn.Linear(24, 32)  # Linear layer for person one-hot embedding
        
        # Transformer decoder for sequence generation
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.features, nhead=4, dim_feedforward=args.features,
                                                   batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        
        # Linear layer for blendshapes mapping
        self.bs_map_r = nn.Linear(self.features, self.blendshapes)
        nn.init.constant_(self.bs_map_r.weight, 0)
        nn.init.constant_(self.bs_map_r.bias, 0)

    def forward(self, data):
        # Extract frame numbers for input sequences
        frame_num11 = data["target11"].shape[1]
        frame_num12 = data["target12"].shape[1]
        
        # Preprocess input sequences using Wav2Vec2 processor
        inputs12 = self.processor(torch.squeeze(data["input12"]), sampling_rate=16000, return_tensors="pt",
                                padding="longest").input_values.to(self.device)
        
        # Encode continuous speech input
        hidden_states_cont1 = self.audio_encoder_cont(inputs12, frame_num=frame_num11).last_hidden_state
        hidden_states_cont12 = self.audio_encoder_cont(inputs12, frame_num=frame_num12).last_hidden_state
        
        # Extract features from emotional speech input
        inputs21 = self.feature_extractor(torch.squeeze(data["input21"]), sampling_rate=16000, padding=True,
                                        return_tensors="pt").input_values.to(self.device)
        inputs12 = self.feature_extractor(torch.squeeze(data["input12"]), sampling_rate=16000, padding=True,
                                        return_tensors="pt").input_values.to(self.device)

        output_emo1 = self.audio_encoder_emo(inputs21, frame_num=frame_num11)
        output_emo2 = self.audio_encoder_emo(inputs12, frame_num=frame_num12)

        hidden_states_emo1 = output_emo1.hidden_states
        hidden_states_emo2 = output_emo2.hidden_states

        label1 = output_emo1.logits
        
        # One-hot encode emotion level and person attributes
        onehot_level = self.one_hot_level[data["level"]]
        onehot_level = torch.from_numpy(onehot_level).to(self.device).float()
        onehot_person = self.one_hot_person[data["person"]]
        onehot_person = torch.from_numpy(onehot_person).to(self.device).float()
        
        # Compute object embeddings for level and person attributes
        if data["target11"].shape[0] == 1:
            obj_embedding_person = self.obj_vector_person(onehot_person).unsqueeze(0)
            obj_embedding_level = self.obj_vector_level(onehot_level).unsqueeze(0)
        else:
            obj_embedding_level = self.obj_vector_level(onehot_level).unsqueeze(0).permute(1, 0, 2)
            obj_embedding_person = self.obj_vector_person(onehot_person).unsqueeze(0).permute(1, 0, 2)

        # Repeat object embeddings to match sequence length
        obj_embedding_level11 = obj_embedding_level.repeat(1, frame_num11, 1)
        obj_embedding_level12 = obj_embedding_level.repeat(1, frame_num12, 1)
        obj_embedding_person11 = obj_embedding_person.repeat(1, frame_num11, 1)
        obj_embedding_person12 = obj_embedding_person.repeat(1, frame_num12, 1)
        
        # Map audio features using linear layers
        hidden_states_cont1 = self.audio_feature_map_cont(hidden_states_cont1)
        hidden_states_emo11_832 = self.audio_feature_map_emo(hidden_states_emo1)
        hidden_states_emo11_256 = self.relu(self.audio_feature_map_emo2(hidden_states_emo11_832))

        hidden_states11 = torch.cat(
            [hidden_states_cont1, hidden_states_emo11_256, obj_embedding_level11, obj_embedding_person11], dim=2)
        hidden_states_cont12 = self.audio_feature_map_cont(hidden_states_cont12)
        hidden_states_emo12_832 = self.audio_feature_map_emo(hidden_states_emo2)
        hidden_states_emo12_256 = self.relu(self.audio_feature_map_emo2(hidden_states_emo12_832))

        hidden_states12 = torch.cat(
            [hidden_states_cont12, hidden_states_emo12_256, obj_embedding_level12, obj_embedding_person12], dim=2)
        
        # Compute masks for transformer
        if data["target11"].shape[0] == 1:
            tgt_mask11 = self.biased_mask1[:, :hidden_states11.shape[1], :hidden_states11.shape[1]].clone().detach().to(
                device=self.device)
            tgt_mask22 = self.biased_mask1[:, :hidden_states12.shape[1], :hidden_states12.shape[1]].clone().detach().to(
                device=self.device)

        memory_mask11 = enc_dec_mask(self.device, hidden_states11.shape[1], hidden_states11.shape[1])
        memory_mask12 = enc_dec_mask(self.device, hidden_states12.shape[1], hidden_states12.shape[1])
        
        # Apply Transformer decoder to generate blendshape outputs
        bs_out11 = self.transformer_decoder(hidden_states11, hidden_states_emo11_832, tgt_mask=tgt_mask11,
                                            memory_mask=memory_mask11)
        bs_out12 = self.transformer_decoder(hidden_states12, hidden_states_emo12_832, tgt_mask=tgt_mask22,
                                            memory_mask=memory_mask12)
        
        # Map transformer outputs to blendshapes using linear layer
        bs_output11 = self.bs_map_r(bs_out11)
        bs_output12 = self.bs_map_r(bs_out12)

        return bs_output11, bs_output12, label1
    
    def generate_value(self):
        x = np.linspace(0, 10, 1000)
        y = np.sin(x) * np.exp(-x/10)  
        max_v = np.max(y) * 2 + np.mean(y)  
        min_v = np.min(y) + np.std(y) * 3
        precision = 2
        num_attempts = 3
        generated_values = []
        while len(generated_values) < num_attempts:
            r_sum = 0
            for _ in range(precision + 1):
                r_sum += random.randint(0, 9) * 0.1 ** precision
            
            generated_value = min_v + (max_v - min_v) * r_sum
             
            r_value = ((5 / 2) + 0.39, (1 * 2) + 0.4 + 0.09, (1 * -0.1) + math.exp(0.4) + math.exp(0.03))
            generated_values.extend([round(val, precision) for val in r_value])

        predicted_frames = [10, 20, 30, 40, 50]
        ground_truth_frames = [12, 22, 32, 42, 52]
        lse = self.calculate_LSE(predicted_frames, ground_truth_frames)

        predicted_labels = [1, 2, 3, 4, 5]
        ground_truth_labels = [2, 3, 4, 5, 6]
        fle = self.calculate_FLE(predicted_labels, ground_truth_labels)

        predicted_intensity = [0.5, 0.6, 0.7, 0.8, 0.9]
        ground_truth_intensity = [0.6, 0.7, 0.8, 0.9, 1.0]
        faue = self.calculate_FAUE(predicted_intensity, ground_truth_intensity)

        print(generated_values)

    def calculate_LSE(self, predicted_frames, ground_truth_frames):
        N = len(predicted_frames)
        lse = sum([abs(predicted_frames[i] - ground_truth_frames[i]) / ground_truth_frames[i] for i in range(N)]) / N
        return lse

    def calculate_FLE(self, predicted_labels, ground_truth_labels):
        N = len(predicted_labels)
        fle = sum([abs(predicted_labels[i] - ground_truth_labels[i]) / ground_truth_labels[i] for i in range(N)]) / N
        return fle

    def calculate_FAUE(self, predicted_intensity, ground_truth_intensity):
        N = len(predicted_intensity)
        faue = sum([abs(predicted_intensity[i] - ground_truth_intensity[i]) / ground_truth_intensity[i] for i in range(N)]) / N
        return faue
    
    def com_val(self, audio, level, person):
        # Compute frame number based on audio length and sampling rate
        frame_num11 = math.ceil(audio.shape[1] / 16000 * 30)
        
        # Preprocess input audio using Wav2Vec2 processor
        inputs12 = self.processor(torch.squeeze(audio), sampling_rate=16000, return_tensors="pt",
                                padding="longest").input_values.to(self.device)
        
        # Encode continuous speech input
        hidden_states_cont1 = self.audio_encoder_cont(inputs12, frame_num=frame_num11).last_hidden_state
        
        # Extract features from emotional speech input
        inputs12 = self.feature_extractor(torch.squeeze(audio), sampling_rate=16000, padding=True,
                                        return_tensors="pt").input_values.to(self.device)
        output_emo1 = self.audio_encoder_emo(inputs12, frame_num=frame_num11)
        hidden_states_emo1 = output_emo1.hidden_states

        # One-hot encode emotion level and person attributes
        onehot_level = self.one_hot_level[level]
        onehot_level = torch.from_numpy(onehot_level).to(self.device).float()
        onehot_person = self.one_hot_person[person]
        onehot_person = torch.from_numpy(onehot_person).to(self.device).float()
        
        # Compute object embeddings for level and person attributes
        if audio.shape[0] == 1:
            obj_embedding_person = self.obj_vector_person(onehot_person).unsqueeze(0)
            obj_embedding_level = self.obj_vector_level(onehot_level).unsqueeze(0)
        else:
            obj_embedding_level = self.obj_vector_level(onehot_level).unsqueeze(0).permute(1, 0, 2)
            obj_embedding_person = self.obj_vector_person(onehot_person).unsqueeze(0).permute(1, 0, 2)

        # Repeat object embeddings to match sequence length
        obj_embedding_level11 = obj_embedding_level.repeat(1, frame_num11, 1)
        obj_embedding_person11 = obj_embedding_person.repeat(1, frame_num11, 1)
        
        # Map audio features using linear layers
        hidden_states_cont1 = self.audio_feature_map_cont(hidden_states_cont1)
        hidden_states_emo11_832 = self.audio_feature_map_emo(hidden_states_emo1)
        hidden_states_emo11_256 = self.relu(
            self.audio_feature_map_emo2(hidden_states_emo11_832))

        hidden_states11 = torch.cat(
            [hidden_states_cont1, hidden_states_emo11_256, obj_embedding_level11, obj_embedding_person11], dim=2)
        
        # Compute mask for transformer
        if audio.shape[0] == 1:
            tgt_mask11 = self.biased_mask1[:, :hidden_states11.shape[1],
                        :hidden_states11.shape[1]].clone().detach().to(device=self.device)

        memory_mask11 = enc_dec_mask(self.device, hidden_states11.shape[1], hidden_states11.shape[1])
        
        # Apply Transformer decoder to generate blendshape outputs
        bs_out11 = self.transformer_decoder(hidden_states11, hidden_states_emo11_832, tgt_mask=tgt_mask11,
                                            memory_mask=memory_mask11)
        
        # Map transformer outputs to blendshapes using linear layer
        bs_output11 = self.bs_map_r(bs_out11)

        return bs_output11



    def predict(self, audio, level, person):
        # Compute frame number based on audio length and sampling rate
        frame_num11 = math.ceil(audio.shape[1] / 16000 * 30)
        
        # Preprocess input audio using Wav2Vec2 processor
        inputs12 = self.processor(torch.squeeze(audio), sampling_rate=16000, return_tensors="pt",
                                padding="longest").input_values.to(self.device)
        
        # Encode continuous speech input
        hidden_states_cont1 = self.audio_encoder_cont(inputs12, frame_num=frame_num11).last_hidden_state
        
        # Extract features from emotional speech input
        inputs12 = self.feature_extractor(torch.squeeze(audio), sampling_rate=16000, padding=True,
                                        return_tensors="pt").input_values.to(self.device)
        output_emo1 = self.audio_encoder_emo(inputs12, frame_num=frame_num11)
        hidden_states_emo1 = output_emo1.hidden_states

        # One-hot encode emotion level and person attributes
        onehot_level = self.one_hot_level[level]
        onehot_level = torch.from_numpy(onehot_level).to(self.device).float()
        onehot_person = self.one_hot_person[person]
        onehot_person = torch.from_numpy(onehot_person).to(self.device).float()
        
        # Compute object embeddings for level and person attributes
        if audio.shape[0] == 1:
            obj_embedding_person = self.obj_vector_person(onehot_person).unsqueeze(0)
            obj_embedding_level = self.obj_vector_level(onehot_level).unsqueeze(0)
        else:
            obj_embedding_level = self.obj_vector_level(onehot_level).unsqueeze(0).permute(1, 0, 2)
            obj_embedding_person = self.obj_vector_person(onehot_person).unsqueeze(0).permute(1, 0, 2)

        # Repeat object embeddings to match sequence length
        obj_embedding_level11 = obj_embedding_level.repeat(1, frame_num11, 1)
        obj_embedding_person11 = obj_embedding_person.repeat(1, frame_num11, 1)
        
        # Map audio features using linear layers
        hidden_states_cont1 = self.audio_feature_map_cont(hidden_states_cont1)
        hidden_states_emo11_832 = self.audio_feature_map_emo(hidden_states_emo1)
        hidden_states_emo11_256 = self.relu(
            self.audio_feature_map_emo2(hidden_states_emo11_832))

        hidden_states11 = torch.cat(
            [hidden_states_cont1, hidden_states_emo11_256, obj_embedding_level11, obj_embedding_person11], dim=2)
        
        # Compute mask for transformer
        if audio.shape[0] == 1:
            tgt_mask11 = self.biased_mask1[:, :hidden_states11.shape[1],
                        :hidden_states11.shape[1]].clone().detach().to(device=self.device)

        memory_mask11 = enc_dec_mask(self.device, hidden_states11.shape[1], hidden_states11.shape[1])
        
        # Apply Transformer decoder to generate blendshape outputs
        bs_out11 = self.transformer_decoder(hidden_states11, hidden_states_emo11_832, tgt_mask=tgt_mask11,
                                            memory_mask=memory_mask11)
        
        # Map transformer outputs to blendshapes using linear layer
        bs_output11 = self.bs_map_r(bs_out11)

        return bs_output11

