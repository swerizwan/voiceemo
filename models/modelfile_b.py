import torch
import torch.nn as nn
import numpy as np
import math
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from models.wav2vec import Wav2Vec2Model, Wav2Vec2ForSpeechClassification
from utils.utilsfile import init_biased_mask, enc_dec_mask

# Setting environment variables to use offline mode for Hugging Face datasets and transformers
HF_DATASETS_OFFLINE = 1
TRANSFORMERS_OFFLINE = 1

# Paths to pre-trained models
path = 'jonatasgrosman/wav2vec2-large-xlsr-53-english'
path2 = 'jonatasgrosman/wav2vec-english-speech-emotion-recognition'

# Definition of the main model class
class verhm(nn.Module):
    def __init__(self, args):
        super(verhm, self).__init__()
        
        # Model parameters
        self.f_dim = args.f_dim
        self.b_dim = args.b_dim
        self.device = args.device
        self.batch_size = args.batch_size
        self.m_seq_l = args.m_seq_l
        
        # Load Wav2Vec2 models and processors
        self.voice_encoder_cont = Wav2Vec2Model.from_pretrained(path, local_files_only=True)
        self.processor = Wav2Vec2Processor.from_pretrained(path, local_files_only=True)
        self.voice_encoder_cont.feature_extractor._freeze_parameters()
        self.voice_encoder_emo = Wav2Vec2ForSpeechClassification.from_pretrained(path2, local_files_only=True)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(path2, local_files_only=True)
        self.voice_encoder_emo.wav2vec2.feature_extractor._freeze_parameters()
        
        # Linear layers for feature mapping
        self.voice_feature_map_cont = nn.Linear(1024, 512)
        self.voice_feature_map_emo = nn.Linear(1024, 832)
        self.voice_feature_map_emo2 = nn.Linear(832, 256)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Initialize biased mask for training
        self.biased_mask1 = init_biased_mask(n_head=4, m_seq_l=args.m_seq_l, times=args.times)
        
        # One-hot encoding for level and person
        self.one_hot_level = np.eye(2)
        self.obj_vector_level = nn.Linear(2, 32)
        self.one_hot_person = np.eye(24)
        self.obj_vector_person = nn.Linear(24, 32)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.f_dim, nhead=4, dim_feedforward=args.f_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        
        # Linear layer for output
        self.bs_map_r = nn.Linear(self.f_dim, self.b_dim)
        nn.init.constant_(self.bs_map_r.weight, 0)
        nn.init.constant_(self.bs_map_r.bias, 0)

    # Forward pass of the model
    def forward(self, data):
        # Extract frame numbers
        frame_num11 = data["target11"].shape[1]
        frame_num12 = data["target12"].shape[1]

        # Process input tensors
        inputs12 = self._process_input(data["input12"], frame_num11, 16000)
        inputs21 = self._process_input(data["input21"], frame_num12, 16000)

        # Encode voice features
        hidden_states_cont1 = self._encode_voice(self.voice_encoder_cont, inputs12, frame_num11)
        hidden_states_cont12 = self._encode_voice(self.voice_encoder_cont, inputs12, frame_num12)
        output_emo1 = self.voice_encoder_emo(inputs21, frame_num11)
        output_emo2 = self.voice_encoder_emo(inputs12, frame_num12)
        hidden_states_emo1 = output_emo1.hidden_states
        hidden_states_emo2 = output_emo2.hidden_states

        # Compute labels and embeddings
        label1 = output_emo1.logits
        onehot_level, onehot_person = self._compute_one_hot(data)
        obj_embedding_level, obj_embedding_person = self._compute_embeddings(onehot_level, onehot_person, frame_num11)

        # Apply linear transformations
        hidden_states_cont1 = self.voice_feature_map_cont(hidden_states_cont1)
        hidden_states_cont12 = self.voice_feature_map_cont(hidden_states_cont12)
        hidden_states_emo11_832 = self.voice_feature_map_emo(hidden_states_emo1)
        hidden_states_emo12_832 = self.voice_feature_map_emo(hidden_states_emo2)
        hidden_states_emo11_256 = self.relu(self.voice_feature_map_emo2(hidden_states_emo11_832))
        hidden_states_emo12_256 = self.relu(self.voice_feature_map_emo2(hidden_states_emo12_832))

        # Concatenate features
        hidden_states11 = self._concatenate_features(hidden_states_cont1, hidden_states_emo11_256, obj_embedding_level, obj_embedding_person)
        hidden_states12 = self._concatenate_features(hidden_states_cont12, hidden_states_emo12_256, obj_embedding_level, obj_embedding_person)

        # Apply masks
        tgt_mask11, tgt_mask12 = self._apply_masks(data, hidden_states11.shape[1], hidden_states12.shape[1])

        # Decode features
        bs_out11 = self._decode_features(hidden_states11, hidden_states_emo11_832, tgt_mask11)
        bs_out12 = self._decode_features(hidden_states12, hidden_states_emo12_832, tgt_mask12)

        # Apply linear transformation
        bs_output11 = self.bs_map_r(bs_out11)
        bs_output12 = self.bs_map_r(bs_out12)

        return bs_output11, bs_output12, label1

    # Method to process input data
    def _process_input(self, input_data, frame_num, sampling_rate):
        return self.processor(torch.squeeze(input_data), sampling_rate=sampling_rate, return_tensors="pt",
                            padding="longest").input_values.to(self.device)

    # Method to encode voice features
    def _encode_voice(self, voice_encoder, inputs, frame_num):
        return voice_encoder(inputs, frame_num=frame_num).last_hidden_state

    # Method to compute one-hot encoding for level and person
    def _compute_one_hot(self, data):
        onehot_level = torch.from_numpy(self.one_hot_level[data["level"]]).to(self.device).float()
        onehot_person = torch.from_numpy(self.one_hot_person[data["person"]]).to(self.device).float()
        return onehot_level, onehot_person

    # Method to compute embeddings
    def _compute_embeddings(self, onehot_level, onehot_person, frame_num):
        obj_embedding_level = self.obj_vector_level(onehot_level).unsqueeze(0)
        obj_embedding_person = self.obj_vector_person(onehot_person).unsqueeze(0)
        if frame_num > 1:
            obj_embedding_level = obj_embedding_level.permute(1, 0, 2)
            obj_embedding_person = obj_embedding_person.permute(1, 0, 2)
       
        return obj_embedding_level.repeat(1, frame_num, 1), obj_embedding_person.repeat(1, frame_num, 1)

    # Method to concatenate features
    def _concatenate_features(self, cont_hidden_states, emo_hidden_states, obj_embedding_level, obj_embedding_person):
        return torch.cat([cont_hidden_states, emo_hidden_states, obj_embedding_level, obj_embedding_person], dim=2)

    # Method to apply masks
    def _apply_masks(self, data, length11, length12):
        if data["target11"].shape[0] == 1:
            tgt_mask11 = self.biased_mask1[:, :length11, :length11].clone().detach().to(self.device)
            tgt_mask12 = self.biased_mask1[:, :length12, :length12].clone().detach().to(self.device)
        else:
            tgt_mask11 = None
            tgt_mask12 = None
        memory_mask11 = enc_dec_mask(self.device, length11, length11)
        memory_mask12 = enc_dec_mask(self.device, length12, length12)
        return tgt_mask11, tgt_mask12, memory_mask11, memory_mask12

    # Method to decode features
    def _decode_features(self, hidden_states, emo_hidden_states, tgt_mask):
        return self.transformer_decoder(hidden_states, emo_hidden_states, tgt_mask=tgt_mask, memory_mask=None)

    # Method to make predictions
    def predict(self, voice, level, person):
        frame_num11 = math.ceil(voice.shape[1] / 16000 * 30)
        inputs12 = self.processor(torch.squeeze(voice), sampling_rate=16000, return_tensors="pt",
                                  padding="longest").input_values.to(self.device)
        hidden_states_cont1 = self.voice_encoder_cont(inputs12, frame_num=frame_num11).last_hidden_state
        inputs12 = self.feature_extractor(torch.squeeze(voice), sampling_rate=16000, padding=True,
                                          return_tensors="pt").input_values.to(self.device)
        output_emo1 = self.voice_encoder_emo(inputs12, frame_num=frame_num11)
        hidden_states_emo1 = output_emo1.hidden_states

        onehot_level = self.one_hot_level[level]
        onehot_level = torch.from_numpy(onehot_level).to(self.device).float()
        onehot_person = self.one_hot_person[person]
        onehot_person = torch.from_numpy(onehot_person).to(self.device).float()
        if voice.shape[0] == 1:
            obj_embedding_person = self.obj_vector_person(onehot_person).unsqueeze(0)
            obj_embedding_level = self.obj_vector_level(onehot_level).unsqueeze(0)
        else:
            obj_embedding_level = self.obj_vector_level(onehot_level).unsqueeze(0).permute(1, 0, 2)
            obj_embedding_person = self.obj_vector_person(onehot_person).unsqueeze(0).permute(1, 0, 2)

        obj_embedding_level11 = obj_embedding_level.repeat(1, frame_num11, 1)
        obj_embedding_person11 = obj_embedding_person.repeat(1, frame_num11, 1)
        hidden_states_cont1 = self.voice_feature_map_cont(hidden_states_cont1)
        hidden_states_emo11_832 = self.voice_feature_map_emo(hidden_states_emo1)
        hidden_states_emo11_256 = self.relu(
            self.voice_feature_map_emo2(hidden_states_emo11_832))

        hidden_states11 = torch.cat(
            [hidden_states_cont1, hidden_states_emo11_256, obj_embedding_level11, obj_embedding_person11], dim=2)
        if voice.shape[0] == 1:
            tgt_mask11 = self.biased_mask1[:, :hidden_states11.shape[1],
                         :hidden_states11.shape[1]].clone().detach().to(device=self.device)

        memory_mask11 = enc_dec_mask(self.device, hidden_states11.shape[1], hidden_states11.shape[1])
        bs_out11 = self.transformer_decoder(hidden_states11, hidden_states_emo11_832, tgt_mask=tgt_mask11,
                                            memory_mask=memory_mask11)
        bs_output11 = self.bs_map_r(bs_out11)

        return bs_output11
