import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# Configuration class
_CONFIG_FOR_DOC = "Wav2Vec2Config"
_HIDDEN_STATES_START_POSITION = 2

# Linear interpolation layer
def linear_interpolation(features, input_fps, output_fps, output_len=None):
    """
    Perform linear interpolation on input features.

    Args:
        features (torch.Tensor): Input features.
        input_fps (int): Input frames per second.
        output_fps (int): Output frames per second.
        output_len (int, optional): Output sequence length. Defaults to None.

    Returns:
        torch.Tensor: Interpolated features.
    """
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    output_len = output_len or int(seq_len * output_fps)
    output_features = F.interpolate(features, size=output_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


class Wav2Vec2Model(Wav2Vec2Model):
    def __init__(self, config):
        super().__init__(config)
        self.lm_head = nn.Linear(1024, 32)

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            frame_num=None
    ):
        """
        Forward pass of the Wav2Vec2 model.

        Args:
            input_values (torch.Tensor): Input audio features.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            output_attentions (bool, optional): Whether to output attentions. Defaults to None.
            output_hidden_states (bool, optional): Whether to output hidden states. Defaults to None.
            return_dict (bool, optional): Whether to return a dictionary. Defaults to None.
            frame_num (int, optional): Number of frames. Defaults to None.

        Returns:
            BaseModelOutput: Output of the model.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.feature_extractor(input_values)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = linear_interpolation(hidden_states, 50, 30, output_len=frame_num)
        attention_mask = None  # No attention mask for now

        hidden_states = self.feature_projection(hidden_states)[0]

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@dataclass
class SpeechClassifierOutput(ModelOutput):
    """
    Output class for the speech classifier model.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(F.tanh(self.dense(features)))
        x = self.dropout(F.tanh(self.out_proj(x)))
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        """
        Initializes the Wav2Vec2ForSpeechClassification model.
        
        Args:
            config (Wav2Vec2Config): Configuration object specifying model parameters.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        # Wav2Vec2 model
        self.wav2vec2 = Wav2Vec2Model(config)
        
        # Classification head
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        """
        Freeze feature extractor of the Wav2Vec2 model.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        """
        Applies merging strategy to combine hidden states.
        
        Args:
            hidden_states (torch.Tensor): Hidden states to be merged.
            mode (str): Merging mode. Can be "mean", "sum", or "max".
            
        Returns:
            torch.Tensor: Merged hidden states.
        """
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise ValueError("Invalid pooling mode. Please choose one of ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
            frame_num=None,
    ):
        """
        Forward pass of the model.
        
        Args:
            input_values (torch.Tensor): Input audio values.
            attention_mask (torch.Tensor, optional): Attention mask.
            output_attentions (bool, optional): Whether to return attentions.
            output_hidden_states (bool, optional): Whether to return hidden states.
            return_dict (bool, optional): Whether to return a dictionary of outputs.
            labels (torch.Tensor, optional): Ground truth labels.
            frame_num (int, optional): Number of frames.
            
        Returns:
            SpeechClassifierOutput: Output including loss, logits, hidden states, and attentions.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get outputs from Wav2Vec2 model
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        
        # Linear interpolation for adjusting frame number
        hidden_states1 = linear_interpolation(hidden_states, 50, 30, output_len=frame_num)
        
        # Merge hidden states
        hidden_states = self.merged_strategy(hidden_states1, mode=self.pooling_mode)
        
        # Pass merged hidden states through classifier
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            # Determine loss function based on problem type
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states1,
            attentions=outputs.attentions,
        )


