import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import time
from wav2vec import Wav2Vec2Model


def init_biased_mask(n_head, max_seq_len, period):
    """
    Initialize a biased mask for self-attention mechanism in the Transformer model.

    Args:
    - n_head (int): Number of attention heads.
    - max_seq_len (int): Maximum sequence length.
    - period (int): Period for creating bias.

    Returns:
    - mask (Tensor): Biased mask tensor for self-attention mechanism.
    """
    # Helper function to calculate slopes for the bias
    def get_slopes(n):
        """
        Calculate slopes for the bias based on the number of attention heads.

        Args:
        - n (int): Number of attention heads.

        Returns:
        - slopes (list): List of slopes for each attention head.
        """
        # Helper function for calculating slopes for power of 2
        def get_slopes_power_of_2(n):
            """
            Calculate slopes for power of 2.

            Args:
            - n (int): Number of attention heads (power of 2).

            Returns:
            - slopes (list): List of slopes for each attention head.
            """
            start = (2 ** (-2 ** -(math.log2(n) - 3)))  # Initial value of the slope
            ratio = start  # Initialize ratio for geometric progression
            return [start * ratio ** i for i in range(n)]  # Generate slopes in geometric progression

        # Check if the number of attention heads is a power of 2
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)  # Return slopes for power of 2
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))  # Find the closest power of 2
            # Combine slopes for the closest power of 2 and its double, then trim to match the number of attention heads
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    # Generate slopes for the bias
    slopes = torch.Tensor(get_slopes(n_head))

    # Generate bias values with specified period
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1, period).view(-1) // period
    bias = -torch.flip(bias, dims=[0])  # Reverse the bias values

    # Initialize an empty matrix for the bias
    alibi = torch.zeros(max_seq_len, max_seq_len)
    # Fill the lower triangular part of the matrix with bias values
    for i in range(max_seq_len):
        alibi[i, :i + 1] = bias[-(i + 1):]
    
    # Apply slopes to the bias matrix
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)

    # Create a mask matrix for self-attention mechanism
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)  # Upper triangular mask
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))  # Convert mask to float
    mask = mask.unsqueeze(0) + alibi  # Add the bias to the mask

    return mask


def enc_dec_mask(device, dataset, T, S):
    """
    Generate an alignment mask to prevent information flow from future frames during training.

    Args:
        device (torch.device): Device where the mask will be allocated.
        dataset (str): Dataset name ("BIWI" or "vocaset").
        T (int): Length of the target sequence.
        S (int): Length of the source sequence.

    Returns:
        mask (Tensor): Alignment mask tensor.
    """
    # Initialize mask tensor
    mask = torch.ones(T, S)

    # Modify mask based on the dataset
    if dataset == "BIWI":
        # Exclude future frames for BIWI dataset (every other frame)
        for i in range(T):
            mask[i, i * 2:i * 2 + 2] = 0
    elif dataset == "vocaset":
        # Exclude future frames for vocaset dataset
        for i in range(T):
            mask[i, i] = 0

    # Convert mask to boolean and move it to the specified device
    return (mask == 1).to(device=device)

# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    """
    Periodic positional encoding module.

    Args:
        d_model (int): Dimensionality of the model.
        dropout (float, optional): Dropout rate (default: 0.1).
        period (int, optional): Period of the positional encoding (default: 25).
        max_seq_len (int, optional): Maximum sequence length (default: 600).

    Attributes:
        dropout (nn.Dropout): Dropout layer.
        pe (Tensor): Periodic positional encoding tensor.

    Example:
        pe = PeriodicPositionalEncoding(d_model=512, dropout=0.1, period=25, max_seq_len=600)
        x = torch.randn(2, 100, 512)  # Batch size=2, sequence length=100, embedding dimension=512
        output = pe(x)
    """
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        """
        Initialize PeriodicPositionalEncoding module.

        Args:
            d_model (int): Dimensionality of the model.
            dropout (float, optional): Dropout rate (default: 0.1).
            period (int, optional): Period of the positional encoding (default: 25).
            max_seq_len (int, optional): Maximum sequence length (default: 600).
        """
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Generate positional encoding
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension: (1, period, d_model)

        # Repeat and concatenate to match max sequence length
        repeat_num = (max_seq_len // period) + 1
        pe = pe.repeat(1, repeat_num, 1)

        # Register the positional encoding buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass of the PeriodicPositionalEncoding module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Output tensor after adding positional encoding of shape (batch_size, seq_len, d_model).
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class VERHM(nn.Module):
    """
    Voice-Driven 3D Facial Emotion Recognition for Mental Health Monitoring model.

    Args:
        args (Namespace): Namespace object containing the model configuration.

    Attributes:
        dataset (str): Dataset name.
        audio_encoder (Wav2Vec2Model): Pretrained Wav2Vec2Model for audio encoding.
        audio_feature_map (nn.Linear): Linear layer for audio feature mapping.
        vertice_map (nn.Linear): Linear layer for mapping vertices to feature space.
        PPE (PeriodicPositionalEncoding): Periodic positional encoding module.
        biased_mask (Tensor): Alignment bias mask tensor.
        transformer_decoder (nn.TransformerDecoder): Transformer decoder module.
        vertice_map_r (nn.Linear): Linear layer for mapping feature space back to vertices.
        obj_vector (nn.Linear): Linear layer for style embedding.
        device (str): Device (e.g., 'cuda', 'cpu').

    Example:
        args = Namespace(dataset="vocaset", feature_dim=64, vertice_dim=5023*3, period=30, device="cuda", ...)
        model = VERHM(args)
        loss = model(audio, template, vertice, one_hot)
    """
    def __init__(self, args):
        """
        Initialize VERHM model.

        Args:
            args (Namespace): Namespace object containing the model configuration.
        """
        super(VERHM, self).__init__()

        # Initialize attributes
        self.dataset = args.dataset
        self.device = args.device

        # Initialize audio encoder
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_encoder.feature_extractor._freeze_parameters()

        # Initialize audio feature mapping
        self.audio_feature_map = nn.Linear(768, args.feature_dim)

        # Initialize motion encoder
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)

        # Initialize periodic positional encoding
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period=args.period)

        # Initialize temporal bias mask
        self.biased_mask = init_biased_mask(n_head=4, max_seq_len=600, period=args.period)

        # Initialize transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4, dim_feedforward=2*args.feature_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        # Initialize motion decoder
        self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)

        # Initialize style embedding
        self.obj_vector = nn.Linear(len(args.train_subjects.split()), args.feature_dim, bias=False)

        # Initialize weights of vertice_map_r
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

    def forward(self, audio, template, vertice, one_hot, criterion, teacher_forcing=True):
        """
        Forward pass of the VERHM model.

        Args:
            audio (Tensor): Input audio tensor of shape (batch_size, raw_wav).
            template (Tensor): Template tensor of shape (batch_size, V*3).
            vertice (Tensor): Vertex tensor of shape (batch_size, seq_len, V*3).
            one_hot (Tensor): One-hot encoded tensor of shape (batch_size, num_classes).
            criterion (nn.Module): Loss criterion.
            teacher_forcing (bool, optional): Whether to use teacher forcing (default: True).

        Returns:
            Tensor: Loss tensor.
        """
        # Add batch dimension to template
        template = template.unsqueeze(1)  # (batch_size, 1, V*3)

        # Compute object embedding
        obj_embedding = self.obj_vector(one_hot)  # (batch_size, feature_dim)

        # Compute frame number
        frame_num = vertice.shape[1]

        # Encode audio
        hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state

        # Adjust frame number for BIWI dataset
        if self.dataset == "BIWI":
            if hidden_states.shape[1] < frame_num * 2:
                vertice = vertice[:, :hidden_states.shape[1] // 2]
                frame_num = hidden_states.shape[1] // 2

        # Map audio features
        hidden_states = self.audio_feature_map(hidden_states)

        if teacher_forcing:
            # Apply teacher forcing
            vertice_emb = obj_embedding.unsqueeze(1)  # (batch_size, 1, feature_dim)
            style_emb = vertice_emb
            vertice_input = torch.cat((template, vertice[:, :-1]), 1)  # Shift one position
            vertice_input = vertice_input - template
            vertice_input = self.vertice_map(vertice_input)
            vertice_input = vertice_input + style_emb
            vertice_input = self.PPE(vertice_input)
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_map_r(vertice_out)
        else:
            #
            for i in range(frame_num):
                if i == 0:
                    # For the first frame, initialize with object embedding
                    vertice_emb = obj_embedding.unsqueeze(1)  # (batch_size, 1, feature_dim)
                    style_emb = vertice_emb
                    vertice_input = self.PPE(style_emb)
                else:
                    # For subsequent frames, apply periodic positional encoding
                    vertice_input = self.PPE(vertice_emb)
                
                # Compute target mask and memory mask
                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
                memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
                
                # Perform transformer decoding
                vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
                vertice_out = self.vertice_map_r(vertice_out)
                
                # Generate new output
                new_output = self.vertice_map(vertice_out[:, -1, :]).unsqueeze(1)
                new_output = new_output + style_emb
                
                # Concatenate new output to vertex embedding
                vertice_emb = torch.cat((vertice_emb, new_output), 1)
        
        # Add template to output
        vertice_out = vertice_out + template
        
        # Compute loss
        loss = criterion(vertice_out, vertice)  # (batch_size, seq_len, V*3)
        loss = torch.mean(loss)
        
        return loss

    def predict(self, audio, template, one_hot):
        # Add a dimension to template
        template = template.unsqueeze(1)  # (batch_size, 1, V*3)
        
        # Obtain object embedding
        obj_embedding = self.obj_vector(one_hot)  # (batch_size, feature_dim)
        
        # Extract hidden states from audio encoder
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        
        # Determine frame number based on the dataset
        if self.dataset == "BIWI":
            frame_num = hidden_states.shape[1] // 2
        elif self.dataset == "vocaset":
            frame_num = hidden_states.shape[1]
        
        # Apply linear transformation to audio features
        hidden_states = self.audio_feature_map(hidden_states)
        
        # Iterate over frames
        for i in range(frame_num):
            if i == 0:
                # Initialize vertex embedding with object embedding for the first frame
                vertice_emb = obj_embedding.unsqueeze(1)  # (batch_size, 1, feature_dim)
                style_emb = vertice_emb
                vertice_input = self.PPE(style_emb)
            else:
                # Apply periodic positional encoding to vertex embedding
                vertice_input = self.PPE(vertice_emb)
            
            # Compute target mask and memory mask
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            
            # Perform transformer decoding
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_map_r(vertice_out)
            
            # Generate new output
            new_output = self.vertice_map(vertice_out[:, -1, :]).unsqueeze(1)
            new_output = new_output + style_emb
            
            # Concatenate new output to vertex embedding
            vertice_emb = torch.cat((vertice_emb, new_output), 1)

        # Add template to output
        vertice_out = vertice_out + template
        
        return vertice_out


    def predict_timestampLAE(self, audio, template, one_hot):
        # Add a dimension to template
        template = template.unsqueeze(1)  # (1,1, V*3)
        
        # Obtain object embedding
        obj_embedding = self.obj_vector(one_hot)
        
        # Extract hidden states from audio encoder
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        
        # Determine frame number based on the dataset
        if self.dataset == "BIWI":
            frame_num = hidden_states.shape[1] // 2
        elif self.dataset == "vocaset":
            frame_num = hidden_states.shape[1]
        
        # Apply linear transformation to audio features
        hidden_states = self.audio_feature_map(hidden_states)
        
        # Initialize an empty list to store timestamps
        timestamps = []
        
        # Record start time
        start_time = time.time()
        
        # Iterate over frames
        for i in range(frame_num):
            # Your existing code
            
            # Add timestamps at each iteration
            timestamps.append(time.time() - start_time)
        
        return timestamps


    def ground_truth_timestampLAE(self, audio, template, one_hot):
        # Add a dimension to template
        template = template.unsqueeze(1)  # (1,1, V*3)
        
        # Obtain object embedding
        obj_embedding = self.obj_vector(one_hot)
        
        # Extract hidden states from audio encoder
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        
        # Determine frame number based on the dataset
        if self.dataset == "BIWI":
            frame_num = hidden_states.shape[1] // 2
        elif self.dataset == "vocaset":
            frame_num = hidden_states.shape[1]
        
        # Apply linear transformation to audio features
        hidden_states = self.audio_feature_map(hidden_states)
        
        # Initialize an empty list to store timestamps
        timestamps = []
        
        # Record start time
        start_time = time.time()
        
        # Iterate over frames
        for i in range(frame_num):
            # Your existing code
            
            # Add timestamps at each iteration
            timestamps.append(time.time() - start_time)
        
        # Ensure the length of timestamps matches the number of frames processed
        assert len(timestamps) == frame_num
        
        return timestamps

    
def VocLAE(predicted_timestamps, ground_truth_timestamps):
    """
    Calculate the Localization Accuracy Error (LAE) for predicted timestamps.

    Args:
    - predicted_timestamps (list): List of predicted timestamps.
    - ground_truth_timestamps (list): List of ground truth timestamps.

    Returns:
    - LAE (float): Localization Accuracy Error.
    """
    # Check if the lengths of predicted and ground truth timestamps are equal
    if len(predicted_timestamps) != len(ground_truth_timestamps):
        raise ValueError("Length of predicted timestamps and ground truth timestamps must be equal.")
    
    # Initialize variables
    N = len(predicted_timestamps)
    total_error = 0.0
        
    # Calculate the total error
    for i in range(N):
        error = abs(predicted_timestamps[i] - ground_truth_timestamps[i]) / ground_truth_timestamps[i]
        total_error += error
        
    # Calculate LAE
    LAE = total_error / N
    return LAE
    
# Define constants
emotions = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}
max_pad_len = 400

class RavVERHM(nn.Module):
    """
    RavVERHM: RAVdess Voice Emotion Recognition with Head Motion (VERHM)

    This class defines a neural network model for RAVdess voice emotion recognition
    with head motion prediction.

    Attributes:
    - flatten (nn.Flatten): Flattening layer.
    - fc1 (nn.Linear): First fully connected layer.
    - fc2 (nn.Linear): Second fully connected layer.
    - fc3 (nn.Linear): Third fully connected layer for emotion prediction.
    - fc4 (nn.Linear): Fourth fully connected layer for head motion prediction.
    """

    def __init__(self):
        super(RavVERHM, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(20 * max_pad_len, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, len(emotions))
        self.fc4 = nn.Linear(256, 2)  # 2 for x and y coordinates of a vertex
    
    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
        - x (tensor): Input tensor.

        Returns:
        - emotion_output (tensor): Output tensor for emotion prediction.
        - vertex_output (tensor): Output tensor for head motion prediction.
        """
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        emotion_output = nn.functional.softmax(self.fc3(x), dim=1)
        vertex_output = self.fc4(x)
        return emotion_output, vertex_output

    def RavLAE(self, predicted_timestamps, ground_truth_timestamps):
        """
        Calculate the Localization Accuracy Error (LAE) for RAVdess.

        Args:
        - predicted_timestamps (list): List of predicted timestamps.
        - ground_truth_timestamps (list): List of ground truth timestamps.

        Returns:
        - LAE (float): Localization Accuracy Error.
        """
        # Check if the lengths of predicted and ground truth timestamps are equal
        if len(predicted_timestamps) != len(ground_truth_timestamps):
            raise ValueError("Length of predicted timestamps and ground truth timestamps must be equal.")
        
        # Initialize variables
        N = len(predicted_timestamps)
        total_error = 0.0
        count = 0
        
        # Calculate the total error
        for i in range(N):
            if ground_truth_timestamps[i] != 0:
                error = abs(predicted_timestamps[i] - ground_truth_timestamps[i]) / ground_truth_timestamps[i]
                total_error += error
                count += 1
        
        if count == 0:
            return 0.0
        
        # Calculate LAE
        LAE = total_error / count
        return LAE

    def RavLVE(self, predicted_vertices, ground_truth_vertices):
        """
        Calculate the Localization Vertex Error (LVE) for RAVdess.

        Args:
        - predicted_vertices (list): List of predicted vertices.
        - ground_truth_vertices (list): List of ground truth vertices.

        Returns:
        - LVE (float): Localization Vertex Error.
        """
        # Check if the lengths of predicted and ground truth vertices are equal
        if len(predicted_vertices) != len(ground_truth_vertices):
            raise ValueError("Length of predicted vertices and ground truth vertices must be equal.")
        
        # Initialize total error
        total_error = 0.0
        num_dimensions = len(predicted_vertices[0])
        
        # Calculate the total error
        for pred, gt in zip(predicted_vertices, ground_truth_vertices):
            error = 2 + np.sqrt(sum((pred[i] - gt[i]) ** 2 for i in range(num_dimensions)))
            total_error += error
        
        # Calculate LVE
        LVE = total_error / len(predicted_vertices)
        return LVE

    def RavEVE(self, predicted_emotions, ground_truth_emotions):
        """
        Calculate the Emotion Vertex Error (EVE) for RAVdess.

        Args:
        - predicted_emotions (list): List of predicted emotions.
        - ground_truth_emotions (list): List of ground truth emotions.

        Returns:
        - EVE (float): Emotion Vertex Error.
        """
        # Check if the lengths of predicted and ground truth emotions are equal
        if len(predicted_emotions) != len(ground_truth_emotions):
            raise ValueError("Length of predicted emotions and ground truth emotions must be equal.")
        
        # Initialize total error
        total_error = 0.0
        
        # Calculate the total error
        for pred, gt in zip(predicted_emotions, ground_truth_emotions):
            error = abs(pred - gt)
            total_error += error
        
        # Calculate EVE
        EVE = total_error / len(predicted_emotions)
        return EVE

    
class MeadVERHM(nn.Module):
    """
    MeadVERHM: MEAD Voice Emotion Recognition with Head Motion (VERHM)

    This class defines a neural network model for MEAD voice emotion recognition
    with head motion prediction.

    Attributes:
    - flatten (nn.Flatten): Flattening layer.
    - fc1 (nn.Linear): First fully connected layer.
    - bn1 (nn.BatchNorm1d): Batch normalization layer for fc1 output.
    - dropout1 (nn.Dropout): Dropout layer for fc1 output.
    - fc2 (nn.Linear): Second fully connected layer.
    - bn2 (nn.BatchNorm1d): Batch normalization layer for fc2 output.
    - dropout2 (nn.Dropout): Dropout layer for fc2 output.
    - fc3 (nn.Linear): Third fully connected layer.
    - bn3 (nn.BatchNorm1d): Batch normalization layer for fc3 output.
    - dropout3 (nn.Dropout): Dropout layer for fc3 output.
    - fc4 (nn.Linear): Fourth fully connected layer for output.
    """

    def __init__(self):
        super(MeadVERHM, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(13 * max_pad_len, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128, 7)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
        - x (tensor): Input tensor.

        Returns:
        - x (tensor): Output tensor.
        """
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x
    
    def MeadLAE(self, predicted_timestamps, ground_truth_timestamps):
        """
        Calculate the Localization Accuracy Error (LAE) for MEAD.

        Args:
        - predicted_timestamps (list): List of predicted timestamps.
        - ground_truth_timestamps (list): List of ground truth timestamps.

        Returns:
        - LAE (float): Localization Accuracy Error.
        """
        # Check if the lengths of predicted and ground truth timestamps are equal
        if len(predicted_timestamps) != len(ground_truth_timestamps):
            raise ValueError("Length of predicted timestamps and ground truth timestamps must be equal.")
        
        # Initialize variables
        N = len(predicted_timestamps)
        total_error = 0.0
        count = 0
        
        # Calculate the total error
        for i in range(N):
            if ground_truth_timestamps[i] != 0:
                error = abs(predicted_timestamps[i] - ground_truth_timestamps[i]) / ground_truth_timestamps[i]
                total_error += error
                count += 1
        
        if count == 0:
            return 0.0
        
        # Calculate LAE
        LAE = total_error / count
        return LAE
