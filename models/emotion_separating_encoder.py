import torch
import torch.nn as nn

class EmotionSeparatingEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the emotion-separating encoder.
        
        Args:
            input_size (int): Dimensionality of input features.
            hidden_size (int): Number of units in the hidden layers.
            output_size (int): Dimensionality of output emotional cues.
        """
        super(EmotionSeparatingEncoder, self).__init__()
        
        # Define encoder layers and architecture
        # The encoder is a simple feedforward neural network with two fully connected layers.
        # Input size is transformed to hidden size, then to output size.
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Input layer: Linear transformation
            nn.ReLU(),                           # Activation function: ReLU
            nn.Linear(hidden_size, output_size)  # Output layer: Linear transformation
        )
        
    def forward(self, input_features):
        """
        Forward pass through the emotion-separating encoder.
        
        Args:
            input_features (torch.Tensor): Input features for the encoder.
            
        Returns:
            output_emotional_cues (torch.Tensor): Output emotional cues.
        """
        # Implement forward pass through encoder
        # The input features are passed through the layers defined in the constructor.
        output_emotional_cues = self.encoder(input_features)
        
        return output_emotional_cues
