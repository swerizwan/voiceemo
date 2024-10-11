import torch
import torchvision.models as models

def train_resnet_model(facial_images, ground_truth_morphs, num_epochs=10):
    """
    Train a ResNet model for facial expression reconstruction.
    
    Args:
        facial_images (torch.Tensor): Tensor containing facial images.
        ground_truth_morphs (torch.Tensor): Tensor containing ground truth morph coefficients.
        num_epochs (int): Number of epochs for training (default is 10).
        
    Returns:
        trained_model: Trained ResNet model for facial expression reconstruction.
    """
    # Define ResNet model architecture
    # Load pre-trained ResNet18 model
    resnet_model = models.resnet18(pretrained=False)
    
    # Customize last layer for regression task
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = torch.nn.Linear(num_ftrs, len(ground_truth_morphs[0]))
    
    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = resnet_model(facial_images)
        
        # Compute the loss
        loss = criterion(outputs, ground_truth_morphs)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return resnet_model

def predict_morph_coefficients(facial_images, trained_model):
    """
    Predict morph coefficients using a trained ResNet model.
    
    Args:
        facial_images (torch.Tensor): Tensor containing facial images.
        trained_model: Trained ResNet model for facial expression reconstruction.
        
    Returns:
        outputs: Predicted morph coefficients.
    """
    # Predict morph coefficients using trained ResNet model
    with torch.no_grad():
        outputs = trained_model(facial_images)
    
    return outputs
