import torch

def calculate_loss(predicted_morphs, ground_truth_morphs, predicted_style_features, style_features, 
                   predicted_emotion_labels, emotion_labels, lambda_recon=1.0, lambda_style=0.5, 
                   lambda_smooth=0.05, lambda_emotion=0.1):
    """
    Calculate the total loss based on various components.
    
    Args:
        predicted_morphs (torch.Tensor): Predicted morph coefficients.
        ground_truth_morphs (torch.Tensor): Ground truth morph coefficients.
        predicted_style_features (torch.Tensor): Predicted style features.
        style_features (torch.Tensor): Target style features.
        predicted_emotion_labels (torch.Tensor): Predicted emotion labels.
        emotion_labels (torch.Tensor): Ground truth emotion labels.
        lambda_recon (float): Weight for reconstruction loss (default is 1.0).
        lambda_style (float): Weight for style loss (default is 1.0).
        lambda_smooth (float): Weight for smoothness loss (default is 1.0).
        lambda_emotion (float): Weight for emotion classification loss (default is 1.0).
        
    Returns:
        total_loss (torch.Tensor): Total loss.
    """
    # Reconstruction Loss: Measures the difference between the predicted morph coefficients and the ground truth.
    reconstruction_loss = torch.nn.MSELoss()(predicted_morphs, ground_truth_morphs)
    
    # Style Loss: Measures the difference between the predicted style features and the target style features.
    style_loss = torch.nn.MSELoss()(predicted_style_features, style_features)
    
    # Smoothness Loss: Penalizes sudden changes in morph coefficients, promoting smooth animations.
    smoothness_loss = torch.nn.MSELoss()(predicted_morphs[:, 1:], predicted_morphs[:, :-1])
    
    # Emotion Classification Loss: Measures how well the predicted emotion labels match the ground truth.
    emotion_classification_loss = torch.nn.CrossEntropyLoss()(predicted_emotion_labels, emotion_labels)
    
    # Combine the individual losses with their respective weights.
    total_loss = lambda_recon * reconstruction_loss + lambda_style * style_loss + \
                 lambda_smooth * smoothness_loss + lambda_emotion * emotion_classification_loss
    
    return total_loss
