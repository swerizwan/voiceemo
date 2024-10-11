import torch
import math

def init_biased_mask(n_head, max_seq_len, period):
    """
    Initialize a biased mask for self-attention mechanism.

    Args:
        n_head (int): Number of attention heads.
        max_seq_len (int): Maximum sequence length.
        period (int): Period for the biased mask.

    Returns:
        torch.Tensor: Biased mask for self-attention.
    """
    def get_slopes(n):
        """
        Calculate slopes for the bias.

        Args:
            n (int): Number of attention heads.

        Returns:
            list: List of slopes for each attention head.
        """
        def get_slopes_power_of_2(n):
            """
            Calculate slopes for a power of 2.

            Args:
                n (int): Number of attention heads.

            Returns:
                list: List of slopes for a power of 2.
            """
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                               :n - closest_power_of_2]

    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1, period).view(-1) // (period)
    bias = - torch.flip(bias, dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i + 1] = bias[-(i + 1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

import numpy as np

def calculate_mean_std(data):
    """
    Calculate the mean and standard deviation of a dataset.

    Args:
        data (numpy.ndarray): Input dataset.

    Returns:
        tuple: Mean and standard deviation of the dataset.
    """
    mean = np.mean(data)
    std = np.std(data)
    return mean, std

def sigmoid(x):
    """
    Compute the sigmoid function for the input.

    Args:
        x (float or numpy.ndarray): Input value or array.

    Returns:
        float or numpy.ndarray: Sigmoid of the input.
    """
    return 1 / (1 + np.exp(-x))

def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.

    Args:
        vec1 (numpy.ndarray): First vector.
        vec2 (numpy.ndarray): Second vector.

    Returns:
        float: Cosine similarity between the two vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    cosine_sim = dot_product / (norm_vec1 * norm_vec2)
    return cosine_sim

def euclidean_distance(vec1, vec2):
    """
    Calculate the Euclidean distance between two vectors.

    Args:
        vec1 (numpy.ndarray): First vector.
        vec2 (numpy.ndarray): Second vector.

    Returns:
        float: Euclidean distance between the two vectors.
    """
    return np.linalg.norm(vec1 - vec2)

def preprocess_text(text):
    """
    Preprocess the input text by removing special characters and converting to lowercase.

    Args:
        text (str): Input text.

    Returns:
        str: Preprocessed text.
    """
    text = text.lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text



def enc_dec_mask(device, T, S):
    """
    Generate an encoder-decoder mask for attention mechanism.

    Args:
        device (torch.device): Device to place the mask tensor.
        T (int): Length of the sequence in the encoder.
        S (int): Length of the sequence in the decoder.

    Returns:
        torch.Tensor: Encoder-decoder mask.
    """
    mask = torch.ones(T, S).to(device)
    for i in range(T):
        mask[i, i] = 0
    return (mask == 1).to(device=device)
