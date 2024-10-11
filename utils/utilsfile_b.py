import torch
import math

def get_slopes(n):
    """
    Function to generate slopes for initializing a biased mask used in multi-head attention mechanisms.

    Args:
        n (int): Number of attention heads.

    Returns:
        list: List of slopes for initializing the biased mask.
    """
    def get_slopes_power_of_2(n):
        """
        Function to generate slopes for a number of heads that is a power of 2.

        Args:
            n (int): Number of attention heads.

        Returns:
            list: List of slopes for initializing the biased mask.
        """
        # Calculate the start value for slopes
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        # Generate slopes using geometric progression
        return [start * ratio ** i for i in range(n)]

    # Check if n is a power of 2
    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        # Find the closest power of 2
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        # Generate slopes for the closest power of 2
        slopes_closest_power_of_2 = get_slopes_power_of_2(closest_power_of_2)
        # Select every other slope to match the required number of heads
        return slopes_closest_power_of_2 + get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

def init_biased_mask(n_head, m_seq_l, times):
    """
    Function to initialize a biased mask for multi-head attention.

    Args:
        n_head (int): Number of attention heads.
        m_seq_l (int): Maximum sequence length.
        times (int): Parameter determining the spacing between biases.

    Returns:
        torch.Tensor: Biased mask tensor for multi-head attention.
    """
    # Calculate slopes
    slopes = torch.tensor(get_slopes(n_head))
    # Generate biases at regular intervals
    bias = torch.arange(0, m_seq_l, times).unsqueeze(1).repeat(1, times).view(-1) // times
    bias = -torch.flip(bias, dims=[0])
    # Initialize a lower triangular matrix
    alibi = torch.tril(torch.ones(m_seq_l, m_seq_l))
    # Apply biases to the lower triangular matrix
    for i in range(m_seq_l):
        alibi[i, :i + 1] = bias[-(i + 1):]
    # Scale the biases by slopes
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    # Initialize a mask tensor with 1s in the upper triangle and 0s in the lower triangle
    mask = torch.triu(torch.ones(m_seq_l, m_seq_l)) == 1
    # Convert the mask tensor to float and apply masked_fill to set non-diagonal elements to -inf
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    # Add the biased mask to the upper triangle of the original mask
    mask = mask.unsqueeze(0) + alibi
    return mask

def enc_dec_mask(device, T, S):
    """
    Function to initialize an encoder-decoder attention mask.

    Args:
        device: Device to be used for tensor operations.
        T (int): Length of the target sequence.
        S (int): Length of the source sequence.

    Returns:
        torch.Tensor: Encoder-decoder attention mask tensor.
    """
    # Initialize a mask tensor with 1s
    mask = torch.ones(T, S, device=device)
    # Set diagonal elements to 0 to prevent decoder from attending to future tokens
    mask[torch.arange(T), torch.arange(T)] = 0
    return mask
