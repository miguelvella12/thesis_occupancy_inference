import torch

def simulate_random_delays(x_batch, min_delay, max_delay):
    """
    Apply a random temporal shift to each sequence in the batch to simulate variable sensor delay.

    :param x_batch: Tensor of shape [batch_size, seq_len, input_dim]
    :param min_delay: Minimum delay (in timesteps)
    :param max_delay: Maximum delay (in timesteps)
    :return: Tensor with each sample shifted backward by a random number of timesteps
    """
    x_batch = x_batch.clone()
    for i in range(x_batch.size(0)):
        delay_steps = torch.randint(min_delay, max_delay + 1, (1,)).item() # Random delay for each sequence
        x_batch[i] = shift_sequence_backward(x_batch[i], delay_steps) # Shift the sequence backward
    return x_batch

def shift_sequence_backward(x_seq, delay_steps):
    """
    Shift sensor data backward by N timesteps, simulating lagged availability of sensor data.

    :param x_seq: Tensor of shape [seq_len, input_dim] representing the input sequence.
    :param delay_steps: How many timesteps to shift backward.
    :return: Shifted sequence
    """

    if delay_steps == 0:
        return x_seq
    x_seq = x_seq.clone() # Clone to avoid modifying the original tensor
    shifted = torch.roll(x_seq, shifts=-delay_steps, dims=0) # Shift the sequence backward
    shifted[-delay_steps:] = 0 # New entries at start are unknown
    return shifted

def apply_fixed_delay_to_batch(x_batch, delay_steps, strategy="shift"):
    """
    Apply a fixed delay to a batch of sequences by shifting the entire batch backward by N timesteps.

    :param strategy: Strategy to apply the delay. Only option at the moment is "shift".
    :param x_batch: Tensor of shape [batch_size, seq_len, input_dim] representing the input batch.
    :param delay_steps: How many timesteps to shift backward.
    :return: Tensor of shape [batch_size, seq_len, input_dim] with fixed delays applied.
    """

    x_batch = x_batch.clone()
    for i in range(x_batch.size(0)):
        if strategy == "shift":
            x_batch[i] = shift_sequence_backward(x_batch[i], delay_steps)
        else:
            raise ValueError("Invalid strategy. Use 'shift'.")

    return x_batch