import numpy as np
import torch

def time_reverse(eeg_data, paug=0.5):
    """
    Applies Time Reversal augmentation to EEG data.

    Parameters:
    - eeg_data: PyTorch tensor of EEG data (shape: samples x time x channels)
    - paug: Probability of reversing the time of each sample

    Returns:
    - reversed_eeg_data: EEG data after applying time reversal as a PyTorch tensor
    """
    # Convert the input EEG data from PyTorch tensor to NumPy array
    eeg_data_np = eeg_data.numpy()
    
    # Generate a mask for each sample to decide whether to reverse the time
    reverse_mask = np.random.binomial(1, paug, size=(eeg_data_np.shape[0]))

    # Apply the time reversal based on the mask
    reversed_eeg_data_np = np.array([sample[:, ::-1] if reverse else sample for sample, reverse in zip(eeg_data_np, reverse_mask)])

    # Convert the reversed EEG data back to PyTorch tensor
    reversed_eeg_data = torch.from_numpy(reversed_eeg_data_np).float()

    return reversed_eeg_data

# Example usage
train_data = torch.ones((4, 256, 22)) * torch.arange(4).reshape((4, 1, 1))
reversed_train_data = time_reverse(train_data, paug=0.5)

# Print original and reversed data for comparison
print("Original Data (Sample 0, Channel 0):", train_data[0, :, 0])
print("Reversed Data (Sample 0, Channel 0):", reversed_train_data[0, :, 0])
