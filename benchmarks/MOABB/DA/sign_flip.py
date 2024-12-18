import numpy as np
import torch

def sign_flip(eeg_data, paug=0.5):
    """
    Applies Sign Flip augmentation to EEG data.

    Parameters:
    - eeg_data: PyTorch tensor of EEG data (shape: samples x time x channels)
    - paug: Probability of flipping the sign of each channel

    Returns:
    - flipped_eeg_data: EEG data after applying sign flip as a PyTorch tensor
    """
    # Convert the input EEG data from PyTorch tensor to NumPy array
    eeg_data_np = eeg_data.numpy()
    
    # Generate a flip mask for each sample, with the same shape as the channels
    # The mask will be -1 with probability paug and 1 with probability (1 - paug)
    flip_mask = np.random.binomial(1, paug, size=(eeg_data_np.shape[0], 1, eeg_data_np.shape[2])) * 2 - 1

    # Apply the flip mask to the EEG data
    flipped_eeg_data_np = eeg_data_np * flip_mask

    # Convert the flipped EEG data back to PyTorch tensor
    flipped_eeg_data = torch.from_numpy(flipped_eeg_data_np).float()

    return flipped_eeg_data

# Example usage
train_data = torch.ones((4, 256, 22)) * torch.arange(4).reshape((4, 1, 1))
flipped_train_data = sign_flip(train_data, paug=0.5)

# Print original and flipped data for comparison
print("Original Data (Sample 0, Channel 0):", train_data[0, :, 0])
print("Flipped Data (Sample 0, Channel 0):", flipped_train_data[0, :, 0])
