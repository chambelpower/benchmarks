import numpy as np
import torch

def channels_dropout(eeg_data, pdrop=0.5):
    """
    Applies Channels Dropout augmentation to EEG data.

    Parameters:
    - eeg_data: PyTorch tensor of EEG data (shape: samples x time x channels)
    - pdrop: Probability of dropping (setting to zero) each channel

    Returns:
    - dropout_eeg_data: EEG data after applying channels dropout as a PyTorch tensor
    """
    # Convert the input EEG data from PyTorch tensor to NumPy array
    eeg_data_np = eeg_data.numpy()
    
    # Generate a dropout mask for each sample, with the same shape as the channels
    # The mask will be 0 with probability pdrop and 1 with probability (1 - pdrop)
    dropout_mask = np.random.binomial(1, 1 - pdrop, size=(eeg_data_np.shape[0], 1, eeg_data_np.shape[2]))

    # Apply the dropout mask to the EEG data
    dropout_eeg_data_np = eeg_data_np * dropout_mask

    # Convert the dropout EEG data back to PyTorch tensor
    dropout_eeg_data = torch.from_numpy(dropout_eeg_data_np).float()

    return dropout_eeg_data

# Example usage
train_data = torch.ones((4, 256, 22)) * torch.arange(4).reshape((4, 1, 1))
dropout_train_data = channels_dropout(train_data, pdrop=0.5)

# Print original and dropout data for comparison
print("Original Data (Sample 0, Channel 0):", train_data[0, :, 0])
print("Dropout Data (Sample 0, Channel 0):", dropout_train_data[0, :, 0])
