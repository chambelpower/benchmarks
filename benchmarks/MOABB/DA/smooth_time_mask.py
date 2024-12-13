import numpy as np
import torch

def smooth_time_mask(eeg_data, mask_length, paug=0.5, lambda_val=10):
    """
    Applies Smooth Time Mask (STM) augmentation to EEG data.

    Parameters:
    - eeg_data: PyTorch tensor of EEG data (shape: samples x time x channels)
    - mask_length: Length of the mask in time steps
    - paug: Probability of applying the mask to each sample
    - lambda_val: Temperature parameter for the sigmoid functions

    Returns:
    - masked_eeg_data: EEG data after applying smooth time mask as a PyTorch tensor
    """
    def sigmoid(x, lambda_val):
        return 1 / (1 + np.exp(-lambda_val * x))
    
    eeg_data_np = eeg_data.numpy()
    masked_eeg_data_np = np.copy(eeg_data_np)

    for sample in masked_eeg_data_np:
        if np.random.rand() < paug:
            t_max = sample.shape[0]
            t_cut = np.random.randint(0, t_max - mask_length)
            t = np.arange(t_max)
            mask = sigmoid(t - t_cut, lambda_val) * sigmoid(t_cut + mask_length - t, lambda_val)
            sample *= mask[:, np.newaxis]
    
    masked_eeg_data = torch.from_numpy(masked_eeg_data_np).float()
    return masked_eeg_data

# Example usage
train_data = torch.ones((4, 256, 22)) * torch.arange(4).reshape((4, 1, 1))
masked_train_data = smooth_time_mask(train_data, mask_length=50, paug=0.5, lambda_val=10)

# Print original and masked data for comparison
print("Original Data (Sample 0, Channel 0):", train_data[0, :, 0])
print("Masked Data (Sample 0, Channel 0):", masked_train_data[0, :, 0])
