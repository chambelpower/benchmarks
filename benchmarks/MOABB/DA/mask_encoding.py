import numpy as np
import torch

def mask_encoding(eeg_data, mask_ratio, paug=0.5, strategy=2):
    """
    Applies Mask Encoding (EEG-ME) augmentation to EEG data.
    
    Parameters:
    - eeg_data: PyTorch tensor of EEG data (shape: samples x time x channels)
    - mask_ratio: Ratio of the mask length to the total data length
    - paug: Probability of applying the mask to each sample
    - strategy: Mask variation strategy (1 for individual samples, 2 for mini-batches)
    
    Returns:
    - masked_eeg_data: EEG data after applying mask encoding as a PyTorch tensor
    """
    eeg_data_np = eeg_data.numpy()
    masked_eeg_data_np = np.copy(eeg_data_np)
    
    num_samples, num_time_steps, num_channels = eeg_data_np.shape
    mask_length = int(num_time_steps * mask_ratio)
    mini_batch_size = 32  # Assume a mini-batch size (adjust as necessary)
    
    for sample_idx in range(num_samples):
        if np.random.rand() < paug:
            if strategy == 1:
                # Random variation for individual samples
                rm = np.random.randint(0, num_time_steps - mask_length)
            elif strategy == 2:
                # Consistent value for mini-batch samples, random across different mini-batches
                batch_idx = sample_idx // mini_batch_size
                np.random.seed(batch_idx)  # Ensure the same seed for all samples in the mini-batch
                rm = np.random.randint(0, num_time_steps - mask_length)
                
            # Apply the mask to all channels of the current sample
            masked_eeg_data_np[sample_idx, rm:rm + mask_length, :] = 0
    
    masked_eeg_data = torch.from_numpy(masked_eeg_data_np).float()
    return masked_eeg_data

# Example usage
train_data = torch.ones((4, 256, 22)) * torch.arange(4).reshape((4, 1, 1))
masked_train_data = mask_encoding(train_data, mask_ratio=0.2, paug=0.5, strategy=2)

# Print original and masked data for comparison
print("Original Data (Sample 0, Channel 0):", train_data[0, :, 0])
print("Masked Data (Sample 0, Channel 0):", masked_train_data[0, :, 0])
