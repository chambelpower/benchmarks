import numpy as np
import torch
import torch.nn as nn

class MaskEncoding(nn.Module):
    def __init__(self, mask_ratio, paug=0.5, strategy=2):
        super(MaskEncoding, self).__init__()
        self.mask_ratio = mask_ratio
        self.paug = paug
        self.strategy = strategy

    def forward(self, waveforms):
        # Move tensor to CPU and convert to numpy
        eeg_data_np = waveforms.cpu().numpy()
        masked_eeg_data_np = np.copy(eeg_data_np)

        num_samples, num_time_steps, num_channels = eeg_data_np.shape
        mask_length = int(num_time_steps * self.mask_ratio)
        mini_batch_size = 32  # Assume a mini-batch size (adjust as necessary)

        for sample_idx in range(num_samples):
            if np.random.rand() < self.paug:
                if self.strategy == 1:
                    # Random variation for individual samples
                    rm = np.random.randint(0, num_time_steps - mask_length)
                elif self.strategy == 2:
                    # Consistent value for mini-batch samples, random across different mini-batches
                    batch_idx = sample_idx // mini_batch_size
                    np.random.seed(batch_idx)  # Ensure the same seed for all samples in the mini-batch
                    rm = np.random.randint(0, num_time_steps - mask_length)

                # Apply the mask to all channels of the current sample
                masked_eeg_data_np[sample_idx, rm:rm + mask_length, :] = 0

        # Convert back to PyTorch tensor and move to the original device
        masked_eeg_data = torch.from_numpy(masked_eeg_data_np).float().to(waveforms.device)
        return masked_eeg_data
