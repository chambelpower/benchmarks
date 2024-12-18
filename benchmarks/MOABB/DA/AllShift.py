import numpy as np

import torch
import torch.nn as nn

class AllShift(nn.Module):
    def __init__(self, shift_range=10, sample_rate=128):
        super(AllShift, self).__init__()
        self.shift_range = shift_range
        self.sample_rate = sample_rate

    def forward(self, waveforms):
        """
        Shift sampled data by +10ms and -10ms.
        
        Parameters:
        waveforms (torch.Tensor): The input data to be augmented. 
                                  Shape should be (n_samples, time_points, n_channels).
        
        Returns:
        torch.Tensor: The augmented data with +10ms and -10ms shifts.
        """
        shift_amount = self.shift_range * self.sample_rate // 1000
        
        # Ensure waveforms is a PyTorch tensor
        if not isinstance(waveforms, torch.Tensor):
            raise TypeError("Input waveforms must be a torch.Tensor")
        
        # Perform the shift
        shifted_data_positive = torch.roll(waveforms, shifts=shift_amount, dims=1)
        shifted_data_negative = torch.roll(waveforms, shifts=-shift_amount, dims=1)
        
        # Concatenate the shifted data along the batch dimension
        augmented_data = torch.cat([shifted_data_negative, shifted_data_positive], dim=0)
        
        return augmented_data



# # Generate some example data
# train_data = np.ones((4, 256, 22)) * np.arange(4).reshape((4, 1, 1))
    
# # Create an instance of the AllAmp augmenter
# amp_augmenter = AllShift()
    
# # Use the augmenter on your data
# augmented_data = amp_augmenter.forward(train_data)
# print("Original data shape:", train_data.shape)
# print("Augmented data shape:", augmented_data.shape)