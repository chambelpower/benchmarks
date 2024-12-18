import torch
import torch.nn as nn

class ChannelPermutation(nn.Module):
    def __init__(self, permutation_prob=0.1):
        """
        Channel Permutation Data Augmentation.

        Parameters:
        - permutation_prob (float): Probability of permuting the order of channels.
        """
        super(ChannelPermutation, self).__init__()
        self.permutation_prob = permutation_prob

    def forward(self, waveforms):
        """
        Apply channel permutation augmentation.

        Parameters:
        - waveforms (torch.Tensor): Input EEG data with shape (batch_size, num_timepoints, num_channels).

        Returns:
        - concatenated_data (torch.Tensor): Concatenated original and augmented data.
        """
        batch_size, num_timepoints, num_channels = waveforms.size()

        augmented_data = []

        for i in range(batch_size):
            # Check if permutation should be applied to the current sample
            if torch.rand(1).item() < self.permutation_prob:
                # Generate a random permutation of channel indices
                permutation_indices = torch.randperm(num_channels)
                # Apply permutation to the current sample
                permuted_sample = waveforms[i, :, permutation_indices]
                augmented_data.append(permuted_sample)
            else:
                augmented_data.append(waveforms[i])

        # Stack the list into a tensor
        augmented_data = torch.stack(augmented_data)

        return augmented_data
