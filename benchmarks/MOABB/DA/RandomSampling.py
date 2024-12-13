import torch
import random
import torch.nn as nn
import torch.nn.functional as F

class RandomSampling(nn.Module):
    def __init__(self, sampling_prob=0.1):
        super(RandomSampling, self).__init__()
        self.sampling_prob = sampling_prob

    def forward(self, eeg_data):
        batch_size, num_timepoints, num_channels = eeg_data.size()
        augmented_data = []

        for i in range(batch_size):
            if torch.rand(1).item() < self.sampling_prob:
                # Sample indices for the current item
                sampled_indices = sorted(random.sample(range(num_timepoints), int(0.8 * num_timepoints)))
                sampled_data = eeg_data[i, sampled_indices, :]
                
                # Pad the sampled data back to the original length
                padding_size = num_timepoints - len(sampled_indices)
                padded_data = F.pad(sampled_data, (0, 0, 0, padding_size))
                augmented_data.append(padded_data)
            else:
                augmented_data.append(eeg_data[i, :, :])

        # Stack the results back into a tensor
        return torch.stack(augmented_data)

# # Example usage:
# eeg_augmentor = RandomSampling(sampling_prob=1)
# signal = torch.ones((4, 256, 22)) * torch.arange(4).reshape((4, 1, 1))
# augmented_signal = eeg_augmentor.forward(signal)
# print(augmented_signal.shape)
