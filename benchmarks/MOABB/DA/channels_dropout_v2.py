import numpy as np
import torch
import torch.nn as nn

class ChannelsDropout(nn.Module):
	def __init__(self, pdrop=0.5):
		super(ChannelsDropout, self).__init__()
		self.pdrop = pdrop

	def forward(self, waveforms):
		# Convert the input EEG data from PyTorch tensor to NumPy array
		eeg_data_np = waveforms.numpy()
    
		# Generate a dropout mask for each sample, with the same shape as the channels
		# The mask will be 0 with probability pdrop and 1 with probability (1 - pdrop)
		dropout_mask = np.random.binomial(1, 1 - self.pdrop, size=(eeg_data_np.shape[0], 1, eeg_data_np.shape[2]))

		# Apply the dropout mask to the EEG data
		dropout_eeg_data_np = eeg_data_np * dropout_mask

		# Convert the dropout EEG data back to PyTorch tensor
		dropout_eeg_data = torch.from_numpy(dropout_eeg_data_np).float()

		return dropout_eeg_data

# train_data = torch.ones((4, 256, 22)) * torch.arange(4).reshape((4, 1, 1))
# channels_dropout = ChannelsDropout(pdrop=0.5)
# dropout_train_data = channels_dropout(train_data)

# print(train_data.shape)
# print(dropout_train_data.shape)