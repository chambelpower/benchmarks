import numpy as np
import torch
import torch.nn as nn 

class SignFlip(nn.Module):
	def __init__(self, paug=0.5):
		super(SignFlip, self).__init__()
		self.paug = paug

	def forward(self, waveforms):
		# Convert the input EEG data from PyTorch tensor to NumPy array
		eeg_data_np = waveforms.numpy()
		
		# Generate a flip mask for each sample, with the same shape as the channels
		# The mask will be -1 with probability paug and 1 with probability (1 - paug)
		flip_mask = np.random.binomial(1, self.paug, size=(eeg_data_np.shape[0], 1, eeg_data_np.shape[2])) * 2 - 1

		# Apply the flip mask to the EEG data
		flipped_eeg_data_np = eeg_data_np * flip_mask

		# Convert the flipped EEG data back to PyTorch tensor
		flipped_eeg_data = torch.from_numpy(flipped_eeg_data_np).float()

		return flipped_eeg_data

# train_data = torch.ones((4, 256, 22)) * torch.arange(4).reshape((4, 1, 1))
# sign_flip = SignFlip(paug=0.5)
# flipped_train_data = sign_flip(train_data)

# print(train_data.shape)
# print(flipped_train_data.shape)