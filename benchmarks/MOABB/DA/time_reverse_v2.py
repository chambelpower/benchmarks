import numpy as np
import torch
import torch.nn as nn

class TimeReverse(nn.Module):
	def __init__(self, paug=0.5):
		super(TimeReverse, self).__init__()
		self.paug = paug

	def forward(self, waveforms):
		# Convert the input EEG data from PyTorch tensor to NumPy array
		eeg_data_np = waveforms.numpy()
		
		# Generate a mask for each sample to decide whether to reverse the time
		reverse_mask = np.random.binomial(1, self.paug, size=(eeg_data_np.shape[0]))

		# Apply the time reversal based on the mask
		reversed_eeg_data_np = np.array([sample[:, ::-1] if reverse else sample for sample, reverse in zip(eeg_data_np, reverse_mask)])

		# Convert the reversed EEG data back to PyTorch tensor
		reversed_eeg_data = torch.from_numpy(reversed_eeg_data_np).float()

		return reversed_eeg_data

# train_data = torch.ones((4, 256, 22)) * torch.arange(4).reshape((4, 1, 1))
# time_reverse = TimeReverse(paug=0.5)
# reversed_train_data = time_reverse(train_data)

# print(train_data.shape)
# print(reversed_train_data.shape)