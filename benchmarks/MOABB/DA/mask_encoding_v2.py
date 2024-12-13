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
		eeg_data_np = waveforms.numpy()
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
    	
		masked_eeg_data = torch.from_numpy(masked_eeg_data_np).float()
		return masked_eeg_data

# train_data = torch.ones((4, 256, 22)) * torch.arange(4).reshape((4, 1, 1))
# mask = MaskEncoding(mask_ratio=0.2, paug=0.5, strategy=2)
# masked_train_data = mask(train_data)

# print(train_data.shape)
# print(masked_train_data.shape)