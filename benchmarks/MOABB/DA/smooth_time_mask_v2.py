import numpy as np
import torch
import torch.nn as nn

class SmoothTimeMask(nn.Module):
	def __init__(self, mask_length, paug=0.5, lambda_val=10):
		super(SmoothTimeMask, self).__init__()
		self.mask_length = mask_length
		self.paug = paug
		self.lambda_val = lambda_val

	def forward(self, waveforms):
	    def sigmoid(x, lambda_val):
	        return 1 / (1 + np.exp(-lambda_val * x))
	    
	    # Move tensors to CPU and convert to NumPy
	    eeg_data_np = waveforms.cpu().numpy()
	    masked_eeg_data_np = np.copy(eeg_data_np)

	    for sample in masked_eeg_data_np:
	        if np.random.rand() < self.paug:
	            t_max = sample.shape[0]
	            t_cut = np.random.randint(0, t_max - self.mask_length)
	            t = np.arange(t_max)
	            mask = sigmoid(t - t_cut, self.lambda_val) * sigmoid(t_cut + self.mask_length - t, self.lambda_val)
	            sample *= mask[:, np.newaxis]
	    
	    # Convert back to PyTorch tensor and move to the original device
	    masked_eeg_data = torch.from_numpy(masked_eeg_data_np).float().to(waveforms.device)
	    return masked_eeg_data

# train_data = torch.ones((4, 256, 22)) * torch.arange(4).reshape((4, 1, 1))
# smooth = SmoothTimeMask(mask_length=50, paug=0.5, lambda_val=10)
# masked_train_data = smooth(train_data)

# print(train_data.shape)
# print(masked_train_data.shape)