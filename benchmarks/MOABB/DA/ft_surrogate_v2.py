import numpy as np
import torch
import torch.nn as nn

class Ft_Surrogate(nn.Module):
	def __init__(self, lambda_=1.0, probability=0.5):
		super(Ft_Surrogate, self).__init__()
		self.lambda_=lambda_
		self.probability=probability

	def forward(self, waveforms):

		eeg_data_np = waveforms.numpy()  # Convert to numpy for processing
		surrogate_eeg_data_np = np.zeros_like(eeg_data_np)

		for sample_idx, sample in enumerate(eeg_data_np):
			for channel_idx, channel in enumerate(sample.T):  # Transpose to iterate over channels
				if np.random.rand() < self.probability:
					# Perform Fourier transform
					fft_coeffs = np.fft.fft(channel)
					amplitudes = np.abs(fft_coeffs)
					phases = np.angle(fft_coeffs)
                
					# Generate new random phases
					new_phases = np.random.uniform(0, self.lambda_ * 2 * np.pi, size=phases.shape)
                
					# Construct new Fourier coefficients with original amplitudes and new phases
					new_fft_coeffs = amplitudes * np.exp(1j * new_phases)
                
					# Perform inverse Fourier transform to get the surrogate signal
					surrogate_channel = np.fft.ifft(new_fft_coeffs).real
                
					surrogate_eeg_data_np[sample_idx, :, channel_idx] = surrogate_channel
				else:
					surrogate_eeg_data_np[sample_idx, :, channel_idx] = channel

		surrogate_eeg_data = torch.from_numpy(surrogate_eeg_data_np).float()  # Convert back to PyTorch tensor

		return surrogate_eeg_data


# train_data = torch.ones((4, 256, 22)) * torch.arange(4).reshape((4, 1, 1))
# ftSurrogate = Ft_Surrogate(lambda_=1.0, probability=0.5)
# surrogate_train_data = ftSurrogate(train_data)

# print(train_data.shape)
# print(surrogate_train_data.shape)