import torch
import torch.nn as nn

class PeakAmp(nn.Module):
    def __init__(self, factors=[0.9, 1.1], time_window=(250, 650), sample_rate=256):
        """
        Initialize the PeakAmp augmenter.
        
        Parameters:
        factors (list of float): The factors by which to multiply the data for augmentation.
        time_window (tuple): The time window around the peak to apply the amplification (in milliseconds).
        sample_rate (int): The sampling rate of the data in Hz.
        """
        super(PeakAmp, self).__init__()
        self.factors = factors
        self.time_window = time_window
        self.sample_rate = sample_rate
        self.samples_per_ms = sample_rate / 1000

    def forward(self, waveforms):
        """
        Apply the Peak-Amp augmentation.
        
        Parameters:
        waveforms (torch.Tensor): The input data to be augmented.
                                  Shape should be (n_samples, time_points, n_channels).
        
        Returns:
        torch.Tensor: The augmented data with near-peak amplifications.
        """
        if not isinstance(waveforms, torch.Tensor):
            raise TypeError("Input waveforms must be a torch.Tensor")

        start_sample = int(self.time_window[0] * self.samples_per_ms)
        end_sample = int(self.time_window[1] * self.samples_per_ms)
        
        augmented_data = []
        for factor in self.factors:
            amplified_data = waveforms.clone()  # Use clone to create a copy of the tensor
            amplified_data[:, start_sample:end_sample] *= factor
            augmented_data.append(amplified_data)
        
        augmented_data = torch.cat(augmented_data, dim=0)
        # print(augmented_data.shape)
        return augmented_data


# # Generate some example data
# train_data = np.ones((4, 256, 22)) * np.arange(4).reshape((4, 1, 1))
    
# # Create an instance of the PeakAmp augmenter
# peak_amp_augmenter = PeakAmp(factors=[0.9, 1.1], time_window=(250, 650), sample_rate=256)
    
# # Use the augmenter on your data
# augmented_data = peak_amp_augmenter.forward(train_data)
    
# print("Original data shape:", train_data.shape)
# print("Augmented data shape:", augmented_data.shape)
