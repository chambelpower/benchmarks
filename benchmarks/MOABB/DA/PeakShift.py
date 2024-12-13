import torch
import torch.nn as nn

class PeakShift(nn.Module):
    def __init__(self, shift_range=(-10, 10), time_window=(250, 650), sample_rate=256):
        """
        Initialize the PeakShift augmenter.
        
        Parameters:
        shift_range (tuple): The range of time shifts in milliseconds.
        time_window (tuple): The time window around the peak to apply the shift (in milliseconds).
        sample_rate (int): The sampling rate of the data in Hz.
        """
        super(PeakShift, self).__init__()
        self.shift_range = shift_range
        self.time_window = time_window
        self.sample_rate = sample_rate
        self.samples_per_ms = sample_rate / 1000

    def forward(self, waveforms):
        """
        Apply the Peak-Shift augmentation.
        
        Parameters:
        waveforms (torch.Tensor): The input data to be augmented.
                                  Shape should be (n_samples, time_points, n_channels).
        
        Returns:
        torch.Tensor: The augmented data with near-peak shifts.
        """
        if not isinstance(waveforms, torch.Tensor):
            raise TypeError("Input waveforms must be a torch.Tensor")

        start_sample = int(self.time_window[0] * self.samples_per_ms)
        end_sample = int(self.time_window[1] * self.samples_per_ms)
        shifts = torch.arange(self.shift_range[0], self.shift_range[1] + 1, 10)

        augmented_data = []
        for shift in shifts:
            shifted_data = waveforms.clone()  # Use clone to create a copy of the tensor
            shift_samples = int(shift * self.samples_per_ms)
            if shift_samples != 0:
                # Use torch.roll to shift the data
                shifted_data[:, start_sample:end_sample] = torch.roll(waveforms[:, start_sample:end_sample], shifts=shift_samples, dims=1)
            augmented_data.append(shifted_data)
        
        augmented_data = torch.cat(augmented_data, dim=0)
        print(augmented_data.shape)
        return augmented_data



# # Generate some example data
# train_data = np.ones((4, 256, 22)) * np.arange(4).reshape((4, 1, 1))
    
# # Create an instance of the PeakShift augmenter
# peak_shift_augmenter = PeakShift(shift_range=(-10, 10), time_window=(250, 650), sample_rate=256)
    
# # Use the augmenter on your data
# augmented_data = peak_shift_augmenter.forward(train_data)
    
# print("Original data shape:", train_data.shape)
# print("Augmented data shape:", augmented_data.shape)
