import torch
import torch.nn as nn

class AllAmp(nn.Module):
    def __init__(self, factors=[0.9, 1.1]):
        """
        Initialize the AllAmp augmenter.
        
        Parameters:
        factors (list of float): The factors by which to multiply the data for augmentation.
        """
        super(AllAmp, self).__init__()
        self.factors = factors

    def forward(self, waveforms):
        """
        Multiply all sampled data by specified factors.
        
        Parameters:
        waveforms (torch.Tensor): The input data to be augmented. 
                                  Shape should be (n_samples, time_points, n_channels).
        
        Returns:
        torch.Tensor: The augmented data with all-time multiplications.
        """

        # Ensure waveforms is a PyTorch tensor
        if not isinstance(waveforms, torch.Tensor):
            raise TypeError("Input waveforms must be a torch.Tensor")

        # Perform the augmentation using PyTorch tensors
        augmented_data = [waveforms * factor for factor in self.factors]
        
        # Concatenate the augmented data along the batch dimension
        augmented_data = torch.cat(augmented_data, dim=0)

        #print(waveforms.shape)
        #print(augmented_data.shape)
       
        return augmented_data



# # Generate some example data
# train_data = np.ones((4, 256, 22)) * np.arange(4).reshape((4, 1, 1))
    
# # Create an instance of the AllAmp augmenter
# amp_augmenter = AllAmp(factors=[0.9, 1.1])
    
# # Use the augmenter on your data
# augmented_data = amp_augmenter.forward(train_data)
    
# print("Original data shape:", train_data.shape)
# print("Augmented data shape:", augmented_data.shape)
