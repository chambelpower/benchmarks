import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def loss_function(x, x_reconstructed, mu, logvar):
    recon_loss = nn.functional.mse_loss(x_reconstructed, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        h = torch.relu(self.fc1(z))
        x_reconstructed = self.fc2(h)  # Removing sigmoid for regression
        return x_reconstructed

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, learning_rate, epochs, num_samples):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.epochs=epochs
        self.num_samples=num_samples

    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward_vae(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

    # Generate new data points
    def generate_augmented_data(self, num_samples):
        with torch.no_grad():
            # Sample from the normal distribution
            z = torch.randn(num_samples, latent_dim)
            # Decode the samples to get new data points
            augmented_data = self.decoder(z)
            # Reshape the data to the original shape
            augmented_data = augmented_data.view(num_samples, 256, 22)
        return augmented_data

    def forward(self, waveforms):
        # Flatten the training data for the VAE
        flattened_waveforms = waveforms.view(waveforms.size(0), -1)

        # Create DataLoader
        train_dataset = TensorDataset(flattened_waveforms)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

        # Training loop
        for epoch in range(self.epochs):
            for batch_idx, (data,) in enumerate(train_loader):
                self.optimizer.zero_grad()
                x_reconstructed, mu, logvar = self.forward_vae(data)
                loss = loss_function(data, x_reconstructed, mu, logvar)
                loss.backward()
                self.optimizer.step()

            #print(f'Epoch {epoch+1}, Loss: {loss.item()}')

        # Generate new samples
        augmented_data = self.generate_augmented_data(self.num_samples)
        
        return augmented_data


# #Example usage:
# # Hyperparameters
# input_dim = 256 * 22  # Flattened input dimension
# hidden_dim = 400
# latent_dim = 20
# learning_rate = 1e-3
# epochs = 10
# num_samples=5
# vae = VAE(input_dim, hidden_dim, latent_dim, learning_rate, epochs, num_samples)

# #Use the provided signal
# signal = torch.ones((4, 256, 22)) * torch.arange(4).reshape((4, 1, 1,))

# vae.forward(signal)