import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def eeg_to_stft_images(eeg_data, fs=128, nperseg=22, noverlap=11):
    """
    Converts EEG signals to STFT images.
    
    Parameters:
    - eeg_data: NumPy array of EEG data (shape: samples x channels x time)
    - fs: Sampling frequency
    - nperseg: Length of each segment for STFT
    - noverlap: Number of points to overlap between segments
    
    Returns:
    - stft_images: List of STFT images for each channel
    """
    stft_images = []
    for sample in eeg_data:
        sample_images = []
        for channel in sample:
            f, t, Zxx = stft(channel, fs=fs, nperseg=nperseg, noverlap=noverlap)
            sample_images.append(np.abs(Zxx))
        stft_images.append(sample_images)
    
    return np.array(stft_images)

def create_multi_input_cnn(input_shape, num_channels, num_classes):
    """
    Creates a multi-input CNN model.
    
    Parameters:
    - input_shape: Shape of each input image (height, width, channels)
    - num_channels: Number of EEG channels
    - num_classes: Number of output classes
    
    Returns:
    - model: Compiled Keras model
    """
    inputs = []
    convs = []
    
    for i in range(num_channels):
        input_img = Input(shape=input_shape)
        inputs.append(input_img)
        
        conv = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_img)
        conv = MaxPooling2D(pool_size=(2, 2))(conv)
        conv = Flatten()(conv)
        
        convs.append(conv)
    
    merged = Concatenate()(convs)
    dense = Dense(128, activation='relu')(merged)
    output = Dense(num_classes, activation='softmax')(dense)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage
train_data = np.ones((4, 256, 22)) * np.arange(4).reshape((4, 1, 1))
stft_images = eeg_to_stft_images(train_data)

# Add a channel dimension to the STFT images
stft_images = stft_images[..., np.newaxis]

input_shape = stft_images.shape[2:]  # Shape of the STFT images (frequency bins, time bins, channels)
num_channels = train_data.shape[1]  # Number of EEG channels
num_classes = 2  # Example number of output classes

model = create_multi_input_cnn(input_shape, num_channels, num_classes)
model.summary()

# Assuming `labels` is a NumPy array of shape (num_samples, num_classes) for one-hot encoded labels
labels = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])  # Example labels

# Split the STFT images for each channel
stft_images_split = [stft_images[:, i, :, :, :] for i in range(num_channels)]

# Train the model
model.fit(stft_images_split, labels, epochs=10, batch_size=32, validation_split=0.2)
