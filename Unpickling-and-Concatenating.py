### CIFAR 10 comes in 5 training batches and 1 test batch. We provide a quick script that joins the training batches into one single training set.

import pickle
import numpy as np
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Path to the folder containing the CIFAR-10 batches
data_dir = 'cifar-10-batches-py'

# Initialize lists to store data and labels
data_list = []
labels_list = []

# Load all 5 training batches
for i in range(1, 6):
    batch = unpickle(os.path.join(data_dir, f'data_batch_{i}'))
    data_list.append(batch[b'data'])         # shape: (10000, 3072)
    labels_list.extend(batch[b'labels'])     # list of 10000 elements

# Concatenate into single arrays
X_train = np.concatenate(data_list, axis=0)      # shape: (50000, 3072)
y_train = np.array(labels_list)                  # shape: (50000,)

# Optional: reshape the images
# Each image is 32x32 with 3 color channels
X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # shape: (50000, 32, 32, 3)
