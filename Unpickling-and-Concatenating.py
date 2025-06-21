### CIFAR 10 comes in 5 training batches and 1 test batch. We provide a quick script that joins the training batches into one single training set.

import pickle
import numpy as np
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Path to the folder containing the CIFAR-10 batches
data_dir = '/Users/kryptonempyrean/Downloads/cifar-10-batches-py'

# Initialize lists to store data and labels
data_list = []
labels_list = []

# Load all 5 training batches
for i in range(1, 6):
    batch = unpickle(os.path.join(data_dir, f'data_batch_{i}'))
    data_list.append(batch[b'data'])         # shape: (10000, 3072)
    labels_list.extend(batch[b'labels'])     # list of 10000 elements

# Concatenate into single arrays
Xtr = np.concatenate(data_list, axis=0)      # shape: (50000, 3072)
Yr = np.array(labels_list)                  # shape: (50000,)

# Each image is 32x32 with 3 color channels
Xtr = Xtr.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # shape: (50000, 32, 32, 3)

# Load the test batch
test_batch = unpickle(os.path.join(data_dir, 'test_batch'))

Xte = test_batch[b'data']                  # shape: (10000, 3072)
Yte = np.array(test_batch[b'labels'])      # shape: (10000,)

# Optional: reshape the images to (N, 32, 32, 3)
Xte = Xte.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # shape: (10000, 32, 32, 3)
