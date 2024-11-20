# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Developer details: 
# Name: Akshat Rastogi and Shubh Gupta
# Role: Developers
# Code ownership rights: PreProd Corp

# Description: This Streamlit app allows users to input features and make predictions using Neural Network.
# MQs: No
# Cloud: No
# Data versioning: No
# Data masking: No

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
# Environment:     
# Python 3.10.11
# streamlit 1.40.0
import torchvision.transforms as T  # Importing the torchvision transforms module for image preprocessing
from torchvision.datasets import CocoDetection  # Importing the CocoDetection dataset class to load COCO dataset
from torch.utils.data import DataLoader  # Importing DataLoader to handle batching and shuffling of data

def get_data_loader(annotations_file, images_dir, batch_size=4, shuffle=True):
    """
    Function to create a DataLoader for the COCO dataset.

    Parameters:
    - annotations_file (str): Path to the COCO annotations file.
    - images_dir (str): Directory where the images are stored.
    - batch_size (int, optional): Number of samples to include in each batch. Default is 4.
    - shuffle (bool, optional): Whether to shuffle the dataset after each epoch. Default is True.

    Returns:
    - DataLoader: A DataLoader object that provides batches of images and their corresponding annotations.
    """

    # Create the COCO dataset object by specifying the directory of images and the annotations file.
    # The transform applied is converting images to PyTorch tensors.
    dataset = CocoDetection(images_dir, annotations_file, transform=T.ToTensor())

    # Create a DataLoader to handle batching and shuffling of the dataset.
    # The collate_fn is defined to merge different samples into a single batch.
    # This is particularly useful when dealing with variable-length sequences or tuples.
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                             collate_fn=lambda x: tuple(zip(*x)))  # Unzips the dataset into a tuple of images and targets

    # Return the created DataLoader object.
    return data_loader


