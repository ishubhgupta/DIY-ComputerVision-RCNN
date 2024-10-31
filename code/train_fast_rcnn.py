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
# Python 3.11.5
# Streamlit 1.36.0
# Import necessary libraries from PyTorch and torchvision
import torchvision.models as models  # For pre-trained models
import torch  # Core library for PyTorch
import torch.nn as nn  # Neural network module
from torchvision.models.detection import fasterrcnn_resnet50_fpn  # Import Faster R-CNN model architecture
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # Import Fast R-CNN predictor head

# Define the device to be used for training or inference (GPU if available, otherwise CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def create_fast_rcnn_model(num_classes):
    """
    Creates a Fast R-CNN model.

    Parameters:
    - num_classes (int): The number of classes for the model (including background).

    Returns:
    - model: The constructed Fast R-CNN model.
    """
    # Load a pre-trained Faster R-CNN model with a ResNet-50 backbone and Feature Pyramid Network (FPN)
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Get the number of input features from the pre-trained box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new predictor for the specified number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model  # Return the configured model

def train_fast_rcnn_model(model, train_loader, num_epochs):
    """
    Trains the Fast R-CNN model.

    Parameters:
    - model: The Fast R-CNN model to train.
    - train_loader: DataLoader containing the training data.
    - num_epochs (int): The number of epochs to train for.
    """
    model.train()  # Set the model to training mode

    # Define the optimizer using Stochastic Gradient Descent (SGD) with specified learning rate and momentum
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Define a learning rate scheduler to reduce the learning rate at specified intervals
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Iterate through the specified number of epochs
    for epoch in range(num_epochs):
        total_loss = 0.0  # Initialize total loss for the epoch

        # Iterate through the batches in the training DataLoader
        for images, targets in train_loader:
            # Move images to the specified device (GPU/CPU)
            images = [image.to(device) for image in images]

            # Prepare targets for training
            targets = []
            for t in targets:
                annotations = t[1]  # Get the annotations from the target
                boxes = []  # Initialize a list to store bounding boxes
                labels = []  # Initialize a list to store class labels
                for ann in annotations:
                    # Append the bounding box coordinates and category ID to respective lists
                    boxes.append(ann['bbox'])
                    labels.append(ann['category_id'])

                # Convert boxes and labels to tensors and move to device
                boxes = torch.tensor(boxes, dtype=torch.float32).to(device)
                labels = torch.tensor(labels, dtype=torch.int64).to(device)

                # Create a target dictionary for the current image
                targets.append({'boxes': boxes, 'labels': labels})

            # Skip this iteration if the number of images and targets does not match
            if len(images) != len(targets):
                continue

            # Move the targets to the specified device
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Compute the loss by passing images and targets to the model
            loss_dict = model(images, targets)
            # Sum the losses from the loss dictionary
            losses = sum(loss for loss in loss_dict.values())

            # Zero the gradients of the optimizer
            optimizer.zero_grad()
            # Backpropagate the losses to compute gradients
            losses.backward()
            # Update model weights based on the gradients
            optimizer.step()

            # Accumulate the total loss for the epoch
            total_loss += losses.item()

        # Step the learning rate scheduler at the end of the epoch
        lr_scheduler.step()
        # Print the epoch number and the total loss
        print(f"Epoch: {epoch + 1}, Loss: {total_loss:.4f}")

def save_fast_rcnn_model(model, path):
    """
    Saves the model state to the specified path.

    Parameters:
    - model: The Fast R-CNN model to save.
    - path (str): The path where the model will be saved.
    """
    # Save the state dictionary of the model to the specified file path
    torch.save(model.state_dict(), path)

def load_fast_rcnn_model(model, path):
    """
    Loads the model state from the specified path.

    Parameters:
    - model: The Fast R-CNN model to load weights into.
    - path (str): The path from which to load the model weights.

    Returns:
    - model: The model with loaded weights, set to evaluation mode.
    """
    # Load the state dictionary from the specified file path
    model.load_state_dict(torch.load(path, map_location=device))
    # Move the model to the specified device
    model.to(device)
    # Set the model to evaluation mode to disable dropout and batch normalization
    model.eval()  
    return model  # Return the model with loaded weights
