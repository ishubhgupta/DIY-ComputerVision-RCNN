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
import torchvision.models as models  # Import torchvision models for use in the model creation
import torch  # Import the PyTorch library
import torch.nn as nn  # Import the neural network module from PyTorch

# Define the device to be used for computations (GPU if available, otherwise CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Import Faster R-CNN model and its predictor from torchvision's detection module
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes):
    """
    Creates a Faster R-CNN model for object detection.

    Args:
        num_classes (int): The number of classes for the object detection task,
                           including the background class.

    Returns:
        model (nn.Module): A Faster R-CNN model with a modified head to fit the number of classes.
    """
    # Load a pre-trained Faster R-CNN model with a ResNet-50 backbone and Feature Pyramid Network (FPN)
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Get the number of input features for the classifier (the output features from the last convolutional layer)
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one that can classify the specified number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model  # Return the modified model

def train_model(model, train_loader, num_epochs):
    """
    Trains the Faster R-CNN model.

    Args:
        model (nn.Module): The Faster R-CNN model to be trained.
        train_loader (DataLoader): DataLoader providing the training dataset.
        num_epochs (int): The number of epochs to train the model for.

    Returns:
        None
    """
    model.train()  # Set the model to training mode
    # Define the optimizer using Stochastic Gradient Descent (SGD) with learning rate, momentum, and weight decay
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    # Set up a learning rate scheduler to adjust the learning rate every few epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Loop over the specified number of epochs
    for epoch in range(num_epochs):
        total_loss = 0.0  # Initialize the total loss for the current epoch

        # Loop over the training dataset provided by the DataLoader
        for images, targets in train_loader:
            # Move images to the specified device (GPU or CPU)
            images = [image.to(device) for image in images]

            # Prepare the target annotations for the model
            targets = []
            for t in targets:
                annotations = t[1]  # Extract the annotations for the current target
                boxes = []  # Initialize a list to store bounding boxes
                labels = []  # Initialize a list to store labels

                # Loop through the annotations to extract boxes and labels
                for ann in annotations:
                    boxes.append(ann['bbox'])  # Append the bounding box to the list
                    labels.append(ann['category_id'])  # Append the category ID to the labels list

                # Convert boxes and labels to tensors and move to the specified device
                boxes = torch.tensor(boxes, dtype=torch.float32).to(device)
                labels = torch.tensor(labels, dtype=torch.int64).to(device)

                # Append the dictionary of boxes and labels to the targets list
                targets.append({'boxes': boxes, 'labels': labels})

            # Skip the iteration if the number of images does not match the number of targets
            if len(images) != len(targets):
                continue

            # Move targets to the specified device
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Compute loss using the model
            loss_dict = model(images, targets)  # Get the loss dictionary from the model
            losses = sum(loss for loss in loss_dict.values())  # Sum the losses

            optimizer.zero_grad()  # Reset the gradients of the optimizer
            losses.backward()  # Backpropagate the losses
            optimizer.step()  # Update the model parameters

            total_loss += losses.item()  # Accumulate the total loss

        lr_scheduler.step()  # Step the learning rate scheduler
        # Print the total loss for the current epoch
        print(f"Epoch: {epoch + 1}, Loss: {total_loss:.4f}")

def save_model(model, path):
    """
    Saves the model's state to the specified path.

    Args:
        model (nn.Module): The model whose state will be saved.
        path (str): The file path where the model state will be saved.

    Returns:
        None
    """
    # Save the model's state dictionary (parameters) to the specified path
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """
    Loads the model's state from the specified path.

    Args:
        model (nn.Module): The model to load the state into.
        path (str): The file path from which to load the model state.

    Returns:
        model (nn.Module): The model with loaded state.
    """
    # Load the state dictionary from the specified path and map to the correct device
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)  # Load the state dictionary into the model
    model.to(device)  # Move the model to the specified device
    model.eval()  # Set the model to evaluation mode
    return model  # Return the model with loaded state



