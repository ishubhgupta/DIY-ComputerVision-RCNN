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
import torchvision.models as models  # Importing the models module from torchvision to access pre-trained models
import torch  # Importing the PyTorch library
import torch.nn as nn  # Importing the nn module to create neural network layers

# Define the device globally, selecting GPU if available, otherwise defaulting to CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class RCNN(nn.Module):
    def __init__(self, num_classes):
        """
        Initialize the RCNN model.
        
        Args:
            num_classes (int): The number of classes for the model (including background).
        """
        super(RCNN, self).__init__()  # Initialize the parent class
        # Load a pre-trained Faster R-CNN model with a ResNet-50 backbone and FPN (Feature Pyramid Network)
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Replace the pre-trained head of the model with a new predictor for the specified number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features  # Get the number of input features
        self.model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets):
        """
        Define the forward pass for the model.
        
        Args:
            images (List[Tensor]): A list of image tensors.
            targets (List[Dict]): A list of target dictionaries containing boxes and labels.
        
        Returns:
            The output of the model.
        """
        return self.model(images, targets)  # Call the model with images and targets

def create_model(num_classes):
    """Creates and returns a Faster R-CNN model instance."""
    return RCNN(num_classes).to(device)  # Initialize the model and move it to the appropriate device

def train_rcnn_model(model, train_loader, num_epochs):
    """
    Train the Faster R-CNN model.
    
    Args:
        model (nn.Module): The Faster R-CNN model to be trained.
        train_loader (DataLoader): DataLoader providing the training data.
        num_epochs (int): The number of training epochs.
    """
    model.train()  # Set the model to training mode
    # Define the optimizer (Stochastic Gradient Descent) with learning rate, momentum, and weight decay
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    # Learning rate scheduler to decrease learning rate after a certain number of epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Iterate over the number of epochs
    for epoch in range(num_epochs):
        total_loss = 0.0  # Initialize total loss for the epoch
        for images, targets in train_loader:  # Iterate through the batches of the training set
            images = [image.to(device) for image in images]  # Move images to the appropriate device

            # Debugging: Print targets to understand its structure
            print("Targets:", targets)

            # Prepare targets for the model
            formatted_targets = []  # Initialize a list to store formatted targets
            for t in targets:
                # Check if t is a list and has at least 2 elements
                if isinstance(t, (list, tuple)) and len(t) > 1:
                    annotations = t[1]  # Assuming second item contains annotations
                else:
                    print(f"Skipping target: expected a list/tuple with annotations but got: {t}")
                    continue  # Skip this target if it does not have the expected structure

                # Ensure annotations are in the correct list format
                if not isinstance(annotations, list):
                    print("Annotations are not in the expected list format:", annotations)
                    continue  # Skip if annotations are not in the correct format

                boxes = []  # List to store bounding boxes
                labels = []  # List to store labels
                for ann in annotations:
                    # Check if ann is a dictionary
                    if isinstance(ann, dict):
                        boxes.append(ann['bbox'])  # Add the bounding box
                        labels.append(ann['category_id'])  # Add the corresponding label
                    else:
                        print("Annotation is not a dictionary:", ann)

                # Convert lists of boxes and labels to tensors
                boxes = torch.tensor(boxes, dtype=torch.float32).to(device)
                labels = torch.tensor(labels, dtype=torch.int64).to(device)
                formatted_targets.append({'boxes': boxes, 'labels': labels})  # Append formatted targets

            # Ensure the number of images matches the number of targets
            if len(images) != len(formatted_targets):
                print(f"Skipping batch: images={len(images)}, targets={len(formatted_targets)}")
                continue  # Skip this batch if they don't match

            # Move targets to the appropriate device
            formatted_targets = [{k: v.to(device) for k, v in t.items()} for t in formatted_targets]

            # Compute loss
            loss_dict = model(images, formatted_targets)  # Forward pass with images and targets
            losses = sum(loss for loss in loss_dict.values())  # Sum up the losses

            optimizer.zero_grad()  # Zero the gradients
            losses.backward()  # Backpropagation
            optimizer.step()  # Update the model parameters

            total_loss += losses.item()  # Accumulate total loss

        lr_scheduler.step()  # Step the learning rate scheduler
        print(f"Epoch: {epoch + 1}, Loss: {total_loss:.4f}")  # Print epoch loss


def save_rcnn_model(model, path):
    """Saves the model state to the specified path."""
    torch.save(model.state_dict(), path)  # Save the model parameters to the given path

def load_rcnn_model(model, path):
    """Loads the model state from the specified path."""
    state_dict = torch.load(path, map_location=device)  # Load model parameters from the specified path
    model.load_state_dict(state_dict)  # Update model's parameters
    model.to(device)  # Move model to the appropriate device
    model.eval()  # Set the model to evaluation mode
    return model  # Return the loaded model

