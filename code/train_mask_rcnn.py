# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Shubh Gupta
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (02 November 2024)
            # Developers: Shubh Gupta
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This script implements the Mask R-CNN model architecture and training pipeline. It includes custom backbone implementation and functions for model creation, training, saving, and loading.

    # Dependency: 
        # Environment:     
            # Python 3.10.11
            # torch==2.5.0
            # torchvision==0.20.0
            # numpy==1.24.3

import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet50

# Set the device for computation. Use GPU if available, otherwise use CPU.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ResNetBackbone(torch.nn.Module):
    """
    Custom ResNet backbone for the Mask R-CNN model.
    It uses the ResNet-50 architecture and modifies it to be compatible with Mask R-CNN.
    """
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        # Load a pre-trained ResNet50 model from torchvision
        backbone = resnet50(pretrained=True)
        # Remove the last two layers (fully connected layers) to get the feature extractor
        self.body = torch.nn.Sequential(*(list(backbone.children())[:-2]))
        # The output channels from the last convolutional layer of ResNet-50
        self.out_channels = 2048  # This matches the output of ResNet50's last layer

    def forward(self, x):
        # Forward pass through the modified ResNet backbone
        return self.body(x)

def create_mask_rcnn_model(num_classes):
    """
    Create a Mask R-CNN model with a custom ResNet backbone.

    Args:
        num_classes (int): Number of classes for the Mask R-CNN model (including background).

    Returns:
        MaskRCNN: Configured Mask R-CNN model.
    """
    # Instantiate the custom ResNet backbone
    backbone = ResNetBackbone()

    # Create the anchor generator for the region proposal network
    # 'sizes' define the scale of anchors and 'aspect_ratios' define the width/height ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),) * 5)

    # Create the Mask R-CNN model using the backbone and anchor generator
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_anchor_generator=anchor_generator)

    return model  # Return the configured Mask R-CNN model

def train_mask_rcnn_model(model, train_loader, num_epochs):
    """
    Train the Mask R-CNN model.

    Args:
        model (MaskRCNN): The Mask R-CNN model to be trained.
        train_loader (DataLoader): DataLoader providing training data.
        num_epochs (int): Number of training epochs.
    """
    model.train()  # Set the model to training mode
    # Define the optimizer (Stochastic Gradient Descent)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    # Learning rate scheduler to adjust the learning rate during training
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        total_loss = 0.0  # Initialize total loss for the epoch
        for images, targets in train_loader:
            images = [image.to(device) for image in images]  # Move images to the specified device

            # Prepare targets for the model
            targets = []
            for t in targets:
                annotations = t[1]  # Get annotations from targets
                boxes = []  # Initialize a list to store bounding boxes
                labels = []  # Initialize a list to store labels
                masks = []   # Initialize a list to store segmentation masks
                for ann in annotations:
                    boxes.append(ann['bbox'])  # Extract bounding box
                    labels.append(ann['category_id'])  # Extract category label
                    masks.append(ann['segmentation'])  # Extract segmentation mask

                # Convert lists to tensors and move them to the specified device
                boxes = torch.tensor(boxes, dtype=torch.float32).to(device)
                labels = torch.tensor(labels, dtype=torch.int64).to(device)

                # Convert masks to a tensor format (binary) with an additional dimension for channels
                masks = [torch.tensor(mask, dtype=torch.uint8).unsqueeze(0) for mask in masks]
                # Concatenate masks into a single tensor or create an empty tensor if there are no masks
                masks = torch.cat(masks).to(device) if masks else torch.empty((0, 0, 0), dtype=torch.uint8)

                # Append prepared target dictionary to the targets list
                targets.append({'boxes': boxes, 'labels': labels, 'masks': masks})

            # Check if the number of images matches the number of targets
            if len(images) != len(targets):
                continue  # Skip if they do not match

            # Move all targets to the specified device
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Compute the loss from the model
            loss_dict = model(images, targets)
            # Sum the losses from the loss dictionary
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()  # Zero the gradients
            losses.backward()  # Backpropagate the losses
            optimizer.step()  # Update model parameters

            total_loss += losses.item()  # Accumulate total loss

        lr_scheduler.step()  # Step the learning rate scheduler
        print(f"Epoch: {epoch + 1}, Loss: {total_loss:.4f}")  # Print epoch loss

def save_mask_rcnn_model(model, path):
    """Saves the model state to the specified path."""
    torch.save(model.state_dict(), path)  # Save the model state dictionary

def load_mask_rcnn_model(model, path):
    """Loads the model state from the specified path."""
    model.load_state_dict(torch.load(path, map_location=device))  # Load the model state
    model.to(device)  # Move model to the specified device
    model.eval()  # Set the model to evaluation mode
    return model  # Return the loaded model

def run_inference(model, images):
    """
    Runs inference on a batch of images and returns predictions.

    Args:
        model (MaskRCNN): The trained Mask R-CNN model.
        images (list of Tensor): List of input images.

    Returns:
        list: Predictions containing boxes, labels, and masks for each image.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        predictions = model(images)  # Get predictions from the model
    return predictions  # Return the predictions
