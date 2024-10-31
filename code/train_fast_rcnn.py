import torchvision.models as models
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Define device globally
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def create_fast_rcnn_model(num_classes):
    """Creates a Fast R-CNN model."""
    # Load a pre-trained Faster R-CNN model (can be used for Fast R-CNN too)
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace the pre-trained head with a new one for your number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model  # Ensure this returns the model object

def train_fast_rcnn_model(model, train_loader, num_epochs):
    """Trains the Fast R-CNN model."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, targets in train_loader:
            images = [image.to(device) for image in images]

            # Prepare targets
            targets = []
            for t in targets:
                annotations = t[1]
                boxes = []
                labels = []
                for ann in annotations:
                    boxes.append(ann['bbox'])
                    labels.append(ann['category_id'])

                boxes = torch.tensor(boxes, dtype=torch.float32).to(device)
                labels = torch.tensor(labels, dtype=torch.int64).to(device)
                targets.append({'boxes': boxes, 'labels': labels})

            if len(images) != len(targets):
                continue

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Compute loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        lr_scheduler.step()
        print(f"Epoch: {epoch + 1}, Loss: {total_loss:.4f}")

def save_fast_rcnn_model(model, path):
    """Saves the model state to the specified path."""
    torch.save(model.state_dict(), path)

def load_fast_rcnn_model(model, path):
    """Loads the model state from the specified path."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model
