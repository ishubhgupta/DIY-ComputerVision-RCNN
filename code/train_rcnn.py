import torchvision.models as models
import torch
import torch.nn as nn

# Define device globally
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class RCNN(nn.Module):
    def __init__(self, num_classes):
        super(RCNN, self).__init__()
        # Load a pre-trained Faster R-CNN model
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Replace the pre-trained head with a new one for your number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets):
        # Forward pass
        return self.model(images, targets)

def create_model(num_classes):
    """Creates a Faster R-CNN model."""
    return RCNN(num_classes).to(device)

def train_rcnn_model(model, train_loader, num_epochs):
    """Trains the Faster R-CNN model."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, targets in train_loader:
            images = [image.to(device) for image in images]

            # Debugging: Print targets to understand its structure
            print("Targets:", targets)

            # Prepare targets
            formatted_targets = []
            for t in targets:
                # Check if t is a list and has at least 2 elements
                if isinstance(t, (list, tuple)) and len(t) > 1:
                    annotations = t[1]  # Assuming second item contains annotations
                else:
                    print(f"Skipping target: expected a list/tuple with annotations but got: {t}")
                    continue  # Skip this target if it does not have the expected structure

                if not isinstance(annotations, list):
                    print("Annotations are not in the expected list format:", annotations)
                    continue  # Skip if annotations are not in the correct format

                boxes = []
                labels = []
                for ann in annotations:
                    # Check if ann is a dictionary
                    if isinstance(ann, dict):
                        boxes.append(ann['bbox'])
                        labels.append(ann['category_id'])
                    else:
                        print("Annotation is not a dictionary:", ann)

                boxes = torch.tensor(boxes, dtype=torch.float32).to(device)
                labels = torch.tensor(labels, dtype=torch.int64).to(device)
                formatted_targets.append({'boxes': boxes, 'labels': labels})

            if len(images) != len(formatted_targets):
                print(f"Skipping batch: images={len(images)}, targets={len(formatted_targets)}")
                continue

            # Move targets to device
            formatted_targets = [{k: v.to(device) for k, v in t.items()} for t in formatted_targets]

            # Compute loss
            loss_dict = model(images, formatted_targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        lr_scheduler.step()
        print(f"Epoch: {epoch + 1}, Loss: {total_loss:.4f}")


def save_rcnn_model(model, path):
    """Saves the model state to the specified path."""
    torch.save(model.state_dict(), path)

def load_rcnn_model(model, path):
    """Loads the model state from the specified path."""
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model


