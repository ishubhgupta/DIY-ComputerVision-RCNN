import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet50

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ResNetBackbone(torch.nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        # Load a pre-trained ResNet50 model
        backbone = resnet50(pretrained=True)
        # Remove the last two layers (fully connected layers)
        self.body = torch.nn.Sequential(*(list(backbone.children())[:-2]))
        # The output channels from the last conv layer
        self.out_channels = 2048  # This should match the output of ResNet50's last layer

    def forward(self, x):
        return self.body(x)

def create_mask_rcnn_model(num_classes):
    # Create the custom backbone
    backbone = ResNetBackbone()

    # Create the anchor generator
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),) * 5)

    # Create the Mask R-CNN model
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_anchor_generator=anchor_generator)

    return model

def train_mask_rcnn_model(model, train_loader, num_epochs):
    """Trains the Mask R-CNN model."""
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
                masks = []
                for ann in annotations:
                    boxes.append(ann['bbox'])
                    labels.append(ann['category_id'])
                    masks.append(ann['segmentation'])  # Assuming segmentation masks are provided

                boxes = torch.tensor(boxes, dtype=torch.float32).to(device)
                labels = torch.tensor(labels, dtype=torch.int64).to(device)

                # Convert masks to a tensor format (binary)
                masks = [torch.tensor(mask, dtype=torch.uint8).unsqueeze(0) for mask in masks]
                masks = torch.cat(masks).to(device) if masks else torch.empty((0, 0, 0), dtype=torch.uint8)

                targets.append({'boxes': boxes, 'labels': labels, 'masks': masks})

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

def save_mask_rcnn_model(model, path):
    """Saves the model state to the specified path."""
    torch.save(model.state_dict(), path)

def load_mask_rcnn_model(model, path):
    """Loads the model state from the specified path."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def run_inference(model, images):
    """Runs inference on a batch of images and returns predictions."""
    model.eval()
    with torch.no_grad():
        predictions = model(images)
    return predictions
