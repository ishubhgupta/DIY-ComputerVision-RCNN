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
     
    # Description: This script provides prediction functionalities for all RCNN models. It includes functions for loading models, processing images, and making predictions with bounding boxes.

    # Dependency: 
        # Environment:     
            # Python 3.10.11
            # torch==2.5.0
            # torchvision==0.20.0 
            # opencv-python==4.8.0.74
            # numpy==1.24.3
            # Pillow==10.0.0

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
import cv2
import os
from torchvision.ops import nms
import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet50
from train_fast_rcnn import create_fast_rcnn_model as create_fast_rcnn
from train_faster_rcnn import create_model as create_faster_rcnn
from train_mask_rcnn import create_mask_rcnn_model as create_mask_rcnn, ResNetBackbone

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def create_rcnn(num_classes):
    # R-CNN specific setup (using pre-trained models or compatible structures)
    model = fasterrcnn_resnet50_fpn(pretrained=True)  # Placeholder for R-CNN architecture
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# General model loading function
def load_model(model_type, path, num_classes):
    if model_type == "Faster R-CNN":
        model = create_faster_rcnn(num_classes)
    elif model_type == "Fast R-CNN":
        model = create_fast_rcnn(num_classes)
    elif model_type == "R-CNN":
        model = create_rcnn(num_classes)
    elif model_type == "Mask R-CNN":
        model = create_mask_rcnn(num_classes)
    else:
        raise ValueError(f"Model type '{model_type}' is not supported.")
    
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model

# Function to load an image from path
def load_image(image_path):
    if not isinstance(image_path, str) or not os.path.isfile(image_path):
        raise ValueError("The provided image path is invalid or the file does not exist.")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from path: {image_path}")
    
    # Convert BGR (OpenCV format) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Preprocess the image for model input
def preprocess_image(image):
    transform = T.ToTensor()
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0).to(device)  # Add batch dimension

# Draw bounding box on the image for the highest score prediction
def draw_best_prediction(image, best_prediction):
    image_copy = image.copy()
    
    if best_prediction:
        box = best_prediction['bbox']
        x1, y1, x2, y2 = map(int, box)  # Ensure integer coordinates for OpenCV
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        cv2.putText(image_copy, f"Label: {best_prediction['label']}, Score: {best_prediction['score']:.2f}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return image_copy

# Make predictions on an image and return the image with boxes and predictions
def predict_on_image(model, image_path, nms_threshold=0.3):
    image = load_image(image_path)
    image_tensor = preprocess_image(image)

    with torch.no_grad():
        predictions = model(image_tensor)

    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    labels = predictions[0]['labels']

    # Apply Non-Maximum Suppression
    keep = nms(boxes, scores, nms_threshold)
    boxes_filtered = boxes[keep].cpu().numpy()
    scores_filtered = scores[keep].cpu().numpy()
    labels_filtered = labels[keep].cpu().numpy()

    if len(scores_filtered) == 0:
        return None, None

    # Find the best prediction based on the highest score
    highest_score_idx = scores_filtered.argmax()
    best_prediction = {
        'bbox': boxes_filtered[highest_score_idx],
        'score': scores_filtered[highest_score_idx],
        'label': labels_filtered[highest_score_idx]
    }

    # Draw the bounding box on the image
    image_with_box = draw_best_prediction(image, best_prediction)

    return image_with_box, best_prediction
