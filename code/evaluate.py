# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Shubh Gupta and Rupal Mishra
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (02 November 2024)
            # Developers: Shubh Gupta and Rupal Mishra
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This Streamlit app allows users to input image and make predictions using RCNN models.

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.10.11
        # Streamlit 1.40.0

import os  # Importing the os module for file system operations
import streamlit as st  # Importing Streamlit for creating web applications
import torch  # Importing PyTorch for deep learning operations
from ingest_transform import get_data_loader  # Importing function to get data loader
from predictor import load_model  # Importing function to load model
from train_faster_rcnn import device  # Importing device from train_faster_rcnn module

def evaluate(eval_model_type: str, test_anno_path: str, test_images_path: str) -> None:
    """
    Function to evaluate the performance of a given model on a test dataset.

    Parameters:
    - eval_model_type (str): The type of the model to be evaluated.
    - test_anno_path (str): The path to the test annotations file.
    - test_images_path (str): The path to the directory containing test images.

    Returns:
    - None: The function does not return any value, it only displays the evaluation results.
    """
    try:
        # Validate input paths
        validate_paths(test_anno_path, test_images_path)

        # Load test data
        test_loader = get_data_loader(test_anno_path, test_images_path)
        if not test_loader:
            raise ValueError("Failed to create test data loader")

        # Load the selected model
        model = load_selected_model(eval_model_type)
        
        model.eval()  # Set model to evaluation mode
        
        # Calculate metrics
        total_loss, correct_detections, total_detections = calculate_metrics(test_loader, model)
        
        # Calculate and display metrics
        display_results(total_loss, correct_detections, total_detections, len(test_loader))
        
    except (FileNotFoundError, ValueError) as e:
        st.error(f"{type(e).__name__}: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error during evaluation: {str(e)}")
        st.write("Full error details:", e)

def validate_paths(test_anno_path: str, test_images_path: str) -> None:
    """
    Function to validate the paths to the test annotations file and the directory containing test images.

    Parameters:
    - test_anno_path (str): The path to the test annotations file.
    - test_images_path (str): The path to the directory containing test images.

    Returns:
    - None: The function does not return any value, it only raises an error if the paths are invalid.
    """
    if not os.path.exists(test_anno_path):
        raise FileNotFoundError(f"Test annotations file not found at: {test_anno_path}")
    if not os.path.exists(test_images_path):
        raise FileNotFoundError(f"Test images directory not found at: {test_images_path}")

def load_selected_model(eval_model_type: str) -> torch.nn.Module:
    """
    Function to load the selected model based on the model type.

    Parameters:
    - eval_model_type (str): The type of the model to be loaded.

    Returns:
    - torch.nn.Module: The loaded model.
    """
    model_path_mapping = {
        "Faster R-CNN": "code/saved_models/faster_rcnn_damage_detection.pth",
        "Fast R-CNN": "code/saved_models/fast_rcnn_damage_detection.pth", 
        "R-CNN": "code/saved_models/rcnn_damage_detection.pth",
        "Mask R-CNN": "code/saved_models/mask_rcnn_damage_detection.pth"
    }
    
    model_path = model_path_mapping.get(eval_model_type)
    if not model_path or not os.path.exists(model_path):
        raise ValueError(f"Invalid or missing model file for type: {eval_model_type}")

    return load_model(eval_model_type, model_path, num_classes=2)

def calculate_metrics(test_loader, model) -> tuple:
    """
    Function to calculate the evaluation metrics for a given model on a test dataset.

    Parameters:
    - test_loader: The DataLoader object that provides batches of images and their corresponding annotations.
    - model: The model to be evaluated.

    Returns:
    - tuple: A tuple containing the total loss, the number of correct detections, and the total number of detections.
    """
    total_loss = 0.0
    correct_detections = 0
    total_detections = 0
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            if not images or not targets:
                st.warning(f"Skipping empty batch {batch_idx}")
                continue
            
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets if isinstance(t, dict)]
            if not targets:
                continue
            
            # Get model predictions
            loss_dict = model(images, targets)
            batch_loss = sum(loss.item() for loss in loss_dict.values()) if loss_dict else 0
            total_loss += batch_loss
            
            # Calculate accuracy metrics
            predictions = model(images)
            for pred_idx, (pred, target) in enumerate(zip(predictions, targets)):
                correct_detections += calculate_iou(pred, target)
                total_detections += len(target.get('boxes', []))
    
    return total_loss, correct_detections, total_detections

def calculate_iou(pred, target) -> int:
    """
    Function to calculate the Intersection over Union (IoU) between predicted and true boxes.

    Parameters:
    - pred: The predicted boxes.
    - target: The true boxes.

    Returns:
    - int: The number of matches.
    """
    pred_boxes = pred.get('boxes', torch.tensor([]).to(device))
    true_boxes = target.get('boxes', torch.tensor([]).to(device))
    
    if len(pred_boxes) > 0 and len(true_boxes) > 0:
        # Calculate IoU between predicted and true boxes
        area1 = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        area2 = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
        
        lt = torch.max(pred_boxes[:, None, :2], true_boxes[:, :2])
        rb = torch.min(pred_boxes[:, None, 2:], true_boxes[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        iou = inter / (union + 1e-6)
        matches = torch.sum(iou.max(dim=1)[0] > 0.5).item()
        return matches
    return 0

def display_results(total_loss: float, correct_detections: int, total_detections: int, num_batches: int) -> None:
    """
    Function to display the evaluation results.

    Parameters:
    - total_loss (float): The total loss.
    - correct_detections (int): The number of correct detections.
    - total_detections (int): The total number of detections.
    - num_batches (int): The total number of batches processed.
    """
    if total_detections < 0:
        raise ValueError("No valid detections found in the test dataset.")
        
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    detection_accuracy = correct_detections / total_detections if total_detections > 0 else 0
    
    # Display results
    st.subheader("Evaluation Results")
    st.write(f"Total batches processed: {num_batches}")
    st.write(f"Total detections: {total_detections}")
    st.write(f"Correct detections: {correct_detections}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Loss", f"{avg_loss:.4f}")
    with col2:
        st.metric("Detection Accuracy", f"{detection_accuracy:.2%}")
    
    # Additional metrics visualization
    st.subheader("Performance Visualization")
    data = {
        'Metric': ['Loss', 'Accuracy'],
        'Value': [float(avg_loss), float(detection_accuracy)]
    }
    st.bar_chart(data, x='Metric', y='Value')