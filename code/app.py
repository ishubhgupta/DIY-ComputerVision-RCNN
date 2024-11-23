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
     
    # Description: This is the main application script that provides a Streamlit web interface for training, evaluating and making predictions with various RCNN models for car damage detection.

    # Dependency: 
        # Environment:     
            # Python 3.10.11
            # streamlit==1.40.0
            # torch==2.5.0
            # torchvision==0.20.0
            # Pillow==10.0.0
            # numpy==1.24.3
            # opencv-python==4.8.0.74
            # pycocotools==2.0.8
            # runpandas==1.5.3
            # joblib==1.3.1

import os
import streamlit as st  # Importing Streamlit for creating the web app interface
import torch
import torchvision.transforms as transforms  # Importing transformations for image preprocessing
from PIL import Image  # Importing Image for image manipulation
from evaluate import evaluate as evc
from ingest_transform import get_data_loader  # Importing the function to load the COCO dataset
from train_faster_rcnn import create_model as create_faster_rcnn_model, train_model, save_model, device  # Importing functions for training Faster R-CNN
from train_fast_rcnn import create_fast_rcnn_model, train_fast_rcnn_model, load_fast_rcnn_model  # Importing functions for training Fast R-CNN
from train_rcnn import RCNN, train_rcnn_model, save_rcnn_model, load_rcnn_model  # Importing R-CNN training functions
from train_mask_rcnn import create_mask_rcnn_model, train_mask_rcnn_model, save_mask_rcnn_model, load_mask_rcnn_model  # Importing Mask R-CNN training functions
from predictor import predict_on_image, load_model  # Importing prediction functions

# Streamlit title for the web application
st.title("Car Damage Detection")

# Define image transformations to be applied to the input images
transform = transforms.Compose([
    transforms.Resize((300, 300)),  # Resize images to a fixed size of 300x300 pixels
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images using ImageNet statistics
])

# Create tabs in the Streamlit app for organizing different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["Input Data Paths", "Train Models", "Model Evaluation", "Make Prediction"])

# Tab 1: Input Data Paths
with tab1:
    st.header("Input Data Paths")  # Header for the first tab
    
    # Define default paths for training annotations and images
    TRAIN_ANNO_PATH = r'Data\Master\train\COCO_train_annos.json'  # Default path for training annotations
    TRAIN_IMAGES_PATH = r'Data\Master\train'  # Default path for training images
    
    # Create text input fields for users to enter paths to training annotations and images
    train_anno_path = st.text_input("Path to Training Annotations", value=TRAIN_ANNO_PATH)
    train_images_path = st.text_input("Path to Training Images", value=TRAIN_IMAGES_PATH)
# Tab 2: Train Models
with tab2:
    st.header("Model Training")  # Header for the model training tab
    
    # Check if valid paths for training annotations and images are provided
    if train_anno_path and train_images_path:
        # Load training data using the get_data_loader function
        train_loader = get_data_loader(train_anno_path, train_images_path)
        num_classes = 2  # Define the number of classes (damage and background)

        # Train Faster R-CNN
        st.subheader("Train Faster R-CNN")  # Subheader for Faster R-CNN training
        num_epochs_faster_rcnn = st.number_input("Enter number of epochs for Faster R-CNN:", min_value=1, value=10)  # User input for number of epochs
        if st.button("Train Faster R-CNN"):  # Button to trigger training
            model = create_faster_rcnn_model(num_classes).to(device)  # Create and move the model to the specified device (GPU/CPU)
            train_model(model, train_loader, num_epochs_faster_rcnn)  # Train the model with the data loader
            save_model(model, 'code/saved_models/faster_rcnn_damage_detection.pth')  # Save the trained model
            st.success("Faster R-CNN model trained and saved successfully!")  # Success message
        
        st.markdown("---")  # Divider

        # Train Fast R-CNN
        st.subheader("Train Fast R-CNN")  # Subheader for Fast R-CNN training
        num_epochs_fast_rcnn = st.number_input("Enter number of epochs for Fast R-CNN:", min_value=1, value=10)  # User input for number of epochs
        if st.button("Train Fast R-CNN"):  # Button to trigger training
            model = create_fast_rcnn_model(num_classes).to(device)  # Create and move the model to the specified device
            train_fast_rcnn_model(model, train_loader, num_epochs_fast_rcnn)  # Train the model with the data loader
            save_model(model, 'code/saved_models/fast_rcnn_damage_detection.pth')  # Save the trained model
            st.success("Fast R-CNN model trained and saved successfully!")  # Success message
        
        st.markdown("---")  # Divider

        # Train R-CNN
        st.subheader("Train R-CNN")  # Subheader for R-CNN training
        num_epochs_rcnn = st.number_input("Enter number of epochs for R-CNN:", min_value=1, value=10)  # User input for number of epochs
        if st.button("Train R-CNN"):  # Button to trigger training
            model = RCNN(num_classes).to(device)  # Create and move the R-CNN model to the specified device
            train_rcnn_model(model, train_loader, num_epochs_rcnn)  # Train the model with the data loader
            save_rcnn_model(model, 'code/saved_models/rcnn_damage_detection.pth')  # Save the trained model
            st.success("R-CNN model trained and saved successfully!")  # Success message

        st.markdown("---")  # Divider

        # Train Mask R-CNN
        st.subheader("Train Mask R-CNN")  # Subheader for Mask R-CNN training
        num_epochs_mask_rcnn = st.number_input("Enter number of epochs for Mask R-CNN:", min_value=1, value=10)  # User input for number of epochs
        if st.button("Train Mask R-CNN"):  # Button to trigger training
            model = create_mask_rcnn_model(num_classes).to(device)  # Create and move the Mask R-CNN model to the specified device
            train_mask_rcnn_model(model, train_loader, num_epochs_mask_rcnn)  # Train the model with the data loader
            save_mask_rcnn_model(model, 'code/saved_models/mask_rcnn_damage_detection.pth')  # Save the trained model
            st.success("Mask R-CNN model trained and saved successfully!")  # Success message
    
    else:
        # Error message if paths are not provided
        st.error("Please provide valid paths for annotations and images in Tab 1.")

# Tab 3: Evaluation
with tab3:
    st.header("Model Evaluation")  # Header for the evaluation tab
    
    # Model selection for evaluation
    eval_model_type = st.selectbox("Select model for evaluation", 
                                  ("Faster R-CNN", "Fast R-CNN", "R-CNN", "Mask R-CNN"),
                                  key="eval_model")
    
    # Input fields for test data
    test_anno_path = st.text_input("Enter path to test annotations:")
    test_images_path = st.text_input("Enter path to test images:")
    
    if st.button("Evaluate Model"):
        if test_anno_path and test_images_path and os.path.exists(test_anno_path) and os.path.exists(test_images_path):
            evc(eval_model_type, test_anno_path, test_images_path)
        else:
            st.error("Please provide valid paths for test annotations and images.")


# Tab 4: Make Prediction
with tab4:
    st.header("Make a Prediction")  # Header for the prediction tab
    
    # Model selection dropdown for choosing which model to use for prediction
    model_type = st.selectbox("Select the model to use for prediction", ("Faster R-CNN", "Fast R-CNN", "R-CNN", "Mask R-CNN"))
    
    # Input field for the user to enter the path to an image for prediction
    image_path = st.text_input("Enter the path to an image for prediction:")
    
    # Define a mapping for model types to their corresponding saved model paths
    model_path_mapping = {
        "Faster R-CNN": "code/saved_models/faster_rcnn_damage_detection.pth",
        "Fast R-CNN": "code/saved_models/fast_rcnn_damage_detection.pth",
        "R-CNN": "code/saved_models/rcnn_damage_detection.pth",
        "Mask R-CNN": "code/saved_models/mask_rcnn_damage_detection.pth"
    }
    
    if st.button("Predict"):  # Button to trigger prediction
        if image_path:  # Check if the image path is provided
            # Load the chosen model based on user selection
            num_classes = 2  # Replace with the actual number of classes for your model
            model_path = model_path_mapping.get(model_type)  # Get the corresponding model path
            
            if model_path is None:  # Check if the model path is valid
                st.error("Selected model type does not have a corresponding path.")
            else:
                try:
                    # Load the model and make predictions on the specified image
                    model = load_model(model_type, model_path, num_classes)
                    image_with_box, best_prediction = predict_on_image(model, image_path)  # Run prediction on the image
                    
                    # Display predictions
                    if best_prediction:  # Check if any predictions were made
                        st.image(image_with_box, caption='Predicted Image with Bounding Box', use_container_width=True)  # Show the image with predictions
                        st.write("Prediction:")  # Output section for prediction details
                        st.write(f"Label: {best_prediction['label']}, Score: {best_prediction['score']:.2f}, BBox: {best_prediction['bbox']}")  # Display label, score, and bounding box
                    else:
                        st.warning("No predictions made.")  # Warning if no predictions are available
                except ValueError as e:  # Handle any errors that occur during prediction
                    st.error(str(e))
        else:
            # Error message if no image path is provided
            st.error("Please provide a valid path for the image.")


