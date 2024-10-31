import streamlit as st
import torchvision.transforms as transforms
from PIL import Image
from ingest_transform import get_data_loader
from train_faster_rcnn import create_model as create_faster_rcnn_model, train_model, save_model, device
from train_fast_rcnn import create_fast_rcnn_model, train_fast_rcnn_model, load_fast_rcnn_model
from train_rcnn import RCNN, train_rcnn_model, save_rcnn_model, load_rcnn_model
from train_mask_rcnn import create_mask_rcnn_model, train_mask_rcnn_model, save_mask_rcnn_model, load_mask_rcnn_model
from predictor import predict_on_image, load_model

# Streamlit title
st.title("Damage Detection with R-CNN Models")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((300, 300)),  # Resize to a fixed size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Create tabs for separate sections
tab1, tab2, tab3 = st.tabs(["Input Data Paths", "Train Models", "Make Prediction"])

# Tab 1: Input Data Paths
with tab1:
    st.header("Input Data Paths")
    
    TRAIN_ANNO_PATH = r'Data\Master\train\COCO_train_annos.json'
    TRAIN_IMAGES_PATH = r'Data\Master\train'
    train_anno_path = st.text_input("Path to Training Annotations", value=TRAIN_ANNO_PATH)
    train_images_path = st.text_input("Path to Training Images", value=TRAIN_IMAGES_PATH)

# Tab 2: Train Models
with tab2:
    st.header("Model Training")
    
    if train_anno_path and train_images_path:
        # Load data
        train_loader = get_data_loader(train_anno_path, train_images_path)
        num_classes = 2  # 1 class (damage) + background
        
        # User selects number of epochs
        num_epochs = st.number_input("Enter number of epochs:", min_value=1, value=2)

        # Train Faster R-CNN
        if st.button("Train Faster R-CNN"):
            model = create_faster_rcnn_model(num_classes).to(device)
            train_model(model, train_loader, num_epochs)
            save_model(model, 'code/saved_models/faster_rcnn_damage_detection.pth')
            st.success("Faster R-CNN model trained and saved successfully!")
        
        # Train Fast R-CNN
        if st.button("Train Fast R-CNN"):
            model = create_fast_rcnn_model(num_classes).to(device)
            train_fast_rcnn_model(model, train_loader, num_epochs)
            save_model(model, 'code/saved_models/fast_rcnn_damage_detection.pth')
            st.success("Fast R-CNN model trained and saved successfully!")
        
        # Train R-CNN
        if st.button("Train R-CNN"):
            model = RCNN(num_classes).to(device)
            train_rcnn_model(model, train_loader, num_epochs)
            save_rcnn_model(model, 'code/saved_models/rcnn_damage_detection.pth')
            st.success("R-CNN model trained and saved successfully!")
        
        # Train Mask R-CNN
        if st.button("Train Mask R-CNN"):
            model = create_mask_rcnn_model(num_classes).to(device)
            train_mask_rcnn_model(model, train_loader, num_epochs)
            save_mask_rcnn_model(model, 'code/saved_models/mask_rcnn_damage_detection.pth')
            st.success("Mask R-CNN model trained and saved successfully!")
    
    else:
        st.error("Please provide valid paths for annotations and images in Tab 1.")


# Tab 3: Make Prediction
with tab3:
    st.header("Make a Prediction")
    
    # Model selection dropdown
    model_type = st.selectbox("Select the model to use for prediction", ("Faster R-CNN", "Fast R-CNN", "R-CNN", "Mask R-CNN"))
    image_path = st.text_input("Enter the path to an image for prediction:")
    
    # Define a mapping for model types to their corresponding .pth file paths
    model_path_mapping = {
        "Faster R-CNN": "code/saved_models/faster_rcnn_damage_detection.pth",
        "Fast R-CNN": "code/saved_models/fast_rcnn_damage_detection.pth",
        "R-CNN": "code/saved_models/rcnn_damage_detection.pth",
        "Mask R-CNN": "code/saved_models/mask_rcnn_damage_detection.pth"
    }
    
    if st.button("Predict"):
        if image_path:
            # Load the chosen model based on user selection
            num_classes = 2  # Replace with actual number of classes for your model
            model_path = model_path_mapping.get(model_type)
            
            if model_path is None:
                st.error("Selected model type does not have a corresponding path.")
            else:
                try:
                    model = load_model(model_type, model_path, num_classes)
                    image_with_box, best_prediction = predict_on_image(model, image_path)
                    
                    # Display predictions
                    if best_prediction:
                        st.image(image_with_box, caption='Predicted Image with Bounding Box', use_column_width=True)
                        st.write("Prediction:")
                        st.write(f"Label: {best_prediction['label']}, Score: {best_prediction['score']:.2f}, BBox: {best_prediction['bbox']}")
                    else:
                        st.warning("No predictions made.")
                except ValueError as e:
                    st.error(str(e))
        else:
            st.error("Please provide a valid path for the image.")

