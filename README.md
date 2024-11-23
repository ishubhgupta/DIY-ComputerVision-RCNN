# DIY-Deep-Learning-NN-PyTorch

This is the **Vehicle Damage Detection** branch.

## Vehicle Damage Detection

### Business Case

Vehicle damage detection is an essential process in the automotive industry, particularly for insurance companies, repair shops, and car rental services. With increasing vehicle usage and the likelihood of accidents, having an accurate and efficient method to assess vehicle damage can significantly enhance operational efficiency, improve customer service, and reduce costs.

This project aims to develop a deep learning model that can accurately classify various types of vehicle damage from images. By automating the assessment process, we can provide quick and reliable evaluations, streamline claims processing for insurers, and assist repair shops in diagnosing damage more effectively. The project will also empower vehicle owners by giving them a clear understanding of the extent of the damage, helping them make informed decisions regarding repairs.

### Industry

Automotive / Insurance / Vehicle Repair

### Problem Statement

Assessing vehicle damage is often a manual and time-consuming process. This can lead to inconsistencies in damage reporting, delays in claims processing, and potential disputes between insurance companies and vehicle owners. Furthermore, novice assessors may struggle with accurately identifying the type and severity of damage, leading to inadequate or incorrect assessments.

To address these challenges, there is a pressing need for a user-friendly solution that utilizes technology to assist in identifying vehicle damage quickly and accurately. Developing a machine learning model to classify vehicle damage based on images will provide immediate support for assessors, enhance customer experience, and contribute to improved operational efficiency.

### Objective

The aim is to build a predictive model that can accurately classify vehicle damage based on images of damaged vehicles. The model will focus on features such as:

* Damage Type: Categories such as scratches, dents, or structural damage.
* Severity Level: An assessment of the damage severity (e.g., minor, moderate, severe).
* Image Data: Images of vehicles showing various types of damage for classification.

By employing this classification model, stakeholders can:

* Quickly identify the type and severity of vehicle damage.
* Streamline the claims process for insurers.
* Enhance customer satisfaction by providing accurate assessments.

- - -

## Important Notes

> ⚠️ **Learning Purpose Only**
> 
> This implementation currently achieves lower accuracy due to:
> - Limited training epochs (default: 10)
> - Small dataset size
> - Basic model configurations
>
> This project is primarily designed for learning and understanding RCNN architectures.

## Directory Structure

<details>
<summary>Click to expand directory tree</summary>

``` plaintext
code/
├── __pycache__/                   (directory for compiled Python files)
├── saved_model/                   (directory for saved model files and training scripts)
│   ├── app.py                     (main application file for the Streamlit web app)
│   ├── predictor.py          (script for classification-related functions and utilities)
│   ├── train_rcnn.py                (script to evaluate model performance on test data)
│   ├── train_fast_rcnn.py (script for ingesting and transforming data into CouchDB)
│   ├── train_faster_rcnn.py (script for ingesting and transforming data into CouchDB)
│   ├── train_mask_rcnn.py (script for ingesting and transforming data into CouchDB)
│   ├── ingest_transform.py        (script for general data ingestion and transformation)
└── Data/
    └── Master/
        └── Dataset                (directory containing vehicle image datasets)
.gitattributes                       (file for managing Git attributes)
.gitignore                          (specifies files and directories to be ignored by Git)
readme.md                           (documentation for the project)
requirements.txt                   (lists the dependencies required for the project)
```
</details>

## Data Definition

The dataset contains features related to various types of vehicle damage, including:

* **Damage Type:** Categories such as scratches, dents, or structural damage.
* **Severity Level:** An assessment of the damage severity (e.g., minor, moderate, severe).
* **Image Data:** Images of vehicles showing various types of damage for classification.

**Training and Testing Data:**

* **Training Samples:** Approximately 15,000 images across various damage types.
* **Testing Samples:** Approximately 3,000 images for evaluation purposes.

- - -

## Program Flow

1. **Data Ingestion:** Load vehicle images and their corresponding metadata from the `Data` directory (e.g., image files and CSV containing damage information) and ingest it into a suitable format for processing. [`ingest_transform.py`]
2. **Data Transformation:** Preprocess the images and metadata, including resizing images, normalizing pixel values, and augmenting data to improve model robustness. The data is then split into training and validation sets. [`ingest_transform.py`]
3. **Model Training:** Train a deep learning model (e.g., using TensorFlow or PyTorch) to classify vehicle damage based on image data. This includes techniques like transfer learning with pre-trained models for improved accuracy. [`rcnn_train.py`, `train_rcnn_fast.py`, `train_rcnn_faster.py`, `train_mask_rcnn.py`]
4. **Manual Prediction:** Allow users to upload images of vehicles for classification, providing real-time predictions on the damage type and severity based on the trained model. This can be done via a command line interface (CLI) or through an API. [`predictor.py`]
5. **Web Application:** A `Streamlit` app that integrates the entire classification pipeline, allowing users to interactively upload images and receive predictions on vehicle damage, complete with additional information and resources. [`app.py`]

- - -

## Steps to Run

1. Install the required packages:

``` bash
pip install -r requirements.txt
```

2. Launch the application:

``` bash
streamlit run code/app.py
```

3. Using the Application:

   a) Input Data Paths (Tab 1):
   - Enter the path to your training annotations JSON file
     Example: `Data/Master/train/COCO_train_annos.json`
   - Enter the path to your training images directory
     Example: `Data/Master/train`

   b) Train Models (Tab 2):
   - Select the number of epochs for training (default: 10)
   - Choose and click one of the training buttons:
     - "Train Faster R-CNN"
     - "Train Fast R-CNN"
     - "Train R-CNN"
     - "Train Mask R-CNN"
   - Wait for the success message indicating training completion

   c) Make Prediction (Tab 3):
   - Select the model type from the dropdown menu
   - Enter the path to your test image
     Example: `Data/Master/test/car_damage_1.jpg`
   - Click "Predict" to view:
     - Image with detected damage regions
     - Damage label
     - Confidence score
     - Bounding box coordinates

   d) Model Evaluation (Tab 4):
   - Select the model type for evaluation
   - Enter the path to test annotations JSON file
     Example: `Data/Master/test/COCO_test_annos.json`
   - Enter the path to test images directory
     Example: `Data/Master/test`
   - Click "Evaluate Model" to view:
     - Average loss
     - Detection accuracy
     - Performance visualization
     - Total detections stats

4. Expected Output:
   - Training: Progress bars and loss metrics during training
   - Prediction: Annotated image with damage detection
   - Evaluation: Detailed performance metrics and charts

Note: Make sure all your data follows the COCO format for annotations and is organized in the correct directory structure as shown in the Directory Structure section above.