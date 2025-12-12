## Food Image Classification
# Deep Learning · Transfer Learning · PyTorch · Computer Vision

A complete end-to-end deep learning project to classify 10 types of food images using transfer learning with ResNet-50, EfficientNet-B0, and ResNet-101.

# This repository includes:

-Full training pipeline

-Experiments with multiple pretrained CNNs

-Macro F1 score evaluation

-Grad-CAM visualizations

-Exported model for deployment (TorchScript + inference script)

-Ready-to-use inference module

# Dataset
Source:

https://drive.google.com/drive/folders/1ZwCHJhAZU_FposqaA8iQHi_TZ2SUz2R9?usp=drive_link

# The dataset contains ~10,000 images across 10 classes:

Cannoli

Ceviche 

Crab Cakes

Frozen Yogurt 

Gnocchi 

Grilled Cheese Sandwich 

Onion Rings 

Pork Chop 

Ravioli 

Spaghetti Bolognese

# Each class includes:
750 training images

250 test images

# Project Structure
Food_image_classification/

│

├── 1_Environment_&_Utilities.ipynb

├── 2_Data_Preparation_&_Organization.ipynb

├── 3_EDA_&_Visualizations.ipynb

├── 4_Training_pipeline_(ResNet_50_baseline).ipynb

├── 5_Experiments_&_Alternatives_(EffNetB0,ResNet101).ipynb


├── 6_demo_and_export.ipynb

│

├── export/

│   ├── model_scripted.pt

│   ├── best_model.pth

│   ├── inference.py

│   ├── sample_predictions.json

│

├── outputs/

├── outputs_experiments/

└── README.md

# Models Trained & Results
| Model | Validation Macro F1 | Test Macro F1 | Notes |
|-------|---------------------|---------------|-------|
| ResNet-50 (baseline) | ~0.90 | 0.9368 | Strong baseline |
| EfficientNetB0 | ~0.89 | 0.8997 | Underperformed slightly | 
| ResNet-101 | ~0.92 | 0.9385 | Best-performing model |

**Thus, Final Model Used for Deployment: ResNet-101**

# Evaluation Metrics
Macro F1 Score (Test Set)

ResNet-101 Final: **0.93855**

# Confusion Matrix:
**ResNet101:**

<img width="1500" height="1200" alt="best_resnet101_full_confmat" src="https://github.com/user-attachments/assets/1d2fb9d6-eef8-4633-a850-80b7e6ae1955" />

**EfficientNetB0:**

<img width="1500" height="1200" alt="best_effnetb0_freeze1_confmat" src="https://github.com/user-attachments/assets/bb8b11c3-5900-4d65-aa4b-b1c8bc0f40fd" />

# Grad-CAM Visualizations:
Shows which regions the CNN focuses on when predicting a class.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f69a7216-bef0-49e8-933f-f4c43232709c" />


# Exported Model (Deployment Ready)
The following files are inside your 'export' directory:

model_scripted.pt        #TorchScript format (portable)

best_model.pth           #PyTorch state_dict

inference.py             #ready-to-use inference script

sample_predictions.json  #example outputs

# Run Inference (Python)
Option 1- Use the inference.py
python3 export/inference.py

Option 2- Import into your own script
from inference import Food10Classifier

clf=Food10Classifier()
prediction=clf.predict("path_to_image.jpg")
print(prediction)

# Requirements
Install dependencies:
pip install torch torchvision timm pillow numpy scikit-learn matplotlib seaborn

# How to Reproduce the Full Training
1️) Run notebooks in order:
1 → 2 → 3 → 4 → 5 → 6

2️) Evaluate the models
Notebook 5 (Cell 10) evaluates:
EfficientNet-B0 & ResNet-101

3️) Export the best model
Run Notebook 6 to generate:
-TorchScript model
-Inference script
-Predictions sample

# Deployment Options
✔ 1. Streamlit App Example
import streamlit as st
from inference import Food10Classifier
from PIL import Image

clf = Food10Classifier()

st.title("🍽️ Food Image Classifier")
uploaded = st.file_uploader("Upload a food image", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    pred = clf.predict(img)
    st.subheader(f"Prediction: {pred}")


**Run:**

**streamlit run app.py**

✔ 2. Gradio App Example
import gradio as gr
from inference import Food10Classifier

clf = Food10Classifier()

def classify(img):
    return clf.predict(img)

gr.Interface(fn=classify, inputs="image", outputs="label").launch()

## Project Summary (for submission / PDF)

-Implemented EDA, data organization, and augmentation

-Trained strong baselines (ResNet-50) and advanced CNNs (EffNet, ResNet-101)

-Achieved 93.85% Macro F1 on official test set

-Used Grad-CAM for model interpretability

-Exported a deployable TorchScript model for production

-Created inference engine + optional Streamlit/Gradio apps




