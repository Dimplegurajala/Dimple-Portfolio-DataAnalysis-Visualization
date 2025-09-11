## Brain Tumor Detection using Convolutional Neural Network (CNN)

## Project Description  
This project focuses on **automating the diagnosis of brain tumors** using deep learning techniques. The goal is to develop a **Convolutional Neural Network (CNN)** model that can accurately detect and classify brain tumors from **MRI images**.

By leveraging AI, this project aims to:  
- Assist radiologists in early tumor detection.  
- Improve diagnostic efficiency in real-world healthcare systems.  
- Demonstrate the potential of deep learning in medical image analysis.  

---

## Methodology  

### 1️⃣ Data Sourcing & Preparation  
- **Dataset:** [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  
- **Preprocessing:**  
  - Image resizing & standardization  
  - Data augmentation (rotation, zoom, flips) to improve generalization  
- **Splits:** Training, validation, and testing sets  

### 2️⃣ Model Architecture  
- Built with **TensorFlow** and **Keras**  
- CNN pipeline includes:  
  - **Convolutional Layers** – for feature extraction  
  - **Pooling Layers** – for dimensionality reduction  
  - **Dense Layers** – for final classification  
- **Output Classes:**  
  - 🧠 **Glioma**  
  - 🧠 **Meningioma**  
  - 🧠 **Pituitary Tumor**  
  - ❌ **No Tumor**  

### 3️⃣ Training & Evaluation  
- Optimized using **Adam optimizer**  
- Evaluated with:  
  - Accuracy  
  - Precision, Recall, F1-score  
  - Confusion matrix  

---

## Installation & Usage  
Install the required Python libraries:: pip install pandas numpy tensorflow keras scikit-learn matplotlib seaborn kagglehub

## Download the Dataset

Set up your Kaggle API
 credentials.

## Run the code cells in Braintumor.ipynb to automatically download and prepare the dataset.
jupyter notebook Braintumor.ipynb

## Model performance is documented in Machine Learning - Final Project.docx

# Includes:
-Accuracy & loss curves
-Confusion matrix visualizations
-Classification report


###  Clone the Repository  
```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
