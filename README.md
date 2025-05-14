# 🧠 Intel Image Classification using Machine Learning & Neural Networks

This project applies traditional machine learning algorithms and neural networks to classify natural scene images from the Intel Image Classification dataset. By extracting features and comparing multiple models, we aim to find the most effective classifier for real-world image recognition tasks.

---

## 📌 Project Overview

- **Goal**: Predict the category of natural scene images (e.g., forest, mountain, sea) using supervised machine learning and neural networks.
- **Approach**: 
  - Dataset loading and preprocessing using Kaggle API
  - Image resizing and feature extraction
  - Model training and evaluation using traditional ML models and a neural network
  - Comparative analysis of model performance using accuracy, classification report, and confusion matrix

---

## 🧰 Tools & Technologies

- **Languages**: Python
- **Libraries**: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `tensorflow`, `keras`, `opencv-python`, `os`, `shutil`
- **Techniques**: Image preprocessing, feature flattening, logistic regression, support vector machines, random forest, XGBoost, ensemble voting, feedforward neural networks, model evaluation

---

## 📂 Dataset

This project uses the **Intel Image Classification** dataset sourced from Kaggle. It contains images categorized into six natural scene classes: buildings, forest, glacier, mountain, sea, and street.

- **Source**: [Intel Image Classification – Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)  
- **License**: Unknown (used under Kaggle’s terms)  
- **Notes**: Dataset is divided into `train`, `test`, and `val` directories for supervised learning.

---

## 📁 Project Structure

```bash
.
├── 01_Introduction.ipynb                    # Overview of project objectives and methodology
├── 02_Dataset_setup_and_exploration.ipynb   # Loading dataset, image resizing, basic stats, and structure
├── 03_Exploratory_Data_Analytics.ipynb      # Visual analysis of class distribution, sample images, and pixel stats
├── 04_Data_preprocessing_cleaning.ipynb     # Data flattening, normalization, label encoding, and train-test split
├── 05_neural_networks.ipynb                 # Feedforward neural network model using Keras
├── 06.1_logistic_regression.ipynb           # Logistic Regression implementation and evaluation
├── 06.2_SVM.ipynb                            # Support Vector Machine model training and evaluation
├── 06.3_Random_forest.ipynb                 # Random Forest model training and analysis
├── 06.4_XGBoost.ipynb                        # XGBoost model setup and performance metrics
└── 07_Ensemble.ipynb                         # Voting ensemble combining top-performing ML models
