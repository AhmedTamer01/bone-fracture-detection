# Bone Fracture Detection using Deep Learning 🦴⚕️

A computer vision system built to automatically classify medical X-ray images, determining whether a bone is fractured or normal with high accuracy.

## 🚀 Project Overview

Medical image analysis is one of the most impactful applications of Deep Learning. Identifying minor fractures in X-rays can be challenging and time-consuming for radiologists. This project aims to assist medical professionals by providing a fast, automated second opinion. 

By leveraging Transfer Learning on a state-of-the-art convolutional neural network, the model effectively extracts complex bone structures and anomalies from grayscale X-ray images.

## ⚙️ Methodology & Architecture

1. **Data Augmentation**: To prevent overfitting and generalize better to real-world clinical variations, the training data was augmented using random rotations, flips, color jitter (brightness/contrast), and random crops using `torchvision.transforms`.
2. **Model Architecture**: The core backbone is **EfficientNetV2-S**, pre-trained on ImageNet. The classifier head was stripped and replaced with a custom dense block:
   - Dropout (0.3)
   - Linear layer (256 units) with ReLU activation
   - Output Linear layer (1 unit) for binary classification (BCEWithLogitsLoss).
3. **Training Strategy**: 
   - Feature extraction phase: The backbone was initially frozen to train only the classifier head.
   - Fine-tuning phase: The backbone was un-frozen to optimize all weights together.
   - **AdamW Optimizer** combined with a `ReduceLROnPlateau` scheduler was used to adaptively lower the learning rate when validation loss plateaued.

## 📊 Dataset

The model utilizes the **Fracture Multi-Region X-Ray Data** dataset, dynamically downloaded and processed via the Kaggle API. It contains thousands of X-ray images organized into standard `train`, `val`, and `test` splits.

## 🛠️ Tech Stack & Libraries Used

- **Deep Learning Framework:** PyTorch, Torchvision
- **Model Backbone:** EfficientNetV2-S (`models.efficientnet_v2_s`)
- **Metrics & Evaluation:** Scikit-learn (Classification Report, ROC-AUC, Confusion Matrix)
- **Data Visualization:** Matplotlib, Seaborn
- **Image Processing:** PIL (Python Imaging Library)

## 📈 Key Results

- High precision and recall in detecting subtle fractures.
- Robust performance across different X-ray contrast levels and orientations due to aggressive data augmentation.
- Efficient inference time suitable for real-time clinical deployment.

---
*This repository is part of my AI Engineer / Data Scientist Portfolio.*
