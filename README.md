# Emotion-Based Music Recommender System

## Overview:
This project implements an emotion-based music recommender system utilizing Convolutional Neural Networks (CNN) and ResNet50V2 architectures. It involves training models to detect emotions in images and recommend music based on the identified emotional context within visual content.

## Files Provided:
- `EmotionDetectionResNet50V2ModelTrain.ipynb`: Jupyter Notebook for training the ResNet50V2 model to detect emotions in images.
- `EmotionDetectionCNNModelTrain.ipynb`: Jupyter Notebook for training the CNN model for emotion detection in images.
- `EmotionBasedMusicRecommderSystem.ipynb`: Jupyter Notebook implementing the music recommender system based on detected emotions.

## Dataset:
The project utilizes the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) from Kaggle, containing facial expression images categorized into different emotions.

## Model Training:
### CNN Model Training:
The notebook `EmotionDetectionCNNModelTrain.ipynb` covers the training of the CNN model.
- Training involves preprocessing images and achieving a Test Loss of 0.92 and Test Accuracy of 66.41%.

### ResNet50V2 Model Training:
The notebook `EmotionDetectionResNet50V2ModelTrain.ipynb` details the training of the ResNet50V2 model.
- Training involves processing images and achieving a Test Loss of 1.09735 and Test Accuracy of 67.37%.

### Access Saved Trained Models
You can access the saved trained models (CNN_Model.h5 and ResNet50V2Model.h5) from [this Google Drive folder](https://drive.google.com/drive/folders/1OwpdNHo1B7ctxh0e7TmbOLisyBg7l78D?usp=sharing).

## Emotion-Based Music Recommender System:
The notebook `EmotionBasedMusicRecommderSystem.ipynb` implements the music recommender system.
- It utilizes the trained models to detect emotions in images and recommends music based on the identified emotional context.

### Functions in `EmotionBasedMusicRecommderSystem.ipynb`:
- `recommend_songs(pred_class)`: Recommends songs based on the predicted emotion class.
- `load_and_preprocess_image(filename, img_shape=224, default_img_shape=128)`: Loads and preprocesses images for emotion detection.
- `predict_and_plot(filename, class_names)`: Predicts emotions in an image and visualizes the prediction.

## Instructions:
1. Download the FER2013 dataset from the provided Kaggle link.
2. Organize the dataset into appropriate train and test directories.
3. Execute `EmotionDetectionCNNModelTrain.ipynb` and `EmotionDetectionResNet50V2ModelTrain.ipynb` for model training.
4. Run `EmotionBasedMusicRecommderSystem.ipynb` to implement the music recommender system.
   
Feel free to explore the provided notebooks to understand the training process and the functionality of the music recommender system.
# Emotion-Based-Music-Recommender-System
