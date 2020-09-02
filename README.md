CNN for Camera Auto-calibration on Soccer Robots
====================================

This project is a Convolutional Neural Network(CNN) for camera auto-calibration on autonomous soccer robots. The codes are divided in following parts:

1. Dataset
2. Data Preparing
3. Model Building
4. Model Training
5. Model Testing
6. Ball Detection Accuracy Calculating

## 1. Dataset

All dataset are described in this part. The folder "dataset" includes all training and testing image samples in type of png and their corresponding labels in type of txt. The folder "save_imgs" includes input images(bad images), original images and all output images from different models.

## 2. Data Preparing

The codes in this part are for data preparing. "Loading_imgs.py" includes a function for load images. "Data_pre_processing.py" includes all functions for data pre-processing, in order data to training later.

## 3. Model Building

The codes in this part are for model building. "Models.py" includes 6 classes for building 6 different models.

## 4. Model Training

The codes in this part are for model training. "Training_Process_of_image_enhancement_CNN.ipynb" is used for training 6 different models.

If you don't want to train models by yourself. You can use the complete trained models in the folder "checkpoints".

## 5. Model Testing

The codes in this part are for model testing."Testing_Process_of_image_enhancement_CNN.ipynb" is used for testing 6 different models. The loss values and cost values and predicted values are displayed in it.

## 6. Ball Detection Accuracy Calculating

The codes in this part are for calculating ball detection accuracy. "ball_location_calculating.py" is the file for calculating ball location, which can run in the robpcupspl algorithms. It can be used under the path: robocupspl/visionng/python_common by using command: python ball_location_calculating.py image_folder_name. All ball location predicted values are saved in the folder ball_predicition_values. "Caculate_bt_acc.ipynb" is the file for ball detection accuracy calculating.



