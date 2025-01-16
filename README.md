# Image-Classifier-Project---Udacity
This project is the final project of the nanodegree I am talking with Udacity, which is in collaboration with AWS. In this project I had to use a pretrained model and build a new classifier based on my purpose which is classifying different flowers types.

## Introduction 
This project implements an AI-based image classifier to predict flower species from input images. The classifier is developed using deep learning techniques and is implemented in Python. This repository contains the code and instructions for training a model, making predictions, and using the classifier in a command-line application.

## Project Structure
The repository consists of the following components:

1. train.py: A script for training the image classifier. It includes functionalities to:
- Load and preprocess datasets.
- Define and train a neural network model.
- Save the trained model to a checkpoint file.

2. predict.py: A script for predicting flower species using a trained model. It includes:
- Functionality to load the saved model checkpoint.
- Predict the species of a flower based on an input image.
- Display the top probabilities and corresponding flower names.

3. Colab Notebook: The notebook (Image_Classifier_Project.ipynb) documents the development process, including:
- Data exploration and preprocessing.
- Model architecture and hyperparameter tuning.
- Evaluation of the trained model.
