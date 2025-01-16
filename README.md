# Image-Classifier-Project---Udacity
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

## Key Features
1. Deep Learning Framework: Utilizes PyTorch for defining, training, and evaluating the neural network.
2. Command-line Application: Offers flexibility to train models and make predictions directly from the command line.
3. Customizability: Allows users to specify hyperparameters, model architecture, and other configurations via command-line arguments.

## Installation
```
git clone https://github.com/razandakkak/Image-Classifier-Project---Udacity.git
```
```
pip install -r requirements.txt
```
After setting up the environment, make sure you have the dataset to train your model on; the data directory should have train, validation and test data folders within it.

## Usage
Training the Model
Use the train.py script to train a new model: 
```
python train.py --data_dir <path_to_data> --save_dir <checkpoint_dir> --epochs <num_epochs> --learning_rate <lr> --arch <architecture>
```
For example:
```
python train.py /content/flowers --save_dir checkpoints --epochs 10 --learning_rate 0.001 --arch "vgg16" --gpu
```

Making Predictions

Use the predict.py script to predict the species of a flower:
```
python predict.py flowers/test/1/image_06743.jpg --checkpoint checkpoints/model.pth --top_k 5 --category_names cat_to_name.json --gpu
```
