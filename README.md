# Capstone-Bangkit-2024-FruitSnap

## About FruitSnap
FruitSnap is an application that uses machine learning technology to classify images of fruits. It utilizes TensorFlow as the main framework for machine learning model building. Utilizing transfer learning techniques, the application uses InceptionV3's pre-trained model on ImageNet datasets, and then retrains the model on fruit-specific datasets to improve classification accuracy.

## Dataset
The data used to train and test our model is taken from Kaggle, is in the form of a zip file, and contains three folders namely train, test, and val. Within each of the train, test, and val folders are eight folders or classes, namely freshapples, freshbanana, freshoranges, freshstrawberry, rottenapples, rottenbanana, rottenoranges, and rottenstrawberry. This division of the dataset helps to ensure unbiased model evaluation.

## Requirements
To run the code in this repository, the following dependencies are required (better import all library needed from notebook file):

1. TensorFlow: A machine learning framework for building and training models.
2. Matplotlib: A plotting library for visualizing data.
3. NumPy: A library for numerical computations.
4. Kaggle API: A library to download datasets from Kaggle.

## Model Architecture

1. Data Augmentation: Uses ImageDataGenerator to perform image augmentation such as rotation, width and height shift, cropping, and zooming to increase the variety of training data.
2. Base Model: Using the InceptionV3 model pre-trained on the ImageNet dataset as a base. This layer is not retrained initially to utilize the features learned from the ImageNet dataset.
3. Custom Layers: Menambahkan beberapa lapisan di atas model pre-trained InceptionV3 untuk menyesuaikan dengan dataset buah-buahan:
   * Flatten: To flatten the output of the base model.
   * Dense: Fully connected layer with 1024 neurons and ReLU activation.
   * Dropout: With a dropout rate of 0.5 to reduce overfitting.
   * Dense: Output layer with the number of neurons corresponding to the number of classes (fruit categories) and softmax activation for classification.
5. Training: The model is trained with two phases:
   * First phase: Train the custom layer with the frozen base model.
   * Second phase: Unfreeze the final few layers of the base model and train the entire model with a lower learning rate for fine-tuning.
