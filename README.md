# Pet-Face-Classification-
This repository contains code for training a deep learning model for image classification using transfer learning with the ResNet50 architecture. The model is trained to classify images of various cat and dog breeds.

Dataset
The dataset consists of images of different cat and dog breeds obtained from various sources. Each image is labeled with the corresponding breed.

Requirements
Python 3.x
TensorFlow
NumPy
pandas
matplotlib
You can install the required Python packages using the following command:
- pip install -r requirements.txt
Usage
Clone the repository:
- git clone https://github.com/your_username/image-classification.git
Navigate to the project directory:
arduino
- cd image-classification
Download the dataset and place it in the data directory.

Run the train.py script to train the model:
- python train.py
Evaluate the trained model using the evaluate.py script:
- python evaluate.py
Visualize the training process using the visualize.py script:
- python visualize.py
Model Architecture
The model architecture consists of a pre-trained ResNet50 base model with additional layers for classification. Data augmentation techniques are applied to improve generalization.

Results
After training the model for 5 epochs, the model achieved an accuracy of approximately 69.23% on the test set.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
TensorFlow
NumPy
pandas
matplotlib
