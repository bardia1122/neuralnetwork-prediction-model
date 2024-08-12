Breast Cancer Malignancy Prediction Model
Overview
This project is a machine learning model built with TensorFlow and Keras to predict the malignancy of breast cancer using features extracted from fine needle aspirates (FNA) of breast masses.

Dataset
Source: UCI Machine Learning Repository
Instances: 569
Features: 30 (e.g., radius, texture, perimeter)
Classes: Malignant (M), Benign (B)
Class Distribution: 357 benign, 212 malignant

Model
Architecture:
2 Hidden Layers with ReLU activation and L2 regularization
Dropout layers to prevent overfitting
Output Layer with Sigmoid activation
Optimizer: Adam
Loss Function: Binary Crossentropy
Results
The model achieves high accuracy in predicting whether breast cancer is malignant or benign, with visualizations provided for training and validation loss.

License
This project is licensed under the MIT License.
