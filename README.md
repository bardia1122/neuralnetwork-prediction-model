
Breast Cancer Malignancy Prediction Model
Project Overview
This project focuses on building a machine learning model using a neural network to predict the malignancy of breast cancer based on the characteristics of cell nuclei extracted from digitized images of fine needle aspirates (FNA) of breast masses. The model leverages TensorFlow and Keras for building and training the neural network.

Dataset Description
The dataset used in this project comes from the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, available through the UCI Machine Learning Repository. The dataset includes various features computed from the digitized images of FNAs of breast masses. These features describe the characteristics of the cell nuclei present in the images.

Number of Instances: 569
Number of Features: 30
Number of Classes: 2 (Malignant, Benign)
Class Distribution: 357 benign, 212 malignant
Attribute Information:
ID Number (Not used in the model)
Diagnosis (M = malignant, B = benign)
Ten real-valued features computed for each cell nucleus:
Radius (mean of distances from center to points on the perimeter)
Texture (standard deviation of gray-scale values)
Perimeter
Area
Smoothness (local variation in radius lengths)
Compactness (perimeter^2 / area - 1.0)
Concavity (severity of concave portions of the contour)
Concave points (number of concave portions of the contour)
Symmetry
Fractal dimension ("coastline approximation" - 1)
The dataset is well-balanced and contains no missing values. Each feature has been recorded with four significant digits.

Project Structure
bash
Copy code
├── data/
│   └── data.csv             # Dataset file
├── model/
│   └── breast_cancer_model.h5  # Trained model (to be saved after training)
├── notebooks/
│   └── breast_cancer_prediction.ipynb  # Jupyter notebook containing the code
├── README.md                # Project readme file
└── requirements.txt         # Python packages required to run the code
Installation and Setup
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/breast-cancer-malignancy-prediction.git
cd breast-cancer-malignancy-prediction
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Run the Jupyter notebook:

Launch Jupyter Notebook:
bash
Copy code
jupyter notebook
Open notebooks/breast_cancer_prediction.ipynb to explore the code and run the model training and evaluation.
Model Description
The model is built using a deep neural network with the following architecture:

Input Layer: 30 input features.
Hidden Layer 1: 15 neurons with ReLU activation and L2 regularization.
Dropout Layer: 50% dropout rate to prevent overfitting.
Hidden Layer 2: 10 neurons with ReLU activation and L2 regularization.
Dropout Layer: 50% dropout rate.
Output Layer: 1 neuron with sigmoid activation to output a probability score between 0 and 1.
Model Compilation
The model is compiled using the following configurations:

Loss Function: Binary Crossentropy
Optimizer: Adam with a learning rate of 0.001
Metrics: Accuracy
Training and Validation
The model is trained using the training data with early stopping applied to monitor the validation loss. The training will stop if the validation loss does not improve for 10 consecutive epochs, and the best weights are restored.

Evaluation
The model is evaluated on the test data, and the following metrics are computed:

Test Loss
Test Accuracy
Additionally, predictions are made on a manually provided test case to demonstrate the model's performance.

Results
After training, the model achieves a high level of accuracy on the test data, demonstrating its ability to predict the malignancy of breast cancer with a high degree of reliability.

Sample Prediction
A sample prediction is made using a manually provided data point representing the features of a breast mass. The model predicts whether the mass is malignant or benign.

Visualizations
The training and validation losses are plotted to visualize the model's performance over the epochs. This plot helps in understanding the model's learning process and diagnosing issues such as overfitting.

Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to create a pull request or raise an issue.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
This project uses the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, which is publicly available from the UCI Machine Learning Repository.
Special thanks to the creators of TensorFlow and Keras for providing the tools to build and train deep learning models easily.
References
UCI Machine Learning Repository: Breast Cancer Wisconsin (Diagnostic) Data Set
