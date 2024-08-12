<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
</head>
<body>

<h1>Breast Cancer Malignancy Prediction Model</h1>

<h2>Overview</h2>
<p>This project is a machine learning model built with TensorFlow and Keras to predict the malignancy of breast cancer using features extracted from fine needle aspirates (FNA) of breast masses.</p>

<h2>Dataset</h2>
<ul>
    <li><strong>Source:</strong> <a href="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29">UCI Machine Learning Repository</a></li>
    <li><strong>Instances:</strong> 569</li>
    <li><strong>Features:</strong> 30 (e.g., radius, texture, perimeter)</li>
    <li><strong>Classes:</strong> Malignant (M), Benign (B)</li>
    <li><strong>Class Distribution:</strong> 357 benign, 212 malignant</li>
</ul>

<h2>Project Structure</h2>
<pre>
<code>
├── data/
│   └── data.csv
├── notebooks/
│   └── breast_cancer_prediction.ipynb
├── README.md
└── requirements.txt
</code>
</pre>

<h2>Installation</h2>
<ol>
    <li>Clone the repo:
        <pre><code>git clone https://github.com/yourusername/breast-cancer-malignancy-prediction.git
cd breast-cancer-malignancy-prediction</code></pre>
    </li>
    <li>Install dependencies:
        <pre><code>pip install -r requirements.txt</code></pre>
    </li>
    <li>Run the Jupyter notebook:
        <pre><code>jupyter notebook</code></pre>
    </li>
</ol>

<h2>Model</h2>
<ul>
    <li><strong>Architecture:</strong>
        <ul>
            <li>2 Hidden Layers with ReLU activation and L2 regularization</li>
            <li>Dropout layers to prevent overfitting</li>
            <li>Output Layer with Sigmoid activation</li>
        </ul>
    </li>
    <li><strong>Optimizer:</strong> Adam</li>
    <li><strong>Loss Function:</strong> Binary Crossentropy</li>
</ul>

<h2>Results</h2>
<p>The model achieves high accuracy in predicting whether breast cancer is malignant or benign, with visualizations provided for training and validation loss.</p>


</body>
</html>
