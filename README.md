# Machine Learning Concepts from Scratch

This repository contains implementations of fundamental machine learning concepts built from scratch using only Python and libraries like `NumPy`. The goal is to provide a clear, hands-on understanding of how common machine learning algorithms and techniques work under the hood.

## Features

- **Supervised Learning Algorithms**: Implementations of classification and regression algorithms such as Decision Trees, Linear Regression, etc.
- **Unsupervised Learning Algorithms**: Implementations of clustering algorithms like K-Means, PCA, etc.
- **Evaluation Techniques**: Methods like Cross-Validation, Information Gain, Entropy, etc.
- **Core Machine Learning Concepts**: Concepts like Entropy, Information Gain, Gradient Descent, PCA, etc., all implemented from scratch.

## Installation

To use the code, you’ll need `Python` and `NumPy`. If you haven't installed `NumPy`, you can do so by running the following command:

```bash
pip install numpy
```
Alternatively, you can clone this repository and run the code directly:
```
git clone https://github.com/SghairiYoussef/ML_Concepts_from_scratch.git
cd ML_Concepts_from_scratch
```
Structure
Here’s a brief overview of the repository structure:
```
ML_Concepts_from_scratch/
│
├── algorithms/             # Implementations of various ML algorithms
│   ├── decision_tree.py    # Decision tree implementation
│   ├── linear_regression.py # Linear regression from scratch
│   └── ...
│
├── evaluation/             # Functions for evaluation techniques
│   ├── cross_validation.py # Cross-validation implementation
│   └── ...
│
├── utils/                  # Helper functions for data preprocessing, etc.
│   └── data_utils.py       # Data utilities
│
├── README.md              # This file
└── requirements.txt        # List of dependencies
```
**Key Files:**
algorithms/: Contains implementations of machine learning algorithms.
evaluation/: Functions that evaluate model performance, such as cross-validation.
utils/: Helper functions for data manipulation, such as normalization and entropy calculation.
**Usage**
Example 1: Decision Tree Learning
The decision_tree.py file contains the implementation of a decision tree algorithm based on Information Gain.

from algorithms.decision_tree import learn_decision_tree
``` Python
# Example dataset
data = [
    {'feature1': 1, 'feature2': 2, 'class': 'A'},
    {'feature1': 3, 'feature2': 4, 'class': 'B'},
    {'feature1': 5, 'feature2': 6, 'class': 'A'},
    {'feature1': 7, 'feature2': 8, 'class': 'A'},
    {'feature1': 9, 'feature2': 10, 'class': 'B'}
]

attributes = ['feature1', 'feature2']
target_attr = 'class'

# Build the decision tree
decision_tree = learn_decision_tree(data, attributes, target_attr)

print(decision_tree)
```
For any questions, feel free to reach out to me.
