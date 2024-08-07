# Basic Machine Learning

This repository contains Jupyter notebooks covering fundamental machine learning algorithms and techniques. Each notebook provides an introduction to the algorithm, its mathematical foundation, implementation in Python, and examples using real datasets.

## Contents

1. [K-Nearest Neighbor](K-nearest%20neighbor.ipynb)
2. [Linear Discriminant Analysis](LINEAR%20DISCRIMINANT%20ANALYSIS.ipynb)
3. [Linear Regression](Linear%20Regression.ipynb)
4. [Multivariate Regression](Multivariate%20regression.ipynb)
5. [Naive Bayes](Naive%20bayes.ipynb)
6. [Principal Component Analysis](Principal%20Component%20Analysis.ipynb)
7. [K-Means Clustering](k-means.ipynb)
8. [K-Medians Clustering](k-medians.ipynb)
9. [K-Means Clustering (Small Dataset)](k_means%20small.ipynb)
10. [Logistic Regression](logistic%20regression.ipynb)
11. [Support Vector Machine](support%20vector%20machine.ipynb)

## Getting Started

To get started with these notebooks, you'll need to have Python and Jupyter installed. We recommend using [Anaconda](https://www.anaconda.com/products/distribution), which comes with both, along with many other useful data science packages.

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

### Installation

1. Clone this repository to your local machine:

    ```sh
    git clone https://github.com/your-username/basic-machine-learning.git
    ```

2. Navigate to the cloned repository:

    ```sh
    cd basic-machine-learning
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run a Jupyter notebook, navigate to the repository directory and start Jupyter Notebook:

```sh
jupyter notebook
```
This will open a new tab in your web browser. From there, you can open any of the provided notebooks and run the code cells.

## Notebooks Overview

### [K-Nearest Neighbor](K-nearest%20neighbor.ipynb)
The K-Nearest Neighbor (KNN) algorithm is a simple, yet powerful, classification method. It works by finding the K closest data points in the training set to a new data point and assigning the most common class among those neighbors to the new data point. This notebook covers the implementation of KNN and its application to classification problems.

### [Linear Discriminant Analysis](LINEAR%20DISCRIMINANT%20ANALYSIS.ipynb)
Linear Discriminant Analysis (LDA) is used for dimensionality reduction while preserving as much of the class discriminatory information as possible. It projects the data onto a lower-dimensional space with good class-separability to avoid overfitting. This notebook explores LDA for both classification and feature extraction.

### [Linear Regression](Linear%20Regression.ipynb)
Linear Regression is a fundamental algorithm for predicting a continuous dependent variable based on one or more independent variables. This notebook provides a comprehensive look at Linear Regression, covering simple linear regression, which deals with one predictor variable, and multiple linear regression, which involves several predictor variables.

### [Multivariate Regression](Multivariate%20regression.ipynb)
Multivariate Regression is an extension of linear regression to multiple dependent variables. It models the relationship between multiple independent variables and multiple dependent variables. This notebook demonstrates how to implement and interpret multivariate regression models.

### [Naive Bayes](Naive%20bayes.ipynb)
Naive Bayes is a probabilistic classifier based on Bayes' Theorem with strong independence assumptions between the features. This notebook explains the Naive Bayes classifier and its variants: Gaussian, Multinomial, and Bernoulli, illustrating how they can be applied to classification problems.

### [Principal Component Analysis](Principal%20Component%20Analysis.ipynb)
Principal Component Analysis (PCA) is a technique used for reducing the dimensionality of datasets, increasing interpretability while minimizing information loss. It does so by transforming the data to a new set of variables, the principal components, which are orthogonal and capture the maximum variance in the data. This notebook demonstrates PCA for feature extraction and data visualization.

### [K-Means Clustering](k-means.ipynb)
K-Means Clustering is an unsupervised learning algorithm used to partition a dataset into K distinct, non-overlapping subsets (clusters). Each data point belongs to the cluster with the nearest mean. This notebook introduces K-Means clustering, with examples showing how to cluster data into meaningful groups.

### [K-Medians Clustering](k-medians.ipynb)
K-Medians Clustering is similar to K-Means, but it uses the median instead of the mean, making it more robust to outliers. This notebook explains the K-Medians clustering algorithm and its application in scenarios where data contains significant outliers.

### [K-Means Clustering (Small Dataset)](k_means%20small.ipynb)
This notebook provides a simpler, faster version of K-Means clustering for smaller datasets. It focuses on the implementation details and practical considerations when working with small datasets.

### [Logistic Regression](logistic%20regression.ipynb)
Logistic Regression is used for binary classification problems. It models the probability that a given input point belongs to a certain class. This notebook offers a detailed look at Logistic Regression, including the logistic function, the cost function, and how to fit and evaluate the model.

### [Support Vector Machine](support%20vector%20machine.ipynb)
Support Vector Machine (SVM) is a powerful classification algorithm that finds the hyperplane that best separates the classes in the feature space. This notebook covers the theory behind SVM, including the concept of the margin and support vectors, and demonstrates how to implement and apply SVM to classification tasks.

