# OMDS 24-25 PROJECT
This repository contains the final project for the "Optimization Methods for Data Science" course (2024-2025), focusing on the implementation of machine learning models from scratch using optimization routines without automatic differentiation tools.

## Authors

[Valeria Avino](https://github.com/vaal4ds)

[Marta Lombardi](https://github.com/martalombardi)

## Project Overview
This project addresses two core machine learning tasks: **age regression** using a **Multi-Layer Perceptron (MLP)** and **gender/ethnicity classification** using **Support Vector Machines (SVMs)**.
The implementations emphasize foundational understanding of optimization and neural network/SVM mechanics.

### Part 1: Multi-Layer Perceptron for Age Regression

* Objective: Implement a neural network to minimize a regularized L2 loss function for age regression.
* Architecture: The network consists of at least two hidden layers, with L2 regularization applied to weights to prevent overfitting.
* Optimization: The optimization routine is developed using scipy.optimize, without reliance on automatic differentiation.
* Hyperparameter Tuning: K-fold cross-validation is utilized to select the best hyperparameters, including the number of layers (L, min: 2, max: 4), number of neurons per hidden layer, regularization term (
lambda), and non-linear activation function (Sigmoid or Hyperbolic Tangent).
* Performance Tracking: Mean Absolute Percentage Error (MAPE) is used to track model performance.

### Part 2: Support Vector Machines for Gender & Ethnicity Classification

* Objective: Implement nonlinear SVMs for binary (gender) and multiclass (ethnicity) classification tasks.
* Kernel Functions: Gaussian Kernel and Polynomial Kernel are explored, with their respective hyperparameters (gamma, p) tuned via k-fold cross-validation.
* Optimization: The SVM dual quadratic problem is solved using a convex optimization method like CVXOPT.
* Decomposition Method: The Most Violating Pair (MVP) decomposition method is implemented for solving subproblems with a fixed dimension of q=2.

### Dataset
The project utilizes the UTKFace dataset, which provides over 20,000 labeled images of human faces. Pre-extracted features from a ResNet convolutional backbone are provided as input vectors. Three specific datasets are used:

* AGE_REGRESSION.csv (float target between 0 and 100) 
* GENDER_CLASSIFICATION.csv (binary target) 
* ETHNICITY_CLASSIFICATION.csv (integer target from 0 to 4) 

Each dataset includes "feat i" columns for features and a "gt" column for the ground truth.

### Repository Contents
The repository is organized to clearly separate implementations for each part of the project.
```
├── Question 1/
│   ├── Functions_11_Avino_Lombardi.py       # Core Python functions for the MLP (forward, backward, loss, activations, etc.)
│   └── run_11_Avino_Lombardi.ipynb          # Notebook for training the MLP, hyperparameter tuning (k-fold), and metrics

├── Question 2/
│   ├── Functions_22_Avino_Lombardi.py        # Core SVM functions (dual solver, kernel functions, MVP decomposition, etc.)
│   └── run_22_Avino_Lombardi.ipynb           # Notebook(s) for SVM training, validation, confusion matrix, etc.

├── Question 3/
│   ├── Functions_23_Avino_Lombardi.py        # Core SVM functions (dual solver, kernel functions, MVP decomposition, etc.)
│   └── run_23_Avino_Lombardi.ipynb           # Notebook for most violating pair (MVP) decomposition method implementation

├── Question 4/
│   ├── Functions_24_Avino_Lombardi.py        # Core SVM functions (dual solver, kernel functions, MVP decomposition, etc.)
│   └── run_24_Avino_Lombardi.ipynb           # Notebook(s) for SVM multiclass classification



├── data/
│   ├── AGE_REGRESSION.csv
│   ├── GENDER_CLASSIFICATION.csv
│   └── ETHNICITY_CLASSIFICATION.csv

├── Avino_Lombardi.pdf                        # Final project report 
└── README.md                                 # This README file
```
