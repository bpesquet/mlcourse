# Predict heart disease

The goal of this lab is to train a model for the diagnosis of coronary artery disease.

## Dataset

The dataset is provided by the Cleveland Clinic Foundation for Heart Disease ([more information](https://archive.ics.uci.edu/ml/datasets/heart+Disease)). The dataset file to use is available [here](https://raw.githubusercontent.com/bpesquet/mlcourse/main/datasets/heart.csv). Each row describes a patient. Below is a description of each column.

|  Column  |                           Description                          |  Feature Type  | Data Type |
|:--------:|:--------------------------------------------------------------:|:--------------:|:---------:|
| Age | Age in years | Numerical | integer |
| Sex | (1 = male; 0 = female) | Categorical | integer |
| CP | Chest pain type (0, 1, 2, 3, 4) | Categorical | integer |
| Trestbpd | Resting blood pressure (in mm Hg on admission to the hospital) | Numerical | integer |
| Chol | Serum cholestoral in mg/dl | Numerical | integer |
| FBS | (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) | Categorical | integer |
| RestECG | Resting electrocardiographic results (0, 1, 2) | Categorical | integer |
| Thalach | Maximum heart rate achieved | Numerical | integer |
| Exang | Exercise induced angina (1 = yes; 0 = no) | Categorical | integer |
| Oldpeak | ST depression induced by exercise relative to rest | Numerical | float |
| Slope | The slope of the peak exercise ST segment | Numerical | integer |
| CA | Number of major vessels (0-3) colored by flourosopy | Numerical | integer |
| Thal | 3 = normal; 6 = fixed defect; 7 = reversable defect | Categorical | string |
| Target | Diagnosis of heart disease (1 = true; 0 = false) | Classification | integer |

## Platform

You may use either a local or remote Python environment for this lab.

The easiest way to obtain a working Python setup is by using a cloud-based [Jupyter notebook](https://jupyter.org/) execution platform like [Google Colaboratory](https://colab.research.google.com/), [Paperspace](https://www.paperspace.com/notebooks) or [Kaggle Notebooks](https://www.kaggle.com/code).

## Tools

To tackle this challenge, you should leverage three essential libraries of the Python ecosystem for Machine Learning: [NumPy](https://numpy.org), [pandas](https://pandas.pydata.org) and [scikit-learn](https://scikit-learn.org).

If any of these tools is new to you, follow the following tutorial(s) before going any further: [NumPy](../../tutorials/numpy/), [pandas](../../tutorials/pandas/), [scikit-learn](../../tutorials/scikit-learn/).

## Training process

You may train any binary classification model on this task, for example a basic [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) implementing the logistic regression algorithm.

To implement the training process, you should take inspiration from the [project workflow](../../lectures/project_workflow/) and [classification performance](../../lectures/classification_performance/) code examples.

## Extra work

Try another model, for example a decision tree, and compare their performances.
