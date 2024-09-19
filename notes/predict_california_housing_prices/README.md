---
marp: true
---

<!-- Apply header and footer to first slide only -->
<!-- _header: "[![Bordeaux INP logo](../ensc_logo.jpg)](https://www.bordeaux-inp.fr)" -->
<!-- _footer: "[Baptiste Pesquet](https://www.bpesquet.fr)" -->

# Predict California housing prices

> This chapter is inspired by the book [Hands-On Machine Learning](https://github.com/ageron/handson-ml2)

---

## Learning objectives

- Discover how to train a Machine Learning model on tabular data.
- Get familiar with the general workflow of a supervised ML project.
- Learn how to leverage the [pandas](https://pandas.pydata.org/) and [scikit-learn](https://scikit-learn.org) libraries.

> You may test the trained model [here](https://housing-prices-api.herokuapp.com/).

---

## Standard workflow of a Machine Learning project

1. **Frame** the problem.
1. Collect, analyze and prepare **data**.
1. Select and train several **models** on data.
1. **Tune** the most promising model.
1. **Deploy** the model to production and monitor it.

[![ML workflow by RedHat](images/wiidii_ml_workflow.png)](https://www.redhat.com/files/summit/session-assets/2019/T957A0.pdf)

---

## Step 1: frame the problem

### Key questions

- What is the business objective?
- How good are the current solutions?
- What data is available?
- Is the problem a good fit for ML?
- What is the expected learning type (supervised or not, batch/online...)?
- How will the model's performance be evaluated?

---

### Properties of ML-friendly problems

- Difficulty to express the actions as rules.
- Data too complex for traditional analytical methods.
  - High number of features.
  - Highly correlated data (data with similar or closely related values).
- Performance > interpretability.
- Data quality is paramount.

---

### Example: predict housing prices in California

- Inputs: housing properties in an area (number of rooms, median income, etc).
- Output: median housing price in the area.

---

## Step 2: collect, analyze and prepare data

### A crucial step

- Real data is messy, incomplete and often scattered across many sources.

- Data labeling is a manual and tedious process.

- Predefined datasets offer a convenient way to bypass the data wrangling step. Alas, using one is not always an option.

---

### The California housing dataset

- Based on data from the 1990 California census.
- The raw CSV file is available [here](https://raw.githubusercontent.com/bpesquet/mlcourse/main/datasets/california_housing.csv).
- It is a slightly modified version of the [original dataset](https://www.dcc.fc.up.pt/%7Eltorgo/Regression/cal_housing.html).

[![Kaggle houses banner](images/kaggle_housesbanner.png)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)

---

### Step 2.1: discover data

Our first objective is to familiarize ourselves with the dataset.

Once data is loaded, the [pandas](https://pandas.pydata.org/) library  provides many useful functions for making sense of it.

---

### Step 2.2: analyze data

The objective here is to gain insights about the data, in order to prepare it optimally for training.

---

### Step 2.3: split the dataset

A simple solution for splitting datasets is to use the `train_test_split`function from scikit-learn.

Just before or after that, inputs (features given to the model) have to be separated from targets (values it must predict).

[![Using train_test_split](images/train-test-split.jpg)](https://mgalarnyk.github.io/)

---

### Step 2.4: data preprocessing

For this dataset, this task involves:

- Handling missing values.
- Scaling data.
- Transforming categorical values into numeric form.

#### Preprocessing pipelines

Data preprocessing is done through a series of sequential operations on data (handling missing values, standardization, one-hot encoding...).

scikit-learn support the definition of **pipelines** for streamlining these operations. This is useful to prevent mistakes and oversights when preprocessing new data.

---

## Step 3: select and train models

---

### An iterative and empirical step

At long last, our data is ready and we can start training models.

This step is often iterative and can be quite empirical. Depending on data and model complexity, it can also be resource-intensive.

![Busy training](images/busy_training.jpg)

---

### The optimization/generalization dilemna

![Underfitting and overfitting](images/underfitting_overfitting.png)

---

### Underfitting and overfitting

- **Underfitting** (sometimes called *bias*): insufficient performance on training set.
- **Overfitting** (sometimes called *variance*): performance gap between training and validation sets.

Ultimately, we look for a tradeoff between underfitting and overfitting.

The goal of the training step is to find a model powerful enough to **overfit the training set**.

---

### Step 3.1: choose an evaluation metric

Model performance is assessed through an **evaluation metric**. Like the loss function, it depends on the problem type.

A classic choice for regression tasks is the **Root Mean Square Error (RMSE)**. It gives an idea of how much error the trained model typically makes in its predictions. Of course, the smaller the better in that case.

$$\mathrm{RMSE}(\pmb{Y}, \pmb{Y'}) = \sqrt{\frac{1}{m}\sum_{i=1}^m (\pmb{y'}^{(i)} - \pmb{y}^{(i)})^2}$$

**Mean Absolute Error** (less sensitive to outliers) and **MSE** may also be used.

---

### Step 3.2: start with a baseline model

For each learning type (supervised, unsupervised...), several models of various complexity exist.

It is often useful to begin the training step by using a basic model. Its results will serve as a **baseline** when training more complicated models. In some cases, its performance might even be surprisingly good.

In this example, a [Linear Regression](../linear_regression/README.md) model would be a good first choice.

---

### Step 3.3: try other models

After obtaining baseline results, other more sophisticated models may be tried, for example a [Decision Tree](../decision_trees_random_forests/README.md) in our case.

---

### Step 3.4: Use a validation set to evaluate model performance

Some results looks too good to be true. They are often cases of severe overfitting to the training set, which means the model won't perform well with unseen data.

One way to assert overfitting is to split training data between a smaller training set and a **validation set**, used only to evaluate model performance after each training iteration.

---

#### Cross-validation

A more sophisticated strategy is to apply **K-fold cross validation**. Training data is randomly split into $K$ subsets called *folds*. The model is trained and evaluated $K$ times, using a different fold for validation.

![K-fold Cross Validation](images/k-fold-cross-validation.png)

---

## Step 4: tune the most promising model

---

### Step 4.1: search for the best hyperparameters

Once a model looks promising, it must be **tuned** in order to offer the best compromise between optimization and generalization.

The goal is to find the set of model properties that gives the best performance. Model properties are often called **hyperparameters** (example: maximum depth for a decision tree).

This step can be either:

- manual, tweaking model hyperparameters by hand.
- automated, using a tool to explore the model hyperparameter spaces.

---

### Step 4.2: check final model performance on test dataset

Now is the time to evaluate the final model on the test set that we put apart before.

An important point is that preprocessing operations should be applied to test data using preprocessing values (mean, categories...) previously computed on training data. This prevents **information leakage** from test data ([explanation](https://machinelearningmastery.com/data-leakage-machine-learning/))

---

## Step 5: deploy to production and maintain the system

### Step 5.1: save the final model and data pipeline

A trained model can be saved to several formats. A standard common is to use Python's built-in persistence model, [pickle](https://docs.python.org/3/library/pickle.html), through the [joblib](https://pypi.org/project/joblib/) library for efficiency reasons.

---

### Step 5.2: deploy the model

This step is highly context-dependent. A deployed model is often a part of a more important system. Some common solutions:

- deploying the model as a web service accessible through an API.
- embedding the model into the user device.

The [Flask](https://flask.palletsprojects.com) web framework is often used to create a web API from a trained Python model.

[![Model deployement on the web](images/model_deployment_web.png)](https://github.com/ageron/handson-ml2)

> You may test the trained model [here](https://housing-prices-api.herokuapp.com/).

---

### Step 5.3: monitor and maintain the system

In order to guarantee an optimal quality of service, the deployed system must be carefully monitored. This may involve:

- Checking the system’s live availability and performance at regular intervals.
- Sampling the system’s predictions and evaluating them.
- Checking input data quality.
- Retraining the model on fresh data.
