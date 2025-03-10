# Scikit-learn Tutorial

> [!NOTE]
> Adapted from [Getting Started with scikit-learn](https://scikit-learn.org/stable/getting_started.html).

## What scikit-learn is

[Scikit-learn](https://scikit-learn.org/stable/index.html) is an open source machine learning library that supports supervised and unsupervised learning. It also provides various tools for model fitting, data preprocessing, model selection, model evaluation, and many other utilities.

## How to import scikit-learn

After [installing scikit-learn](https://scikit-learn.org/stable/install.html), the standard practice is to import the necessary parts *à la carte*.

For this tutorial, we need the following sckit-learn elements.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
```

## Fitting and predicting: estimator basics

Scikit-learn provides dozens of built-in machine learning algorithms and models, called [estimators](https://scikit-learn.org/stable/glossary.html#term-estimators). Each estimator can be fitted to some data using its [fit()](https://scikit-learn.org/stable/glossary.html#term-fit) method.

Here is a simple example where we fit a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier to some very basic data.

```python
# Create a very basic dataset: 2 samples, 3 features
x_train = [[1, 2, 3], [11, 12, 13]]
# Classes for each sample of the dataset
y_train = [0, 1]

# Create a linear regression model
model = LogisticRegression()

# Fit the model to the training data
model.fit(x_train, y_train)
```

The `fit()` method generally accepts two inputs:

- The samples matrix (or design matrix), generally denoted `x_train`. The size of `x_train` is typically `(n_samples, n_features)`, which means that samples are represented as rows and features are represented as columns.
- The target values, generally denoted `y_train`. They are real numbers for regression tasks, or integers for classification (or any other discrete set of values). For unsupervised learning tasks, `y_train` does not need to be specified. `y_train` is usually a 1D array where the `i`th entry corresponds to the target of the `i`th sample (row) of `x_train`.

Both `x_train` and `y_train` are usually expected to be NumPy arrays or equivalent array-like data types, though some estimators work with other formats such as sparse matrices.

Once the estimator is fitted, it can be used for predicting target values of data with the [predict()](https://scikit-learn.org/stable/glossary.html#term-predict) method.

> [!NOTE]
> The [np.testing.assert_allclose()](https://numpy.org/doc/2.2/reference/generated/numpy.testing.assert_allclose.html#numpy.testing.assert_allclose) function asserts equality between two NumPy arrays.

```python
# Predict classes for the training data
y_pred = model.predict(x_train)
print(y_train)
# Assess equality with expected values
np.testing.assert_allclose(y_pred, y_train)

# Predict classes for new data
y_pred = model.predict([[14, 15, 16], [4, 5, 6]])
print(y_pred)
# Assess equality with expected values
np.testing.assert_allclose(y_pred, [1, 0])
```

> [!TIP]
> Learn more about choosing the right model for your use case [here](https://scikit-learn.org/stable/machine_learning_map.html#ml-map).

## Transformers and preprocessors

Machine learning workflows are often composed of different parts. A typical pipeline consists of a preprocessing step that transforms or imputes the data, and a final predictor that predicts target values.

In scikit-learn, preprocessors and transformers follow the same API as the estimator objects (they actually all inherit from the same `BaseEstimator` class). The transformer objects don’t have a `predict()` method but rather a [transform()](https://scikit-learn.org/stable/glossary.html#term-transform) method that outputs a newly transformed samples matrix.

In the following example, the `fit()` method computes the scaling metrics (mean and standard deviation for each feature) and the `transform()` method performs the standardization operation using these metrics.

```python
x = [[0, 15], [1, -10]]

# Scale data according to computed scaling values
x_scaled = StandardScaler().fit(x).transform(x)
print(x_scaled)

# Assess standrdization results (mean=0, std=1 for each feature)
np.testing.assert_allclose(x_scaled.mean(axis=1), [0, 0])
np.testing.assert_allclose(x_scaled.std(axis=1), [1, 1])
```

## Pipelines and model evaluation

Transformers and estimators (predictors) can be combined together into a single unifying object: a [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline). The pipeline offers the same API as a regular estimator: it can be fitted and used for prediction with `fit()` and `predict()`. Using a pipeline will also prevent you from *data leakage*, i.e. disclosing some testing data in your training data.

In the following example, we [load](https://scikit-learn.org/stable/datasets.html#datasets) the [Iris](https://archive.ics.uci.edu/dataset/53/iris) dataset, split it into train and test sets, and fit a pipeline composed of a `StandardScaler` and a `LogisticRegression` classifier on it.

```python
# Load the iris dataset and split it into train and test sets
x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# Create a pipeline object
pipeline = make_pipeline(StandardScaler(), LogisticRegression())

# Fit the whole pipeline to the training set
pipeline.fit(x_train, y_train)
```

Fitting a model to some data does not entail that it will predict well on unseen data. This needs to be directly evaluated. We have just seen the [train_test_split()](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) helper function that splits a dataset into train and test sets, but scikit-learn provides many other tools for model evaluation.

The following code evaluates our model using the [accuracy_score()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) and [classification_report()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) functions.

```python
# We can now use it like any other estimator
y_pred = pipeline.predict(x_test)

# Assess model accuracy on test data
test_acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_acc:.5f}")
assert test_acc > 0.97

# Compute classification metrics (precision, recall, f1-score) on th test set
print(classification_report(y_test, y_pred))
```
