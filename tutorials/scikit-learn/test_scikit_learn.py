"""
Scikit-learn tutorial code
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def test_fitting_predicting():
    """
    Test fitting and prediction functions
    """

    # Create a very basic dataset: 2 samples, 3 features
    x_train = [[1, 2, 3], [11, 12, 13]]
    # Classes for each sample of the dataset
    y_train = [0, 1]

    # Create a linear regression model
    model = LogisticRegression()

    # Fit the model to the training data
    model.fit(x_train, y_train)

    # Predict classes for the training data
    y_pred_train = model.predict(x_train)
    print(y_pred_train)
    # Assess equality with expected values
    np.testing.assert_allclose(y_pred_train, y_train)

    # Predict classes for new data
    y_pred_test = model.predict([[14, 15, 16], [4, 5, 6]])
    print(y_pred_test)
    # Assess equality with expected values
    np.testing.assert_allclose(y_pred_test, [1, 0])


def test_preprocessing():
    """
    Test preprocessing operations
    """

    x = [[0, 15], [1, -10]]

    # Scale data according to computed scaling values
    x_scaled = StandardScaler().fit(x).transform(x)
    print(x_scaled)

    # Assess standrdization results (mean=0, std=1 for all features)
    np.testing.assert_allclose(x_scaled.mean(axis=1), [0, 0])
    np.testing.assert_allclose(x_scaled.std(axis=1), [1, 1])


def test_pipelines_and_evaluation():
    """
    Test pipelines and model evaluation
    """

    # Load the iris dataset and split it into train and test sets
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    # Create a pipeline object
    pipeline = make_pipeline(StandardScaler(), LogisticRegression())

    # Fit the whole pipeline to the training set
    pipeline.fit(x_train, y_train)

    # We can now use it like any other estimator
    y_pred = pipeline.predict(x_test)

    # Assess model accuracy on test data
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_acc:.5f}")
    assert test_acc > 0.97

    # Compute classification metrics (precision, recall, f1-score) on th test set
    print(classification_report(y_test, y_pred))


# Standalone execution
if __name__ == "__main__":
    test_fitting_predicting()
    test_preprocessing()
    test_pipelines_and_evaluation()
