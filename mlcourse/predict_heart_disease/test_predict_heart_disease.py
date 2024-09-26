#!/usr/bin/env python
"""
Prédict heart disease
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def load_dataset(url):
    """Load a dataset in a pandas DataFrame"""

    # Load dataset from url
    dataset = pd.read_csv(url)

    print(f"dataset: {dataset.shape}")

    # Print dataset info
    dataset.info()

    # Print 10 random samples
    dataset.sample(n=10)

    # Print descriptive statistics for all numerical attributes
    dataset.describe()

    return dataset


def split_dataset(df):
    """Split dataset between inputs/targets and training/test sets"""

    # Print distribution of target values
    df["target"].value_counts()

    # Separate inputs from targets
    # Target attribute is removed to create inputs
    df_x = df.drop("target", axis="columns")

    # Targets are stored separately in a new variable
    df_y = df["target"]

    print(f"df_x: {df_x.shape}. df_y: {df_y.shape}")

    # Split dataset between training and test sets
    # A unique call to train_test_split is mandatory to maintain inputs/target correspondance between samples
    df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(
        df_x, df_y, test_size=0.2
    )

    print(f"df_x_train: {df_x_train.shape}. df_y_train: {df_y_train.shape}")
    print(f"df_x_test: {df_x_test.shape}. df_y_test: {df_y_test.shape}")

    return df_x_train, df_x_test, df_y_train, df_y_test


def preprocess_dataset(df_x_train, df_y_train):
    """Preprocess the dataset"""

    # Print numerical and categorical features
    num_features = df_x_train.select_dtypes(include=[np.number]).columns
    print(num_features)

    cat_features = df_x_train.select_dtypes(include=[object]).columns
    print(cat_features)

    # Print all values for the "thal" categorical feature
    df_x_train["thal"].value_counts()

    # Preprocess data to have similar scales and only numerical values

    # This pipeline standardizes numerical features
    # It also one-hot encodes the categorical features
    full_pipeline = ColumnTransformer(
        [
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(), cat_features),
        ]
    )

    # Apply all preprocessing operations to the training set through pipelines
    x_train = full_pipeline.fit_transform(df_x_train)

    # Print preprocessed data shape and first sample
    # "ocean_proximity" attribute has 5 different values
    # To represent them, one-hot encoding has added 4 features to the dataset
    print(f"x_train: {x_train.shape}")
    print(x_train[0])

    # Transform the targets DataFrame into a plain tensor
    y_train = df_y_train.to_numpy()

    return x_train, y_train, full_pipeline


def train_model(x_train, y_train):
    """Train a model on the training set"""

    # Fit a SGD classifier (logistic regression) to the training set
    sgd_model = SGDClassifier(loss="log_loss")
    sgd_model.fit(x_train, y_train)

    # Use cross-validation to evaluate accuracy, using 3 folds
    cv_acc = cross_val_score(sgd_model, x_train, y_train, cv=3, scoring="accuracy")
    print(f"Cross-validation accuracy: {cv_acc}")

    return sgd_model


def evaluate_model(model, x, y):
    """Evaluate the performance of a model"""

    # Compute model predictions
    y_pred = model.predict(x)

    # Compute precision, recall and f1-score for the SGD classifier
    print(classification_report(y, y_pred))


def evaluate_model_on_test_data(model, df_x_test, df_y_test, pipeline):
    """Evaluate a trained model on the test dataset"""

    # Apply preprocessing operations to test inputs
    # Calling transform() and not fit_transform() uses preprocessing values computed on training set
    x_test = pipeline.transform(df_x_test)

    # Transform the targets DataFrame into a plain tensor
    y_test = df_y_test.to_numpy()

    print(f"x_test: {x_test.shape}. y_test: {y_test.shape}")

    evaluate_model(model, x_test, y_test)


def plot_conf_mat(model, x, y):
    """Plot the confusion matrix for a model, inputs and targets"""

    with sns.axes_style("white"):  # Temporary hide Seaborn grid lines
        _ = ConfusionMatrixDisplay.from_estimator(
            model, x, y, values_format="d", cmap=plt.colormaps.get_cmap("Blues")
        )


def test_predict_heart_disease(show_plots=False):
    """Main test function"""

    df_heart = load_dataset(
        "https://raw.githubusercontent.com/bpesquet/mlcourse/main/datasets/heart.csv"
    )

    df_x_train, df_x_test, df_y_train, df_y_test = split_dataset(df_heart)

    x_train, y_train, preprocessing_pipeline = preprocess_dataset(
        df_x_train, df_y_train
    )

    model = train_model(x_train, y_train)

    if show_plots:
        # Improve plots appearance
        sns.set_theme()

        plot_conf_mat(model, x_train, y_train)
        plt.show()

    evaluate_model(model, x_train, y_train)

    evaluate_model_on_test_data(model, df_x_test, df_y_test, preprocessing_pipeline)


# Standalone execution
if __name__ == "__main__":
    test_predict_heart_disease(show_plots=True)
