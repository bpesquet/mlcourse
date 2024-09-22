"""
Recognize handwritten digits
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.linear_model import SGDClassifier


def load_mnist_dataset():
    """Load the MNIST dataset"""

    # Load the MNIST digits dataset from sciki-learn
    images, targets = fetch_openml(
        "mnist_784", version=1, parser="pandas", as_frame=False, return_X_y=True
    )

    print(f"Images: {images.shape}. Targets: {targets.shape}")
    print(f"First 10 labels: {targets[:10]}")

    return images, targets


def plot_digits(images, n_digits=10):
    """Plot some images of the dataset"""

    # Temporary hide Seaborn grid lines
    with sns.axes_style("white"):
        plt.figure()
        for i in range(n_digits):
            digit = images[i].reshape(28, 28)
            _ = plt.subplot(2, 5, i + 1)
            plt.imshow(digit)
    plt.show()


def split_preprocess_dataset(images, targets):
    """Split the dataset between training and test sets"""

    # Split dataset into training and test sets
    train_images, test_images, train_targets, test_targets = train_test_split(
        images, targets, test_size=10000
    )

    # Rescale pixel values from [0,255] to [0,1]
    train_images, test_images = train_images / 255.0, test_images / 255.0

    print(
        f"Training images: {train_images.shape}. Training targets: {train_targets.shape}"
    )
    print(f"Test images: {test_images.shape}. Test targets: {test_targets.shape}")

    return train_images, test_images, train_targets, test_targets


def train_classifier(x, y):
    """Train a classifier on the dataset"""

    # Create a classifier using stochastic gradient descent and logistic loss
    sgd_model = SGDClassifier(loss="log_loss")

    # Train the model on data
    sgd_model.fit(x, y)

    return sgd_model


def evaluate_classifier(model, x, y):
    """Evaluate the performance of a classifier"""

    # The score() function computes accuracy of the SGDClassifier
    acc = model.score(x, y)
    print(f"Accuracy: {acc:.05f}")

    # Using cross-validation to better evaluate accuracy, using 3 folds
    cv_acc = cross_val_score(model, x, y, cv=3, scoring="accuracy")
    print(f"Cross-validation accuracy: {cv_acc}")

    # Compute several metrics about our 5/not 5 classifier
    print(classification_report(y, model.predict(x)))


def plot_conf_mat(model, x, y):
    """Plot the confusion matrix for a model, inputs and targets"""

    with sns.axes_style("white"):  # Temporary hide Seaborn grid lines
        _ = ConfusionMatrixDisplay.from_estimator(
            model, x, y, values_format="d", cmap=plt.colormaps.get_cmap("Blues")
        )
    plt.show()


def train_and_evaluate_classifier(x, y, show_plots):
    """Train and evaluate a classifier on a dataset"""

    model = train_classifier(x, y)

    evaluate_classifier(model, x, y)

    if show_plots:
        # Plot confusion matrix for the classifier
        plot_conf_mat(model, x, y)

    return model


def test_handwritten_digits(show_plots=False):
    """Main test function"""

    images, targets = load_mnist_dataset()

    if show_plots:
        # Improve plots appearance
        sns.set_theme()
        plot_digits(images)

    x_train, x_test, y_train, y_test = split_preprocess_dataset(images, targets)

    print("\n----- Binary classification -----")

    # Transform results into binary values
    # label is true for all 5s, false for all other digits
    y_train_5 = y_train == "5"
    y_test_5 = y_test == "5"

    binary_model = train_and_evaluate_classifier(x_train, y_train_5, show_plots)

    # Evaluate the trained classifier on test data
    evaluate_classifier(binary_model, x_test, y_test_5)

    print("\n----- Multiclass classification -----")

    multiclass_model = train_and_evaluate_classifier(x_train, y_train, show_plots)

    # Evaluate the trained classifier on test data
    evaluate_classifier(multiclass_model, x_test, y_test)


# Standalone execution
if __name__ == "__main__":
    test_handwritten_digits(show_plots=True)
