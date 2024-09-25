"""
Feedforward Neural Network a.k.a. MultiLayer Perceptron (MLP) trained on a 2D dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.datasets import make_circles
import torch
from torch import nn
from torch.utils.data import DataLoader


def plot_2d_dataset(x, y):
    """Plot a 2-dimensional dataset with associated classes"""

    plt.figure()
    plt.plot(x[y == 0, 0], x[y == 0, 1], "or", label=0)
    plt.plot(x[y == 1, 0], x[y == 1, 1], "ob", label=1)
    plt.legend()
    plt.title("2D dataset for binary classification")
    plt.show()


def plot_decision_boundary(model, x, y, device):
    """Plot the frontier between classes for a 2-dimensional dataset.
    Note: x and y must be NumPy arrays, not PyTorch tensors."""

    plt.figure()
    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Compute model output for the whole grid
    z = model(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float().to(device))
    z = z.reshape(xx.shape)
    # Convert PyTorch tensor to NumPy
    zz = z.detach().cpu().numpy()
    # Plot the contour and training examples
    plt.contourf(xx, yy, zz, cmap=plt.colormaps.get_cmap("Spectral"))
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_bright)
    plt.title("Classification results")
    plt.show()


def create_2d_dataset(show_plots):
    """Create a 2D dataset"""

    # Generate 2D data (a large circle containing a smaller circle) with 2 classes
    inputs, targets = make_circles(n_samples=500, noise=0.1, factor=0.3)
    print(f"2D dataset generated. Inputs: {inputs.shape}. Targets: {targets.shape}")

    if show_plots:
        # Improve plots appearance
        sns.set_theme()

        plot_2d_dataset(inputs, targets)

    return inputs, targets


def load_2d_dataset(inputs, targets, batch_size, device):
    """Load a dataset as batches"""

    # Convert dataset to PyTorch tensors
    x_train = torch.from_numpy(inputs).float().to(device)
    # PyTorch loss function expects float results of shape (batch_size, 1) instead of (batch_size,)
    # So we add a new axis and convert them to floats
    y_train = torch.from_numpy(targets[:, np.newaxis]).float().to(device)

    print(f"x_train: {x_train.shape}. y_train: {y_train.shape}")

    # Load data as randomized batches for training
    dataloader = DataLoader(
        list(zip(x_train, y_train)), batch_size=batch_size, shuffle=True
    )

    return dataloader


def count_parameters(model, trainable=True):
    """Return the total number of (trainable) parameters for a model"""

    return (
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable
        else sum(p.numel() for p in model.parameters())
    )


def fit(model, dataloader, criterion, learning_rate, n_epochs, device):
    """Train a model on a dataset, using manual parameters update"""

    for epoch in range(n_epochs):
        for x_batch, y_batch in dataloader:
            # Copy batch data to GPU memory (if available)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Forward pass
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            # Reset the gradients to zero before running the backward pass
            # Avoids accumulating gradients between GD steps
            model.zero_grad()

            # Compute gradients of loss w.r.t. each parameter
            loss.backward()

            # no_grad() avoids tracking operations history when gradients computation is not needed
            with torch.no_grad():
                # Manual gradient descent step: update the weights in the opposite direction of their gradient
                for param in model.parameters():
                    param -= learning_rate * param.grad

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{(epoch + 1):3}/{n_epochs:3}] finished. Loss: {loss.item():.5f}"
            )


def test_classify_2d_data(show_plots=False):
    """Main test function"""

    # pylint: disable=duplicate-code
    # Accessing GPU device if available, or failing back to CPU
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        # else "mps" if torch.backends.mps.is_available()
        # There are performance issues with MPS backend for MLP-like models: using cpu instead
        else "cpu"
    )
    print(f"PyTorch version: {torch.__version__}. using {device} device")
    # pylint: enable=duplicate-code

    # Configuration values and hyperparameters
    input_dim = 2
    hidden_dim = 3  # Number of neurons on the hidden layer
    output_dim = 1  # Only one output for binary classification
    batch_size = 5
    n_epochs = 50
    learning_rate = 0.1

    inputs, targets = create_2d_dataset(show_plots)

    dataloader = load_2d_dataset(inputs, targets, batch_size, device=device)

    # Create the model as a sequential stack of layers
    model = nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=hidden_dim),
        nn.Tanh(),
        nn.Linear(in_features=hidden_dim, out_features=output_dim),
        nn.Sigmoid(),
    ).to(device)
    print(model)
    print(f"Number of trainable parameters: {count_parameters(model)}")

    # Binary cross-entropy loss
    criterion = nn.BCELoss()

    # Training loop
    fit(model, dataloader, criterion, learning_rate, n_epochs, device)

    if show_plots:
        plot_decision_boundary(model, inputs, targets, device)


# Standalone execution
if __name__ == "__main__":
    test_classify_2d_data(show_plots=True)
