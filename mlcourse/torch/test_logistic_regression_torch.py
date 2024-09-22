"""
Logistic Regression with PyTorch
"""

# Relax some linting rules for test code
# pylint: disable=missing-docstring,duplicate-code,too-many-locals,too-many-statements

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import torch
from torch import nn
from torch.utils.data import DataLoader


def test_logistic_regression(show_plots=False):
    print(f"PyTorch version: {torch.__version__}")

    # Accessing GPU device if available, or failing back to CPU
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device} device")

    # Configuration and hyperparameters
    n_samples = 1000
    input_dim = 2
    output_dim = 4  # Number of classes
    batch_size = 32
    n_epochs = 60
    learning_rate = 0.001

    # Generate a toy dataset
    inputs, targets = make_classification(
        n_samples=n_samples,
        n_features=input_dim,
        n_redundant=0,
        n_informative=input_dim,
        n_classes=output_dim,
        n_clusters_per_class=1,
    )

    if show_plots:
        # Plot dataset
        plt.scatter(
            inputs[:, 0], inputs[:, 1], marker="o", c=targets, s=25, edgecolor="k"
        )
        plt.show()

    # Convert dataset to PyTorch tensors
    x_train = torch.from_numpy(inputs).float().to(device)
    y_train = torch.from_numpy(targets).int().to(device)

    # Create data loader
    clf_dataloader = DataLoader(
        list(zip(x_train, y_train)), batch_size=batch_size, shuffle=True
    )

    # Number of samples
    n_samples = len(clf_dataloader.dataset)

    # Number of batches in an epoch (= n_samples / batch_size, rounded up)
    n_batches = len(clf_dataloader)

    # Logistic regression model
    model = nn.Linear(in_features=input_dim, out_features=output_dim).to(device)

    # nn.CrossEntropyLoss computes softmax internally
    criterion = nn.CrossEntropyLoss()

    # Mini-batch stochastic gradient descent
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(n_epochs):
        # Total loss for epoch, divided by number of batches to obtain mean loss
        epoch_loss = 0

        # Number of correct predictions in an epoch, used to compute epoch accuracy
        n_correct = 0

        for x_batch, y_batch in clf_dataloader:
            # Forward pass
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            # Gradient descent step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # Accumulate data for epoch metrics: loss and number of correct predictions
                epoch_loss += loss.item()
                n_correct += (
                    (model(x_batch).argmax(dim=1) == y_batch).float().sum().item()
                )

        # Compute epoch metrics
        mean_loss = epoch_loss / n_batches
        epoch_acc = n_correct / n_samples

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{(epoch + 1):3}/{n_epochs:3}] finished. Mean loss: {mean_loss:.5f}. Accuracy: {epoch_acc * 100:.2f}%"
            )


# Standalone execution
if __name__ == "__main__":
    test_logistic_regression(show_plots=True)
