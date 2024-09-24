"""
Feedforward Neural Network a.k.a. MultiLayer Perceptron (MLP) trained on fashion images
"""

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def plot_fashion_images(data, device, model=None):
    """Plot some images with their associated or predicted classes"""

    # Classes, i.e. fashion categories associated to images and indexed by label
    fashion_classes = (
        "T-Shirt",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle Boot",
    )

    figure = plt.figure()
    cols, rows = 7, 4
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label = data[sample_idx]
        figure.add_subplot(rows, cols, i)

        # Title is either true or predicted label
        if model is None:
            title = fashion_classes[label]
        else:
            # Add a dimension (to match expected shape with batch size) and store image on device memory
            x_img = img[None, :].to(device)
            # Compute predicted label for image
            # Even if the model outputs unormalized logits, argmax gives the predicted label
            pred_label = model(x_img).argmax(dim=1).item()
            title = f"{fashion_classes[pred_label]}?"
        plt.title(title)

        plt.axis("off")
        plt.imshow(img.cpu().detach().numpy().squeeze(), cmap="gray")
    plt.show()


def plot_loss_acc(history, dataset="Training"):
    """Plot training loss and accuracy. Takes a Keras-like History object as parameter"""

    loss_values = history["loss"]
    recorded_epochs = range(1, len(loss_values) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(recorded_epochs, loss_values, ".--", label=f"{dataset} loss")
    ax1.set_ylabel("Loss")
    ax1.legend()

    acc_values = history["acc"]
    ax2.plot(recorded_epochs, acc_values, ".--", label=f"{dataset} accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.legend()

    final_loss = loss_values[-1]
    final_acc = acc_values[-1]
    fig.suptitle(
        f"{dataset} loss: {final_loss:.5f}. {dataset} accuracy: {final_acc*100:.2f}%"
    )
    plt.show()


def fetch_fashion_dataset(data_folder):
    """Download the Fashion-MNIST images dataset"""

    # Download and construct the Fashion-MNIST images dataset
    train_dataset = datasets.FashionMNIST(
        root=f"{data_folder}",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    test_dataset = datasets.FashionMNIST(
        root=f"{data_folder}",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    return train_dataset, test_dataset


def load_fashion_dataset(train_dataset, test_dataset, batch_size):
    """Load the Fashion-MNIST images dataset as batches"""

    # Load the dataset
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class FashionNet(nn.Module):
    """Neural network for fashion articles classification"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        # Flatten the input image of shape (1, 28, 28) into a vector of shape (28*28,)
        self.flatten = nn.Flatten()

        # Define a sequential stack of linear layers and activation functions
        self.layer_stack = nn.Sequential(
            # Hidden layer with 28x28=784 inputs
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            # Output layer
            nn.Linear(in_features=64, out_features=output_dim),
        )

    def forward(self, x):
        """Define the forward pass of the model"""

        # Apply flattening to input
        x = self.flatten(x)

        # Compute output of layer stack
        logits = self.layer_stack(x)

        # Logits are a vector of raw (non-normalized) predictions
        # This vector contains 10 values, one for each possible class
        return logits


def fit(model, dataloader, criterion, optimizer, n_epochs, device):
    """Train a model on a dataset, using a predefined gradient descent optimizer"""

    # Object storing training history
    history = {"loss": [], "acc": []}

    # Number of samples
    n_samples = len(dataloader.dataset)

    # Number of batches in an epoch (= n_samples / batch_size, rounded up)
    n_batches = len(dataloader)

    print(f"Training started! {n_samples} samples. {n_batches} batches per epoch")

    # Train the model
    for epoch in range(n_epochs):
        # Total loss for epoch, divided by number of batches to obtain mean loss
        total_loss = 0

        # Number of correct predictions in an epoch, used to compute epoch accuracy
        n_correct = 0

        for x_batch, y_batch in dataloader:
            # Copy batch data to GPU memory (if available)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Forward pass
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            # Backprop and gradient descent step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # Accumulate data for epoch metrics: loss and number of correct predictions
                total_loss += loss.item()
                n_correct += (
                    (model(x_batch).argmax(dim=1) == y_batch).float().sum().item()
                )

        # Compute epoch metrics
        mean_loss = total_loss / n_batches
        epoch_acc = n_correct / n_samples

        print(
            f"Epoch [{(epoch + 1):3}/{n_epochs:3}] finished. Mean loss: {mean_loss:.5f}. Accuracy: {epoch_acc * 100:.2f}%"
        )

        # Record epoch metrics for later plotting
        history["loss"].append(mean_loss)
        history["acc"].append(epoch_acc)

    return history


def evaluate(model, dataloader, device):
    """Evaluate a model in inference mode"""

    # Number of samples
    n_samples = len(dataloader.dataset)

    # Compute model accuracy on test data
    with torch.no_grad():
        n_correct = 0

        for x_batch, y_batch in dataloader:
            # Copy batch data to GPU memory (if available)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Compute number of correct prediction for the current batch
            n_correct += (model(x_batch).argmax(dim=1) == y_batch).float().sum().item()

        test_acc = n_correct / n_samples
        print(f"Evaluated accuracy: {test_acc * 100:.2f}%")


def test_feedforward_neural_network_fashion_images(show_plots=False):
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
    data_folder = "./_output"
    input_dim = 784  # 28x28
    hidden_dim = 64  # Number of neurons on the hidden layer
    output_dim = 10  # Number of classes
    batch_size = 64
    n_epochs = 10
    learning_rate = 0.001

    train_dataset, test_dataset = fetch_fashion_dataset(data_folder)

    train_loader, test_loader = load_fashion_dataset(
        train_dataset, test_dataset, batch_size
    )

    model = FashionNet(input_dim, hidden_dim, output_dim).to(device)
    print(model)

    # nn.CrossEntropyLoss computes softmax internally
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer for GD
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = fit(model, train_loader, criterion, optimizer, n_epochs, device)

    # Evaluate model performance on test data
    evaluate(model, test_loader, device)

    if show_plots:
        # Improve plots appearance
        sns.set_theme()

        # Plot training history
        plot_loss_acc(history)

        # Plot model predictions for some test images
        plot_fashion_images(test_dataset, device, model)


# Standalone execution
if __name__ == "__main__":
    test_feedforward_neural_network_fashion_images(show_plots=True)
