"""
Convolutional Neural Network (CNN) a.k.a. convnet
"""

# Relax some linting rules for test code
# pylint: disable=missing-docstring,duplicate-code,too-many-locals,too-many-statements

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Convnet(nn.Module):
    """Convnet for fashion articles classification"""

    def __init__(self, n_classes=10):
        super().__init__()

        # Define a sequential stack
        self.layer_stack = nn.Sequential(
            # 2D convolution, output dimensions: (32, 26, 26)
            # Without padding, output_dim = (input_dim - kernel_size + 1) / stride
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            # Max pooling, output dimensions: (32, 13, 13)
            nn.MaxPool2d(kernel_size=2),
            # 2D convolution, output dimensions: (64, 11, 11)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            # Max pooling, output dimensions: (64, 5, 5)
            nn.MaxPool2d(kernel_size=2),
            # Flattening layer, output dimensions: (64x5x5 = 1600,)
            nn.Flatten(),
            nn.Linear(in_features=1600, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=n_classes),
        )

    def forward(self, x):
        """Define the forward pass of the model"""

        # Compute output of layer stack
        logits = self.layer_stack(x)

        # Logits are a vector of raw (non-normalized) predictions
        # This vector contains 10 values, one for each possible class
        return logits


def test_convolutional_neural_network(show_plots=False):
    print(f"PyTorch version: {torch.__version__}")

    # Accessing GPU device if available, or failing back to CPU
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device} device")

    # Configuration and hyperparameters
    data_folder = "./_output"
    batch_size = 64
    n_epochs = 10
    learning_rate = 0.001

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

    # Load the dataset
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    # Number of samples
    n_samples = len(train_loader.dataset)

    # Number of batches in an epoch (= n_samples / batch_size, rounded up)
    n_batches = len(train_loader)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Create the convolutional network
    model = Convnet().to(device)
    print(model)

    # nn.CrossEntropyLoss computes softmax internally
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer for GD
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(n_epochs):
        # Total loss for epoch, divided by number of batches to obtain mean loss
        epoch_loss = 0

        # Number of correct predictions in an epoch, used to compute epoch accuracy
        n_correct = 0

        for x_batch, y_batch in train_loader:
            # Copy batch data to GPU memory (if available)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

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

        print(
            f"Epoch [{(epoch + 1):3}/{n_epochs:3}] finished. Mean loss: {mean_loss:.5f}. Accuracy: {epoch_acc * 100:.2f}%"
        )

    # Compute model accuracy on test data
    with torch.no_grad():
        n_correct = 0

        for x_batch, y_batch in test_loader:
            # Copy batch data to GPU memory (if available)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            y_pred = model(x_batch)

            n_correct += (model(x_batch).argmax(dim=1) == y_batch).float().sum().item()

        test_acc = n_correct / len(test_loader.dataset)
        print(f"Test accuracy: {test_acc * 100:.2f}%")

    if show_plots:
        plot_images(data=test_dataset, device=device, model=model)


def plot_images(data, device, model=None):
    """Plot some images with their associated or predicted labels"""

    # Items, i.e. fashion categories associated to images and indexed by label
    fashion_items = (
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
    cols, rows = 5, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label = data[sample_idx]
        figure.add_subplot(rows, cols, i)

        # Title is item associated to either true or predicted label
        if model is None:
            title = fashion_items[label]
        else:
            # Add a dimension (to match expected shape with batch size) and store image on device memory
            x_img = img[None, :].to(device)
            # Compute predicted label for image
            # Even if the model outputs unormalized logits, argmax gives the predicted label
            pred_label = model(x_img).argmax(dim=1).item()
            title = f"{fashion_items[pred_label]}?"
        plt.title(title)

        plt.axis("off")
        plt.imshow(img.cpu().detach().numpy().squeeze(), cmap="gray")
    plt.show()


# Standalone execution
if __name__ == "__main__":
    test_convolutional_neural_network(show_plots=True)
