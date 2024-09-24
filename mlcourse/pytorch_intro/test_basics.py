"""
PyTorch Basics
"""

import math
import numpy as np
from sklearn.datasets import make_circles
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models


def test_tensor_manipulation():
    """Test tensors manipulation"""

    # Create a 1D tensor with predefined values
    x = torch.tensor([5.5, 3])
    assert x.shape == torch.Size([2])

    # Create a 2D tensor filled with random numbers from a uniform distribution
    x = torch.rand(5, 3)
    assert x.shape == torch.Size([5, 3])

    # Addition operator
    y1 = x + 2

    # Addition method
    y2 = torch.add(x, 2)
    assert torch.equal(y1, y2)

    # Create a deep copy of a tensor
    # detach() removes its output from the computational graph (no gradient computation)
    # https://stackoverflow.com/a/62496418
    x1 = x.detach().clone()

    # In-place addition: tensor is mutated
    x.add_(2)
    assert torch.equal(x, x1 + 2)

    # Indexing is similar to the Python/NumPy syntax
    # Example: set all values of second column to zero
    x[:, 1] = 0

    # PyTorch allows a tensor to be a view of an existing tensor
    # View tensors share the same underlying data with their base tensor
    # Example : reshaping into a 1D tensor (a vector)
    x_view = x.view(15)
    assert x_view.shape == torch.Size([15])

    # The dimension identified by -1 is inferred from other dimensions
    assert x.view(-1, 5).shape == torch.Size([3, 5])
    assert x.view(
        -1,
    ).shape == torch.Size([15])

    # The reshape() function mimics the NumPy API
    # Example: reshaping into a (3,5) tensor, creating a view if possible
    assert x.reshape(3, -1).shape == torch.Size([3, 5])

    # Create a PyTorch tensor from a NumPy array
    a = np.random.rand(2, 2)
    b = torch.from_numpy(a)
    assert b.shape == torch.Size([2, 2])

    # Obtain a NumPy array from a PyTorch tensor
    a = torch.rand(2, 2)
    b = a.numpy()
    assert b.shape == (2, 2)

    # Accessing GPU device if available, or failing back to CPU
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"PyTorch version: {torch.__version__}. using {device} device")

    # Copy tensor to GPU memory (if available)
    x_device = x.to(device)

    # Create a copy of a GPU-based tensor in CPU memory
    _ = x_device.cpu()

    # Obtain a NumPy array from a GPU-based tensor
    _ = x_device.detach().cpu().numpy()


def test_autodiff():
    """Test autograd engine"""

    # Example 1: basic operations

    # Create tensor with gradient computation activated
    # (By default, operations are not tracked on user-created tensors)
    x = torch.tensor(1.0, requires_grad=True)
    w = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)

    # Apply basic operations
    y = w * x + b
    assert y.requires_grad is True

    # Compute gradients
    y.backward()

    # Print the gradients
    assert x.grad == 2  # x.grad = dy/dx = w
    assert w.grad == 1  # w.grad = dy/dw = x
    assert b.grad == 1  # b.grad = dy/db

    # no_grad() avoids tracking operations history when gradients computation is not needed
    with torch.no_grad():
        y_no = w * x + b
        assert y_no.requires_grad is False

    # Example 2: a slighly more complex computational graph

    # Create two tensors with gradient computation activated
    x1 = torch.tensor([2.0], requires_grad=True)
    x2 = torch.tensor([5.0], requires_grad=True)

    # y = f(x1,x2) = ln(x1) + x1.x2 - sin(x2)
    v1 = torch.log(x1)
    v2 = x1 * x2
    v3 = torch.sin(x2)
    v4 = v1 + v2
    y = v4 - v3

    # Compute gradients
    y.backward()

    # dy/dx1 = 1/x1 + x2 = 1/2 + 5
    assert x1.grad == 5.5
    # dy/dx2 = x1 - cos(x2) = 2 - cos(5) = 1.7163...
    assert x2.grad == 2 - torch.cos(torch.tensor(5))


def test_dataset_loading():
    """Test dataset loading"""

    # Relative path for saving and loading models
    data_folder = "./_output"

    # Number of samples in each batch
    batch_size = 32

    # Example 1: loading an integrated dataset

    # Download and construct the MNIST handwritten digits training dataset
    mnist = datasets.MNIST(
        root=data_folder, train=True, transform=transforms.ToTensor(), download=True
    )

    # Fetch one data pair (read data from disk)
    image, label = mnist[0]
    # MNIST samples are bitmap images of shape (color_depth, height, width)
    # Color depth is 1 for grayscale images
    assert image.shape == torch.Size([1, 28, 28])
    # Image label is a scalar value
    assert isinstance(label, int)

    # Data loader (this provides queues and threads in a very simple way).
    mnist_dataloader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)

    # Number of batches in an epoch (= n_samples / batch_size, rounded up)
    n_batches = len(mnist_dataloader)
    assert n_batches == math.ceil(len(mnist) / batch_size)

    # Loop-based iteration is the most convenient way to train models on batched data
    for x_batch, y_batch in mnist_dataloader:
        # x_batch contains inputs for the current batch
        assert x_batch.shape == torch.Size([batch_size, 1, 28, 28])
        # y_batch contains targets for the current batch
        assert y_batch.shape == torch.Size([batch_size])
        # ... (Training code for the current batch should be written here)

    # Example 2: loading a scikit-learn dataset

    # Number of generated samples
    n_samples = 500

    # Generate 2D data (a large circle containing a smaller circle)
    inputs, targets = make_circles(n_samples=n_samples, noise=0.1, factor=0.3)
    assert inputs.shape == (n_samples, 2)
    assert targets.shape == (n_samples,)

    # Create tensor for inputs
    x_train = torch.from_numpy(inputs).float()
    assert x_train.shape == torch.Size([n_samples, 2])

    # Create tensor for targets (labels)
    y_train = torch.from_numpy(targets).int()
    assert y_train.shape == torch.Size([n_samples])

    # Load data as randomized batches for training
    _ = DataLoader(list(zip(x_train, y_train)), batch_size=batch_size, shuffle=True)

    # ... (Exploit dataloader as seen above)

    # Example 3: loading a custom dataset

    class CustomDataset(Dataset):
        """A custom Dataset class must implement three functions: __init__, __len__, and __getitem__"""

        def __init__(self):
            # Init internal state (file paths, etc))
            # ...
            pass

        def __len__(self):
            # Return the number of samples in the dataset
            # ...
            return 1

        def __getitem__(self, index):
            # Load, preprocess and return one data sample (inputs and label)
            # ...
            pass

    custom_dataset = CustomDataset()
    _ = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True)

    # ... (Exploit dataloader as seen above)


def test_model_loading_saving():
    """Test model loading and saving"""

    # Relative path for saving and loading models
    model_folder = "./_output"

    # Download and load the pretrained model ResNet-18
    resnet = models.resnet18(weights="ResNet18_Weights.DEFAULT")

    # Save model parameters (recommended way of saving models)
    resnet_weights_filepath = f"{model_folder}/resnet_weights.pth"
    torch.save(resnet.state_dict(), resnet_weights_filepath)

    # Load untrained model
    resnet = models.resnet18()

    # Load saved weights
    resnet.load_state_dict(torch.load(resnet_weights_filepath, weights_only=True))

    # Print the architecture of the model
    print(resnet)


# Standalone execution
if __name__ == "__main__":
    test_tensor_manipulation()
    test_autodiff()
    test_dataset_loading()
    test_model_loading_saving()
