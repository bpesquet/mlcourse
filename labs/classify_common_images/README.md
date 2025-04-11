# Classify common images

The goal of this lab is to train a neural network to classify common images.

## Dataset

The [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60,000 32x32 colour images in 10 classes, with 6,000 images per class. The classes are completely mutually exclusive. There are 50,000 training images and 10,000 test images.

![Training outcome](images/cifar10.png)

## Platform

You may use either a local or remote Python environment for this lab.

The easiest way to obtain a working Python setup is by using a cloud-based [Jupyter notebook](https://jupyter.org/) execution platform like [Google Colaboratory](https://colab.research.google.com/), [Paperspace](https://www.paperspace.com/notebooks) or [Kaggle Notebooks](https://www.kaggle.com/code).

## Tools

This challenge can be tackled by writing the neural network code "by hand" or by using a dedicated library. A prominent choice is [PyTorch](https://pytorch.org/), which facilitates the creation and training of networks of any size.

Follow the following tutorials if you have little to no experience with this library:

- [PyTorch Fundamentals](https://github.com/bpesquet/pytorch-tutorial/tree/main/pytorch_tutorial/fundamentals)
- [Linear Regression with PyTorch](https://github.com/bpesquet/pytorch-tutorial/tree/main/pytorch_tutorial/linear_regression)
- [Logistic Regression with PyTorch](https://github.com/bpesquet/pytorch-tutorial/tree/main/pytorch_tutorial/logistic_regression)
- [MultiLayer Perceptron with PyTorch](https://github.com/bpesquet/pytorch-tutorial/tree/main/pytorch_tutorial/multilayer_perceptron)
- [Convolutional Neural Network with PyTorch](https://github.com/bpesquet/pytorch-tutorial/tree/main/pytorch_tutorial/convolutional_neural_network)

## Training process

You should start by training àn MLP model with one hidden layer on the dataset. Make sure to define a model architecture suited to the training data, and to use the appropriate loss function for this task.

Once this is done, train a convolutional neural network on the same dataset and assess the difference in results between the two architectures.

After training, plot some images from the test set with the associated model predictions. You may use the following function to do so.

```python
def plot_images(dataset, device, model=None):
    """
    Plot images from a dataset with (optionally) the associated model predictions.

    Args:
        dataset (torch.Dataset): a PyTorch dataset
        device (torch.device): a PyTorch device
        model (torch.nn.Module): a PyTorch model
    """
    cifar10_classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    figure = plt.figure()
    cols, rows = 5, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]

        # Unnormalize image (see transform during dataset loading)
        img_plt = img / 2 + 0.5
        # Cnvert the tensor containing the image to a NumPy array.
        # Its dimensions are transposed from (color_channels, height, width) to (height, width, color_channels) for plotting
        img_plt = np.transpose(img_plt.cpu().detach().numpy(), (1, 2, 0))
        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        plt.imshow(img_plt)

        # Title is the class associated to either ground truth or predicted label
        if model is None:
            title = cifar10_classes[label]
        else:
            # Add a dimension to match expected shape with batch size, and store image on device memory
            x_img = img[None, :].to(device)
            # Compute predicted label for image.
            # Even if the model outputs unormalized logits, argmax gives us the predicted label
            pred_label = model(x_img).argmax(dim=1).item()
            title = f"{cifar10_classes[pred_label]}?"
        plt.title(title)

    # Return the new plot
    return plt.gcf()
```

## Extra work

- Compute accuracy for both models on the test dataset.
- Try to achieve an accuracy of `70%` or more on the test set with one of the models.
- Integrate [TensorBoard](https://www.tensorflow.org/tensorboard) for an easier visualization of the training results.
