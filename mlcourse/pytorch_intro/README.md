# Introduction to PyTorch

## PyTorch Basics

This [example](test_basics.py) demonstrates several fundamental aspects of PyTorch:

- Tensor manipulation
- Autodifferentiation engine
- Dataset loading
- Model loading and saving

### References

- [PyTorch official website](https://pytorch.org)
- [PyTorch tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [A gentle introduction to autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [Writing Custom Datasets, DataLoaders and Transforms](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

## Linear Regression

This [example](test_linear_regression.py) uses the [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) class to implement linear regression on a simple 2D dataset.

After gradients computation, parameters are updated manually to better illustrate how [gradient descent](../../notes/gradient_descent/README.md) works.

## Logistic Regression

This [example](test_logistic_regression.py) uses the [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) class to implement logistic regression on a simple dataset generated via [scikit-learn](https://scikit-learn.org).

It uses the [cross-entropy](../../notes/classification_performance/README.md#choosing-a-loss-function-1) a.k.a. negative log-likelihood loss, implemented by the [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) class.

> PyTorch also offers the [NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) class implementing the negative log-likelihood loss. A key difference is that `CrossEntropyLoss` expects *logits*  (raw, unnormalized predictions) as inputs, and uses [LogSoftmax](https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax) to transform them into probabilities before computing its output. Using `CrossEntropyLoss` is equivalent to applying `LogSoftmax` followed by `NLLLoss` ([more details](https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81)).

An [optimizer](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) is used to update the model's parameters during gradient descent.

## Feedforward Neural Network

Thie [example](test_feedforward_neural_network.py) trains a model to classify fashion images. The dataset consists of:

- a training set containing 60,000 28x28 grayscale images, each of them associated with a label (fashion category) from 10 classes;
- a test set of 10,000 images with the same properties.

A [PyTorch class](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html) simplifies the loading process of this dataset.

The model is defined as a subclass of the [Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) class. The is the standard way to create models in PyTorch. Their constructor defines the layer architecture and their `forward()` method defines the forward pass of the model.

The first operation applied to model inputs is the [Flatten](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html) layer. It reshapes the input images of shape `(1, 28, 28)` into a 1D tensor (a vector) processed by the linear layers.

The training algorithm uses the [cross-entropy](../../notes/classification_performance/README.md#choosing-a-loss-function-1) a.k.a. negative log-likelihood loss, implemented by the [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) class.

> This is equivalent to combining the [LogSoftmax](https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax) and [NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) classes (see above).
