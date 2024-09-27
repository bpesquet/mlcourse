# Feedforward Neural Networks

Thie [example](test_feedforward_neural_network.py) trains a model to classify fashion images. The dataset consists of:

- a training set containing 60,000 28x28 grayscale images, each of them associated with a label (fashion category) from 10 classes;
- a test set of 10,000 images with the same properties.

A [PyTorch class](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html) simplifies the loading process of this dataset.

The model is defined as a subclass of the [Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) class. The is the standard way to create models in PyTorch. Their constructor defines the layer architecture and their `forward()` method defines the forward pass of the model. That is all PyTorch needs to compute gradients thanks to its [autodifferentiation engine](../pytorch_intro/README.md#pytorch-basics).

The first operation applied to model inputs is the [Flatten](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html) layer. It reshapes the input images of shape `(1, 28, 28)` into a 1D tensor (a vector) processed by the linear layers.

The training algorithm uses the [cross-entropy](../../notes/classification_performance/README.md#choosing-a-loss-function-1) a.k.a. negative log-likelihood loss, implemented by the [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) class.

> This is equivalent to combining the [LogSoftmax](https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax) and [NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) classes (see [here](../pytorch_intro/README.md#logistic-regression)).
