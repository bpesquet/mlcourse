# Feedforward Neural Networks

## Classify 2D dataset

This [example](test_feedforward_neural_network_2d_data.py) trains a classifying model on a 2D dataset generated with [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html). It is designed to mimic the experience of the [TensorFlow Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=2,2&seed=0.17539&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false).

Generated data (NumPy tensors) needs to be converted to PyTorch tensors before training a PyTorch-based model. These new tensors are stored in the memory of the available device (GPU ou CPU).

In order to use [mini-batch SGD](../../notes/gradient_descent/README.md#mini-batch-sgd), data needs to be passed to the model as small, randomized batches during training. The Pytorch [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) class abstracts away this complexity.

A PyTorch model is defined by combining elementary blocks, known as *modules*. Here, the model is created as an instance of the [Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) class. This is a shortcut syntax.

PyTorch provides out-of-the-box implementations for many gradient descent optimization algorithms ([Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html), [RMSProp](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html), etc). This example updates parameters "by hand" to better illustrate the gradient descent algorithm.

## Recognize fashion images

Thie [example](test_feedforward_neural_network_fashion_images.py) trains a model to classify fashion images. The dataset consists of:

- a training set containing 60,000 28x28 grayscale images, each of them associated with a label (fashion category) from 10 classes;
- a test set of 10,000 images with the same properties.

A [PyTorch class](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html) simplifies the loading process of this dataset.

The model is defined as a subclass of the [Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) class. The is the standard way to create models in PyTorch. Their constructor creates the layer architecture and their `forward()` method defines the forward pass of the model.

The first operation applied to model inputs is the [Flatten](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html) layer. It reshapes the input images of shape (1, 28, 28) into a 1D tensor (a vector) processed by the linear layers.

The training algorithm uses the [cross-entropy](../../notes/assessing_classification_performance/README.md#choosing-a-loss-function-1) a.k.a. negative log-likelihood loss, implemented by the [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) class.

> PyTorch also offers the [NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) class implementing the negative log-likelihood loss. A key difference is that `CrossEntropyLoss` expects *logits*  (raw, unnormalized predictions) as inputs, and uses [LogSoftmax](https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax) to transform them into probabilities before computing its output. Using `CrossEntropyLoss` is equivalent to applying `LogSoftmax` followed by `NLLLoss` ([more details](https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81)).
