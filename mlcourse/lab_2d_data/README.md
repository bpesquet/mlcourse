# Feedforward Neural Networks

## Classify 2D dataset

This [example](test_feedforward_neural_network_2d_data.py) trains a classifying model on a 2D dataset generated with [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html). It is designed to mimic the experience of the [TensorFlow Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=2,2&seed=0.17539&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false).

Generated data (NumPy tensors) needs to be converted to PyTorch tensors before training a PyTorch-based model. These new tensors are stored in the memory of the available device (GPU ou CPU).

In order to use [mini-batch SGD](../../notes/gradient_descent/README.md#mini-batch-sgd), data needs to be passed to the model as small, randomized batches during training. The Pytorch [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) class abstracts away this complexity.

A PyTorch model is defined by combining elementary blocks, known as *modules*. Here, the model is created as an instance of the [Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) class. This is a shortcut syntax.

PyTorch provides out-of-the-box implementations for many gradient descent optimization algorithms ([Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html), [RMSProp](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html), etc). This example updates parameters "by hand" to better illustrate the gradient descent algorithm.
