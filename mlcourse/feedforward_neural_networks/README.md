# Feedforward Neural Networks

## Classify 2D dataset

This [example](test_feedforward_neural_network_2d_data.py) trains a classifying model on a 2D dataset. It is designed to mimic the experience of the [TensorFlow Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=2,2&seed=0.17539&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false).

The model is created as an instance of the [Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) class. This is a shortcut syntax.

## Recognize fashion images

Thie [example](test_feedforward_neural_network_fashion_images.py) trains a model to classify fashion images.

The model is defined as a subclass of the [Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) class. The is the standard way to create models in PyTorch.

The first operation applied to model inputs is the [Flatten](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html) layer. It reshapes the input images of shape (1, 28, 28) into a 1D tensor (a vector) processed by the linear layers.
