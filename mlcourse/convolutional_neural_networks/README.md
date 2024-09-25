# Convolutional Neural Networks

This [example](test_convolutional_neural_network.py) builds a convnet to recognize fashion items. The architecture of the model is as follows.

![FashionNet architecture](images/fashionnet.png)

The model leverages the following PyTorch classes:

- [Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html): an ordered container of modules.
- [Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html): for 2D convolutions.
- [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html): the corresponding activation function.
- [MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html): to apply max pooling.
- [Flatten](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html): to flatten the extracted features into a vector.
- [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html): fully connected layer used for final classification.
