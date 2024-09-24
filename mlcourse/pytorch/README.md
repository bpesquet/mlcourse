# PyTorch code examples

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

It uses the [cross-entropy](../../notes/handwritten_digits/README.md#choosing-a-loss-function-1) a.k.a. negative log-likelihood loss, implemented by the [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) class.

> This is equivalent to combining the [LogSoftmax](https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax) and [NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) classes ([more details](https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81)).

An [optimizer](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) is used to update the model's parameters during gradient descent.
