# Introduction to PyTorch

## Linear Regression

This [example](test_linear_regression.py) uses the [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) class to implement linear regression on a simple 2D dataset.

After gradients computation, parameters are updated manually to better illustrate how [gradient descent](../../notes/gradient_descent/README.md) works.

## Logistic Regression

This [example](test_logistic_regression.py) uses the [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) class to implement logistic regression on a simple dataset generated via [scikit-learn](https://scikit-learn.org).

It uses the [cross-entropy](../../notes/classification_performance/README.md#choosing-a-loss-function-1) a.k.a. negative log-likelihood loss, implemented by the [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) class.

> PyTorch also offers the [NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) class implementing the negative log-likelihood loss. A key difference is that `CrossEntropyLoss` expects *logits*  (raw, unnormalized predictions) as inputs, and uses [LogSoftmax](https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax) to transform them into probabilities before computing its output. Using `CrossEntropyLoss` is equivalent to applying `LogSoftmax` followed by `NLLLoss` ([more details](https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81)).

An [optimizer](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) is used to update the model's parameters during gradient descent.
