# Machine Learning Course

This repository contains the public material for my Machine Learning course: [lectures](lectures/), [tutorials](tutorials/) and [lab works](labs/).

I have tried to write them in such a way that they should be accessible to anyone wanting to learn the subject, regardless of whether you are one of my students or not.

## Table of Contents

- 📚 [Artificial Intelligence: past, present, future(s) ⤴](https://github.com/bpesquet/bpesquet.github.io/blob/master/content/presentations/chembiona-2925/index.md)

### Supervised learning fundamentals

Supervised Learning is a subset of Machine Learning in which expected results are fed into the system alongside training data.

- 📚 [Principles of supervised learning](lectures/supervised_learning_principles/)
- 📚 [End-to-end project workflow](lectures/project_workflow/)
- 📚 [Assessing classification performance](lectures/classification_performance/)
- 📚 [Learning via Gradient Descent](lectures/gradient_descent/)
- 🛠️ [NumPy](tutorials/numpy/), [pandas](tutorials/pandas/), [scikit-learn](tutorials/scikit-learn/)
- 👩🏽‍💻 [Predict heart disease](labs/predict_heart_disease/)

### Classic algorithms

- 📚 🚧 [Linear Regression](lectures/linear_regression/)
- 📚 🚧 [Decision Trees & Random Forests](lectures/decision_trees_random_forests/)
- ... (more to come)

### Neural networks and Deep Learning

Deep Learning is a subset of Machine Learning based on the usage of large neural networks trained on vast amounts of data.

#### Feedforward neural networks

- 📚 [Feedforward Neural Networks](lectures/feedforward_neural_networks/)
- 🛠️ [PyTorch Fundamentals ⤴](https://github.com/bpesquet/pytorch-tutorial/tree/main/pytorch_tutorial/fundamentals), [Linear Regression with PyTorch ⤴](https://github.com/bpesquet/pytorch-tutorial/tree/main/pytorch_tutorial/linear_regression), [Logistic Regression with PyTorch ⤴](https://github.com/bpesquet/pytorch-tutorial/tree/main/pytorch_tutorial/logistic_regression), [MultiLayer Perceptron with PyTorch ⤴](https://github.com/bpesquet/pytorch-tutorial/tree/main/pytorch_tutorial/multilayer_perceptron)
- 👨‍💻 [Recognize handwritten digits](labs/recognize_handwritten_digits/)

#### Convolutional neural networks

- 📚 [Convolutional Neural Networks](lectures/convolutional_neural_networks/)
- 🛠️ [Convolutional Neural Network with PyTorch ⤴](https://github.com/bpesquet/pytorch-tutorial/tree/main/pytorch_tutorial/convolutional_neural_network)
- 👩🏼‍💻 [Classify common images](labs/classify_common_images/)

#### Recurrent neural networks

- 📚 [Recurrent Neural Networks](lectures/recurrent_neural_networks/)

#### Large Language Models

- 📚 [Large Language Models](lectures/large_language_models/)

#### ... (More to come)

### Reinforcement Learning

Reinforcement Learning is a subset of Machine Learning concerned with the maximization of rewards in a dynamic environment.

- 🚧 [Introduction to Reinforcement Learning](lectures/rl_introduction/)
- ... (more to come)

## Usage

> [uv](https://docs.astral.sh/uv/) needs to be available on your system.

```bash
git clone https://github.com/bpesquet/mlcourse.git
cd mlcourse
uv sync
uv run python {path to example file}
```

## Development notes

### Toolchain

This project is built with the following software:

- [uv](https://docs.astral.sh/uv/) for project management;
- [ruff](https://docs.astral.sh/ruff/) for code formatting and linting;
- [pytest](https://docs.pytest.org) for testing.

### Useful commands

```bash
# Format all Python files
uvx ruff format

# Lint all Python files and fix any fixable errors
uvx ruff check --fix

# Run all code examples as unit tests.
# The optional -s flag prints code output.
uv run pytest [-s]
```

## License

[Creative Commons](LICENSE) for textual content and [MIT](CODE_LICENSE) for code.

Copyright © 2024-present [Baptiste Pesquet](https://bpesquet.fr).
