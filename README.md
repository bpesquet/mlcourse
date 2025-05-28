# Machine Learning Course

![Dynamic TOML Badge: Python](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fbpesquet%2Fmlcourse%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&query=%24.tool.poetry.dependencies.python&logo=python&logoColor=white&logoSize=auto&label=Python&labelColor=%233776AB&color=black)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/bpesquet/mlcourse/ci.yaml)

This repository contains the public material for my Machine Learning course: [lecture notes](lectures/), [tutorials](tutorials/) and [lab works](labs/).

I have tried to write them in such a way that they should be accessible to anyone wanting to learn the subject, regardless of whether you are one of my students or not.

## Table of Contents

- 📚 [Artificial Intelligence: past, present, future(s) ⤴](https://github.com/bpesquet/bpesquet.github.io/blob/master/content/presentations/chembiona-2925/index.md)
- 📚 [Machine Learning: an introduction](lectures/ml_introduction/)

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

```bash
git clone https://github.com/bpesquet/mlcourse.git
cd mlcourse
poetry install
python {path to Python code file}
```

## Development notes

### Toolchain

This project is built with the following software:

- [Poetry](https://python-poetry.org/) for dependency management;
- [Black](https://github.com/psf/black) for code formatting;
- [Pylint](https://github.com/pylint-dev/pylint) to detect mistakes in the code;
- [pytest](https://docs.pytest.org) for testing the code;
- [Marp](https://marp.app/) for showcasing notes as slideshows during lectures.

### Useful commands

```bash
# Reformat all Python files
black .

# Check the code for mistakes
pylint lectures/* tutorials/*

# Run all code examples as unit tests
# The optional -s flag prints code output
pytest [-s] .
```

## License

[Creative Commons](LICENSE) for textual content and [MIT](CODE_LICENSE) for code.

Copyright © 2024-present [Baptiste Pesquet](https://bpesquet.fr).
