# Machine Learning Course

This repository contains the public material for my Machine Learning course: [lecture notes](lectures/) with code examples and [lab works](labs/).

I have tried to write them in such a way that they should be accessible to anyone wanting to learn the subject, regardless of whether you are one of my students or not.

## Table of Contents

- [About Artificial Intelligence](lectures/about_ai/)
- [Machine Learning: an introduction](lectures/ml_introduction/)

### Supervised learning fundamentals

Supervised Learning is a subset of Machine Learning in which expected results are fed into the system alongside training data.

- [Principles of supervised learning](lectures/supervised_learning_principles/)
- [End-to-end project workflow](lectures/project_workflow/)
- [Assessing classification performance](lectures/classification_performance/)
- [Learning via Gradient Descent](lectures/gradient_descent/)
- [Lab: Predict heart disease](labs/predict_heart_disease/)

### Classic algorithms

- 🚧 [Linear Regression](lectures/linear_regression/)
- 🚧 [Decision Trees & Random Forests](lectures/decision_trees_random_forests/)
- ... (more to come)

### Neural networks and Deep Learning

Deep Learning is a subset of Machine Learning based on the usage of large neural networks trained on vast amounts of data.

- [Feedforward Neural Networks](lectures/feedforward_neural_networks/)
- [Lab: Introduction to PyTorch](labs/pytorch_intro/)
- [Lab: Classify 2D data with a neural network](labs/classify_2d_data/)
- Convolutional Neural Networks [ [notes](notes/convolutional_neural_networks/README.md) | [example](/mlcourse/convolutional_neural_networks/) ]
- ... (more to come)

### Reinforcement Learning

Reinforcement Learning is a subset of Machine Learning concerned with the maximization of rewards in a dynamic environment.

- 🚧 [Introduction to Reinforcement Learning](lectures/rl_introduction/)
- ... (more to come)

## Usage

```bash
git clone https://github.com/bpesquet/mlcourse.git
cd mlcourse
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
pylint lectures

# Run all code examples as unit tests
# The -s flag prints code output
pytest [-s] .
```

## License

[Creative Commons](LICENSE) for textual content and [MIT](CODE_LICENSE) for code.

Copyright © 2024-present [Baptiste Pesquet](https://bpesquet.fr).
