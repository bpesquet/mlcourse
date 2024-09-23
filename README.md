# Machine Learning Course

This repository contains the material for my Machine Learning course: [lecture notes](notes/), [code examples](mlcourse/) and [lab works](labs/). I have tried to write them in such a way that they should be accessible to anyone wanting to learn the subject, regardless of whether you are one of my students or not.

## Table of Contents

- About Artificial Intelligence [ [notes](notes/about_ai//README.md) ]
- Machine Learning: an introduction [ [notes](notes/ml_introduction/README.md) ]

### Supervised Learning

> Update in progress!

Supervised Learning is a subset of Machine Learning in which expected results are fed into the system alongside training data.

- Principles of supervised learning [ [notes](notes/supervised_learning_principles/README.md) ]
- 👨🏻‍💻 Essential tools [ [lab](labs/essential_tools/README.md) ]
- Predict California housing prices [ [notes](notes/california_housing_prices/README.md) | [code](mlcourse/test_california_housing_prices.py) ]
- Recognize handwritten digits [ [notes](notes/handwritten_digits/README.md) | [code](/mlcourse/test_handwritten_digits.py) ]
- 🚧 👩🏻‍💻 Predict heart disease [ [lab](labs/predict_hear_disease/README.md) ]
- Learning via Gradient Descent [ [notes](notes/gradient_descent/README.md) ]
- 🚧 Linear Regression [ [notes](notes/linear_regression/README.md) ]
- 🚧 Decision Trees & Random Forests [ [notes](notes/decision_trees_random_forests/README.md) ]
- 🚧 Artificial Neural Networks [ [notes](notes/artificial_neural_networks/README.md) ]
- ... (more to come)

### Deep Learning

> Update in progress!

Deep Learning is a subset of Machine Learning based on the usage of large neural networks trained on vast amounts of data.

- 👨🏻‍💻 Introduction to PyTorch [ [lab](labs/pytorch/README.md) | [code](/mlcourse/torch/README.md) ]
- 🚧 Convolutional Neural Networks [ [notes](notes/convolutional_neural_networks/README.md) ]
- ... (more to come)

### Reinforcement Learning

> Soon!

Reinforcement Learning is a subset of Machine Learning concerned with the maximization of rewards in a dynamic environment.

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
black mlcourse

# Check the code for mistakes
pylint mlcourse

# Run all code examples as unit tests
# The -s flag prints code output
pytest [-s] mlcourse
```

## License

[Creative Commons](LICENSE) for the textual content and [MIT](CODE_LICENSE) for the code.

Copyright © 2024-present [Baptiste Pesquet](https://bpesquet.fr).
