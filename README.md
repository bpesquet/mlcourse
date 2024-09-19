# Machine Learning Course

This repository contains the material for my Machine Learning course: [lecture notes](notes/), [code examples](mlcourse/) and [lab works](labs/). I have tried to write them in such a way that they should be accessible to anyone wanting to learn the subject, regardless of whether you are one of my students or not.

## Table of Contents

- Machine Learning: an introduction [ [notes](notes/ml_introduction/README.md) ]

### Supervised Learning

> Update in progress!

Supervised Learning is a subset of Machine Learning in which expected results are fed into the system alongside training data.

- Principles of supervised learning [ [notes](notes/supervised_learning_principles/README.md) ]
- 👨🏻‍💻 Essential tools [ [lab](labs/essential_tools.md) ]
- Predict California housing prices [ [notes](notes/predict_california_housing_prices/README.md) | [code](mlcourse/test_california_housing_prices.py) ]
- 🚧 Recognize handwritten digits [ [notes]() | [code]() ]
- 🚧 👩🏻‍💻 Predict heart disease [ [lab]() | [code]() ]
- 🚧 Linear Regression [ [notes](notes/linear_regression/README.md) | [code](mlcourse/test_linear_regression.py) ]
- 🚧 Decision Trees & Random Forests [ [notes](notes/decision_trees_random_forests/README.md) | [code]() ]
- ... (more to come)

### Deep Learning

> Soon!

Deep Learning is a subset of Machine Learning based on the usage of large neural networks trained on vast amounts of data.

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
