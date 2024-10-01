# Machine Learning Course

This repository contains the material for my Machine Learning course: [lecture notes](notes/), [code examples](mlcourse/) and [lab works](labs/). I have tried to write them in such a way that they should be accessible to anyone wanting to learn the subject, regardless of whether you are one of my students or not.

## Table of Contents

- About Artificial Intelligence [ [notes](notes/about_ai//README.md) ]
- Machine Learning: an introduction [ [notes](notes/ml_introduction/README.md) ]

### Supervised Learning

Supervised Learning is a subset of Machine Learning in which expected results are fed into the system alongside training data.

#### Fundamentals

- Principles of supervised learning [ [notes](notes/supervised_learning_principles/README.md) ]
- End-to-end project workflow [ [notes](notes/project_workflow/README.md) | [example](mlcourse/project_workflow/) ]
- Assessing classification performance [ [notes](notes/classification_performance/README.md) | [example](/mlcourse/classification_performance/) ]
- Learning via Gradient Descent [ [notes](notes/gradient_descent/README.md) ]
- 👨🏻‍💻 Essential tools [ [lab](labs/essential_tools/README.md) ]
- 👩🏻‍💻 Predict heart disease [ [lab](labs/predict_heart_disease/README.md) | [solution](/mlcourse/predict_heart_disease/) ]

#### Algorithms

> Update in progress!

- 🚧 Linear Regression [ [notes](notes/linear_regression/README.md) ]
- 🚧 Decision Trees & Random Forests [ [notes](notes/decision_trees_random_forests/README.md) ]
- ... (more to come)

### Neural Networks and Deep Learning

> Update in progress!

Deep Learning is a subset of Machine Learning based on the usage of large neural networks trained on vast amounts of data.

- Feedforward Neural Networks [ [notes](notes/feedforward_neural_networks/README.md) | [example](/mlcourse/feedforward_neural_networks/) ]
- 👩🏻‍💻 Introduction to PyTorch [ [lab](labs/pytorch_intro/README.md) | [examples](/mlcourse/pytorch_intro/) ]
- 👨🏻‍💻 Classify 2D data with a neural network [ [lab](labs/classify_2d_data/README.md) | [solution](/mlcourse/classify_2d_data/) ]
- Convolutional Neural Networks [ [notes](notes/convolutional_neural_networks/README.md) | [example](/mlcourse/convolutional_neural_networks/) ]
- ... (more to come)

### Reinforcement Learning

> Update in progress!

Reinforcement Learning is a subset of Machine Learning concerned with the maximization of rewards in a dynamic environment.

- 🚧 Introduction to Reinforcement Learning [ [notes](notes/rl_introduction/README.md) ]
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
black mlcourse

# Check the code for mistakes
pylint mlcourse

# Run all code examples as unit tests
# The -s flag prints code output
pytest [-s] mlcourse
```

## License

[Creative Commons](LICENSE) for textual content and [MIT](CODE_LICENSE) for code.

Copyright © 2024-present [Baptiste Pesquet](https://bpesquet.fr).
