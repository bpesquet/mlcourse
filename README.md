# Machine Learning Course

> [TP 1](https://colab.research.google.com/drive/1vfJ3KAEZMfDo0y2FauHpTnavTWRixxD4?usp=sharing)

This repository contains the public material for my Machine Learning course: [lectures](#-lectures), [tutorials](#️-tutorials), [lab works](#-labs) and [resources](#-resources).

I have tried to write them in such a way that they should be accessible to anyone wanting to learn the subject, regardless of whether you are one of my students or not.

## Table of Contentsv

### 📚 Lectures

#### Overview

- [Artificial Intelligence: past, present, future(s) ⤴](https://github.com/bpesquet/bpesquet.github.io/blob/master/content/presentations/chembiona-2925/index.md)
- [Supervised Learning: a tutorial example](lectures/tutorial_example//)
- [End-to-end project workflow](lectures/project_workflow/)
- [Assessing classification performance](lectures/classification_performance/)
- [Feedforward Neural Networks](lectures/feedforward_neural_networks/)
- [Learning via Gradient Descent](lectures/gradient_descent/)

#### Deep Learning

- [Convolutional Neural Networks](lectures/convolutional_neural_networks/)
- [Recurrent Neural Networks](lectures/recurrent_neural_networks/)
- [Large Language Models](lectures/large_language_models/)

#### Reinforcement Learning

- [Introduction to Reinforcement Learning](lectures/rl_introduction/)
- [Multi-armed Bandits](lectures/rl_bandits/)

### 🛠️ Tutorials

- [NumPy](tutorials/numpy/)
- [pandas](tutorials/pandas/)
- [scikit-learn](tutorials/scikit-learn/)
- [PyTorch ⤴](https://github.com/bpesquet/pytorch-tutorial)

### 👩🏽‍💻 Labs

- [Predict heart disease](labs/predict_heart_disease/)
- [Recognize handwritten digits](labs/recognize_handwritten_digits/)
- [Classify common images](labs/classify_common_images/)

### 💡 Resources

- [Python](https://github.com/bpesquet/python)

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
