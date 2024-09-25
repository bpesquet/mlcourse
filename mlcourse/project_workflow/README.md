# End-to-end project workflow

This [example](test_project_workflow.py) demonstrates how to apply the Machine Learning project workflow to a regression task: predicting housing prices.

[![Kaggle houses banner](images/kaggle_housesbanner.png)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)

The dataset is based on data from the 1990 California census. The raw CSV file is available [here](https://raw.githubusercontent.com/bpesquet/mlcourse/main/datasets/california_housing.csv). It is a slightly modified version of the [original dataset](https://www.dcc.fc.up.pt/%7Eltorgo/Regression/cal_housing.html).

Data preprocessing is done through a series of sequential operations on data:

- handling missing values;
- scaling data:
- encoding categorical features.

A scikit-learn [pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline) streamlines these operations. This is useful to prevent mistakes and oversights when preprocessing new data.
