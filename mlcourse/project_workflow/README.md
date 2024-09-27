# End-to-end project workflow

This [example](test_project_workflow.py) demonstrates how to apply the Machine Learning project workflow to a regression task: predicting housing prices.

The dataset is based on data from the 1990 California census. The raw CSV file is available [here](https://raw.githubusercontent.com/bpesquet/mlcourse/main/datasets/california_housing.csv). It is a slightly modified version of the [original dataset](https://www.dcc.fc.up.pt/%7Eltorgo/Regression/cal_housing.html).

9 features are numerical, one (`ocean_proximity`) is categorical. One feature (`total_bedrooms`) has missing values. `median_house_value` is the target feature (value to predict).

The scikit-learn [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function is used to create training and test sets.

Data preprocessing is done through a series of sequential operations on data:

- handling missing values;
- scaling data;
- encoding categorical features.

A scikit-learn [pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline) streamlines these operations. This is useful to prevent mistakes and oversights when preprocessing new data.

The final model and preprocessing pipeline are saved using Python's built-in persistence model, [pickle](https://docs.python.org/3/library/pickle.html), through the [joblib](https://pypi.org/project/joblib/) library for efficiency reasons.
