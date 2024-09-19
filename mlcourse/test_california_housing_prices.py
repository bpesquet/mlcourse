"""
Predict California housing prices

Inspired by https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib


def load_dataset(url):
    """Load a dataset with pandas"""

    # Load the dataset in a pandas DataFrame
    dataset = pd.read_csv(url)

    # Print dataset shape (rows and columns)
    print(f"Dataset shape: {dataset.shape}")

    # Print a concise summary of the dataset
    # 9 attributes are numerical, one ("ocean_proximity") is categorical
    # "median_house_value" is the target attribute
    # One attribute ("total_bedrooms") has missing values
    dataset.info()

    # Show 10 random samples of the dataset
    print(dataset.sample(n=10))

    # Print descriptive statistics for all numerical attributes
    print(dataset.describe())

    return dataset


def plot_geo_data(df_housing):
    """Plot a geographical representation of the housing dataset"""

    # This dataset has the particularity of including geographical coordinates
    # Visualise prices relative to them
    df_housing.plot(
        kind="scatter",
        x="longitude",
        y="latitude",
        alpha=0.4,
        s=df_housing["population"] / 100,
        label="population",
        c="median_house_value",
        cmap=plt.get_cmap("jet"),
        colorbar=True,
        sharex=False,
    )
    plt.legend()
    plt.show()


def plot_correlation_matrix(df):
    """Plot a correlation matrix for a DataFrame"""

    # Select numerical columns
    df_numerical = df.select_dtypes(include=[np.number])

    plt.subplots()
    sns.heatmap(
        df.corr(numeric_only=True),
        vmax=0.8,
        linewidths=0.01,
        square=True,
        annot=True,
        linecolor="white",
        xticklabels=df_numerical.columns,
        annot_kws={"size": 10},
        yticklabels=df_numerical.columns,
    )
    plt.show()


def plot_dataset(df_housing):
    """Display plots for the housing dataset"""

    # Improve plots appearance
    sns.set_theme()

    df_housing.hist(bins=30)
    plt.show()

    plot_geo_data(df_housing)

    plot_correlation_matrix(df_housing)


def split_dataset(df):
    """Split dataset between inputs/targets and training/test sets"""

    # Separate inputs from targets

    # Target attribute is removed to create inputs
    df_x = df.drop("median_house_value", axis="columns")

    # Targets are stored separately in a new variable
    df_y = df["median_house_value"]

    # Split dataset between training and test sets
    # A unique call to train_test_split is mandatory to maintain inputs/target correspondance between samples
    df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(
        df_x, df_y, test_size=0.2
    )
    print(f"df_x_train: {df_x_train.shape}. df_y_train: {df_y_train.shape}")
    print(f"df_x_test: {df_x_test.shape}. df_y_test: {df_y_test.shape}")

    return df_x_train, df_x_test, df_y_train, df_y_test


def preprocess_dataset(df_x_train, df_y_train):
    """Preprocess a dataset"""

    # Compute percent of missing values among features
    print(df_x_train.isnull().sum() * 100 / df_x_train.isnull().count())

    # Show random samples with missing values
    df_x_train[df_x_train.isnull().any(axis=1)].sample(n=5)

    # Get numerical features
    num_features = df_x_train.select_dtypes(include=[np.number]).columns
    print(num_features)

    # Get categorical features
    cat_features = df_x_train.select_dtypes(include=[object]).columns
    print(cat_features)

    # Print all values for the "ocean_proximity" feature
    df_x_train["ocean_proximity"].value_counts()

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler()),
        ]
    )

    # This pipeline applies the previous one on numerical features
    # It also one-hot encodes the categorical features
    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_features),
            ("cat", OneHotEncoder(), cat_features),
        ]
    )

    # Apply all preprocessing operations to the training set through pipelines
    # Result is a NumPy array, hence no more df_ prefix
    x_train = full_pipeline.fit_transform(df_x_train)

    # Print preprocessed inputs shape and first sample
    # "ocean_proximity" attribute has 5 different values
    # To represent them, one-hot encoding has added 5 features and removed the original one
    print(f"x_train: {x_train.shape}")
    print(x_train[0])

    # Transform the targets DataFrame into a plain tensor
    y_train = df_y_train.to_numpy()

    return x_train, y_train, full_pipeline


def compute_error(model, x, y):
    """Compute error (as root of MSE) for a model and a training set"""

    # Compute model predictions (median house prices) for training set
    y_pred = model.predict(x)

    # Compute the error between actual and expected median house prices
    return np.sqrt(mean_squared_error(y, y_pred))


def train(model, x, y):
    """Train a model on a training set"""

    # Fit model to the data
    model.fit(x, y)

    # Compute the error between actual and expected results
    return compute_error(model, x, y)


def compute_crossval_mean_score(model, x, y):
    """Return the mean of cross validation scores for a model and a training set"""

    cv_scores = -cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=10)
    return np.sqrt(cv_scores).mean()


def train_models(x, y):
    """Train models on a training set and return the best one"""

    lin_model = LinearRegression()
    lin_error = train(lin_model, x, y)
    print(f"Training error for linear regression: {lin_error:.02f}")

    dt_model = DecisionTreeRegressor()
    dt_error = train(dt_model, x, y)
    print(f"Training error for decision tree: {dt_error:.02f}")

    lin_cv_mean = compute_crossval_mean_score(lin_model, x, y)
    print(f"Mean cross-validation error for linear regression: {lin_cv_mean:.02f}")

    dt_cv_mean = compute_crossval_mean_score(dt_model, x, y)
    print(f"Mean cross-validation error for decision tree: {dt_cv_mean:.02f}")

    # Fit a random forest model to the training set
    rf_model = RandomForestRegressor(n_estimators=20)
    rf_error = train(rf_model, x, y)
    print(f"Training error for random forest: {rf_error:.02f}")

    rf_cv_mean = compute_crossval_mean_score(rf_model, x, y)
    print(f"Mean cross-validation error for random forest: {rf_cv_mean:.02f}")

    # Return the best model
    return rf_model


def tune_model(model, x, y):
    """Tune a model on a training set"""

    # Grid search explores a user-defined set of hyperparameter values
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [50, 100, 150], "max_features": [6, 8, 10]},
    ]

    # train across 5 folds, that's a total of 12*5=60 rounds of training
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(x, y)

    # Store the best model found
    final_model = grid_search.best_estimator_

    # Print the best combination of hyperparameters found
    print(grid_search.best_params_)

    return final_model


def evaluate_model_on_test_data(model, df_x_test, df_y_test, pipeline):
    """Evaluate a trained model on the test dataset"""

    # Apply preprocessing operations to test inputs
    # Calling transform() and not fit_transform() uses preprocessing values computed on training set
    x_test = pipeline.transform(df_x_test)

    # Transform the targets DataFrame into a plain tensor
    y_test = df_y_test.to_numpy()

    error = compute_error(model, x_test, y_test)
    print(f"Test error for final model: {error:.02f}")


def save_model(model, pipeline):
    """Save a model and preprocessing pipeline to disk for later reuse"""

    # Relative path for saving and loading models
    model_folder = "./_output"
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    # Serialize final model and input pipeline to disk
    joblib.dump(model, f"{model_folder}/housing_model.pkl")
    joblib.dump(pipeline, f"{model_folder}/housing_pipeline.pkl")

    # (Later in the process)
    # model = joblib.load(f"{model_folder}/housing_model.pkl")
    # pipeline = joblib.load(f"{model_folder}/housing_pipeline.pkl")
    # ...


def test_california_housing_prices(show_plots=False):
    """Main test function"""

    # STEP 1
    # The df_ prefix is used to distinguish dataframes from plain NumPy arrays
    df_housing = load_dataset(
        "https://raw.githubusercontent.com/bpesquet/mlcourse/main/datasets/california_housing.csv"
    )

    # STEP 2
    if show_plots:
        plot_dataset(df_housing)

    df_x_train, df_x_test, df_y_train, df_y_test = split_dataset(df_housing)

    x_train, y_train, preprocessing_pipeline = preprocess_dataset(
        df_x_train, df_y_train
    )

    # STEP 3
    best_model = train_models(x_train, y_train)

    # STEP 4
    final_model = tune_model(best_model, x_train, y_train)

    evaluate_model_on_test_data(
        final_model, df_x_test, df_y_test, preprocessing_pipeline
    )

    # STEP 5
    save_model(final_model, preprocessing_pipeline)


# Standalone execution
if __name__ == "__main__":
    test_california_housing_prices(show_plots=True)
