"""
pandas tutorial code
"""

import pandas as pd


def test_data_structures():
    """
    Test basic data structures
    """

    # Create two data Series
    pop = pd.Series({"CAL": 38332521, "TEX": 26448193, "NY": 19651127})
    area = pd.Series({"CAL": 423967, "TEX": 695662, "NY": 141297})

    # Create a DataFrame containing the two Series.
    # The df_ prefix is used to distinguish pandas DataFrames from plain NumPy arrays
    df_poprep = pd.DataFrame({"Population": pop, "Area": area})

    print(df_poprep)
    print(f"df_poprep: {df_poprep.shape}")
    assert df_poprep.shape == (3, 2)


def test_dataset_exploration():
    """
    Test functions for loading and exploring a dataset
    """

    # Use a slightly modified version of the Diabetes dataset.
    # See https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
    dataset_url = "https://raw.githubusercontent.com/bpesquet/mlcourse/refs/heads/main/datasets/diabetes.csv"

    # Fetch dataset from url and load it into a DataFrame
    df_diabetes = pd.read_csv(dataset_url)
    print(f"df_diabetes: {df_diabetes.shape}")
    assert df_diabetes.shape == (442, 11)

    # Print a consise summary of the DataFrame
    df_diabetes.info()

    # Print the first 5 rows
    print(df_diabetes.head(n=5))

    # Print 10 random rows
    print(df_diabetes.sample(n=10))

    # Print descriptive statistics about the DataFrame
    print(df_diabetes.describe())

    # Convert the DataFrame to a plain NumPy array
    diabetes = df_diabetes.to_numpy()
    print(f"diabetes (NumPy): {diabetes.shape}, dtype: {diabetes.dtype}")

    # Print the last 5 rows
    print(df_diabetes[-5:])

    # Print rows with a blood pressure over 100
    print(df_diabetes[df_diabetes["BP"] > 100])

    # Print the number of rows for each value of the "SEX" column
    print(df_diabetes["SEX"].value_counts())

    # Average age of all patients
    avg_age = df_diabetes["AGE"].mean()
    print(f"Average age: {avg_age:.0f} years")

    # Average age of male patients
    avg_age_male = df_diabetes[df_diabetes["SEX"] == "M"]["AGE"].mean()
    print(f"Average age of male patients: {avg_age_male:.0f} years")

    # Select all columns but "Y"
    df_inputs = df_diabetes.drop("Y", axis="columns")
    print(f"df_inputs: {df_inputs.shape}")

    # Select "Y" column only
    df_y = df_diabetes["Y"]
    print(f"df_y: {df_y.shape}")


# Standalone execution
if __name__ == "__main__":
    test_data_structures()
    test_dataset_exploration()
