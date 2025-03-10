# Pandas Tutorial

This tutorial teaches you the very basics of [pandas](https://pandas.pydata.org). Its complete source code is available [here](test_pandas.py).

## What pandas is

The **pandas** library is dedicated to data analysis and manipulation in Python. It greatly facilitates loading, exploring and processing tabular data files.

## How to import pandas

After [installing pandas](https://pandas.pydata.org/docs/getting_started/index.html), it may be imported into Python code like this.

```python
import numpy as np
import pandas as pd
```

## Basic data structures in pandas

The primary data structures in pandas are implemented as two classes:

- [Series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html#pandas.Series), which represents a single data column: one-dimensional labeled array holding values of any type.
- [DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html#pandas.DataFrame), which is quite similar to as a relational data table, with rows and named columns. A `DataFrame` contains one or more `Series`. The `DataFrame` is a commonly used abstraction for data manipulation.

The following code creates two `Series` and adds them into a `DataFraùe`.

```python
# Create two data Series
pop = pd.Series({"CAL": 38332521, "TEX": 26448193, "NY": 19651127})
area = pd.Series({"CAL": 423967, "TEX": 695662, "NY": 141297})

# Create a DataFrame containing the two Series.
# The df_ prefix is used to distinguish pandas DataFrames from plain NumPy arrays
df_poprep = pd.DataFrame({"Population": pop, "Area": area})
```

A `DataFrame` holds data like a two-dimension array or a table with rows and columns. We can print its content and check its shape.

> [!NOTE]
> The [assert](https://docs.python.org/3/reference/simple_stmts.html#grammar-token-python-grammar-assert_stmt) statements are used to check (and also illustrate) the expected results of previous statements.

```python
print(df_poprep)
print(f"df_poprep: {df_poprep.shape}")
assert df_poprep.shape == (3, 2)
```

## Dataset exploration with pandas

The pandas library is often used to interact with tabular data sources. Here is an overview of some essential operations.

### Loading a tabular dataset

Tabular data often comes under the form of a comma-separated values (csv) file. The [read_csv()](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) function facilitates its loading from a local or remote location. It returns a `DataFrame`.

The following code uses pandas to load a slightly modified version of the [Diabetes](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html) dataset. Its contains 10 baseline variables (age, sex, body mass index, average blood pressure, and six blood serum measurements) obtained for  442 diabetes patients, as well as the response of interest, a quantitative measure of disease progression one year after baseline.

```python
# Use a slightly modified version of the Diabetes dataset.
# See https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
dataset_url = "https://raw.githubusercontent.com/bpesquet/mlcourse/refs/heads/main/datasets/diabetes.csv"

# Fetch dataset from url and load it into a DataFrame
df_diabetes = pd.read_csv(dataset_url)
print(f"df_diabetes: {df_diabetes.shape}")
assert df_diabetes.shape == (442, 11)
```

### Dataset summary

The [info()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html) function prints a concise summary od a `DataFraùe`. It is useful to observe the name and data type of each column.

```python
# Print a consise summary of the DataFrame
df_diabetes.info()
```

### Showing data samples

The [head()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html), [tail()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.tail.html) and [sample()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html) functions show resp. the first, last and some randomly selected rows of a `DataFrame`.

```python
# Print the first 5 rows
print(df_diabetes.head(n=5))

# Print 10 random rows
print(df_diabetes.sample(n=10))
```

### Dataset statistics

A simple way to obtain general statitics (mean, standard deviation...) for each column of a `DataFrame` is by using the [describe()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html) function.

```python
# Print descriptive statistics about the DataFrame
print(df_diabetes.describe())
```

### Bridge with NumPy

You can access a representation of the underlying data by  calling the [to_numpy()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html#pandas.DataFrame.to_numpy) function, which converts a `DataFrame` to a NumPy array.

```python
# Convert the DataFrame to a plain NumPy array
diabetes = df_diabetes.to_numpy()
print(f"diabetes (NumPy): {diabetes.shape}, dtype: {diabetes.dtype}")
```

> [!NOTE]
> NumPy arrays have one dtype for the entire array while pandas DataFrames have one dtype per column. When working with heterogeneous data, the dtype of the resulting NumPy array will be chosen to accommodate all of the data involved. For example, if strings are involved, the result will be of `object` dtype. If there are only floats and integers, the resulting array will be of `float` dtype..

### Indexing and selecting data

Standard Python/NumPy expressions work as intended with DataFrames.

```python
# Print the last 5 rows
print(df_diabetes[-5:])
```

It is also possible to select rows when a specific condition is met (*boolean indexing*).

```python
# Print rows with a blood pressure over 100
print(df_diabetes[df_diabetes["BP"] > 100])
```

> [!TIP]
> Learn more about indexing and selection data [here](https://pandas.pydata.org/docs/user_guide/indexing.html#indexing).

### Aggregation operations

The pandas library offers several functions implementing aggregation operations. They can be combined with boolean indexing to create complex queries.

```python
# Print the number of rows for each value of the "SEX" column
print(df_diabetes["SEX"].value_counts())

# Average age of all patients
avg_age = df_diabetes["AGE"].mean()
print(f"Average age: {avg_age:.0f} years")

# Average age of male patients
avg_age_male = df_diabetes[df_diabetes["SEX"] == "M"]["AGE"].mean()
print(f"Average age of male patients: {avg_age_male:.0f} years")
```

### Data splitting

The [drop()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html) function removes rows or columns from a `DataFrame`. This is useful for separating inputs features from labels.

```python
# Select all columns but "Y"
df_inputs = df_diabetes.drop("Y", axis="columns")
print(f"df_inputs: {df_inputs.shape}")

# Select "Y" column only
df_y = df_diabetes["Y"]
print(f"df_y: {df_y.shape}")
```

## Additional resources

- [10 minutes (or maybe a bit more 😉) to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Essential basic functionality](https://pandas.pydata.org/docs/user_guide/basics.html)
