"""
NumPy tutorial code
"""

import numpy as np
import math


def test_array_fundamentals():
    """
    Test array fundamentals
    """

    a = np.array([1, 2, 3])
    print(a)
    assert a[0] == 1

    a[0] = 10
    # Equality test between two NumPy arrays
    np.testing.assert_allclose(a, [10, 2, 3])

    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert a[1, 3] == 8


def test_array_attributes():
    """
    Test array attributes
    """

    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    assert a.ndim == 2

    assert a.shape == (3, 4)
    assert len(a.shape) == a.ndim

    assert a.size == 12
    assert a.size == math.prod(a.shape)

    # The array contains only integers.
    # "int" for integer, "64" for 64-bit
    assert a.dtype == np.int64

    a = np.array([0.5, 1])
    # The array contains a floating point value
    assert a.dtype == np.float64


def test_array_creation():
    """
    Test various methods for creating arrays
    """

    # Create a 1D array filled with 0s
    a = np.zeros(shape=2)
    np.testing.assert_allclose(a, [0, 0])

    # Create a 2D array filled with 1s
    a = np.ones(shape=(3, 2))
    np.testing.assert_allclose(a, [[1, 1], [1, 1], [1, 1]])

    # Create a 1D array with integer values evenly spaced within a given interval
    a = np.arange(stop=4)
    np.testing.assert_allclose(a, [0, 1, 2, 3])

    # Init a NumPy random number generator
    rng = np.random.default_rng()

    # Create a 2D array with float values sampled from a uniform distribution
    a = rng.uniform(size=(3, 4))
    print(a)
    assert a.shape == (3, 4)

    # Create a 3D array with integer values sampled from a uniform distribution
    a = rng.integers(low=0, high=100, size=(3, 2, 4))
    print(a)
    assert a.shape == (3, 2, 4)


def test_shape_management():
    """
    Test shape management functions
    """

    # Create a 1D array
    data = np.arange(start=1, stop=7)
    assert data.shape == (6,)
    np.testing.assert_allclose(data, [1, 2, 3, 4, 5, 6])

    # Reshape 1D array into a 2D array
    f = data.reshape(2, 3)
    assert f.shape == (2, 3)
    np.testing.assert_allclose(f, [[1, 2, 3], [4, 5, 6]])

    # Reshape 1D array into a 2D array
    g = data.reshape(3, 2)
    assert g.shape == (3, 2)
    np.testing.assert_allclose(g, [[1, 2], [3, 4], [5, 6]])

    # Add a dimension to create a row vector from a 1D array
    d = data[np.newaxis, :]
    assert d.shape == (1, 6)
    np.testing.assert_allclose(d, [[1, 2, 3, 4, 5, 6]])

    # Add a dimension to create a column vector from a 1D array
    e = data[:, np.newaxis]
    assert e.shape == (6, 1)
    np.testing.assert_allclose(e, [[1], [2], [3], [4], [5], [6]])

    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    # Flatten the 2D array into a 1D array
    f = a.flatten()
    assert f.shape == (a.size,)
    np.testing.assert_allclose(f, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # flatten() creates a copy of the original array
    f[0] = 10
    np.testing.assert_allclose(a, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    # Flatten the 2D array into a 1D array
    g = a.ravel()
    assert g.shape == (a.size,)
    np.testing.assert_allclose(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # ravel() creates a view of the original array sharing underlying memory
    g[0] = 10
    np.testing.assert_allclose(a, [[10, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])


def test_indexing_slicing():
    """
    Test array indexing and slicing
    """

    data = np.array([1, 2, 3])

    assert data[0] == 1
    assert data[1] == 2

    # Select elements between indexes 0 (included) and 2 (excluded)
    np.testing.assert_allclose(data[0:2], [1, 2])

    # Select elements starting at index 1 (included)
    np.testing.assert_allclose(data[1:], [2, 3])

    # Select last element
    assert data[-1] == 3

    # Select all elements but last one
    np.testing.assert_allclose(data[:-1], [1, 2])

    # Select last 2 elements
    np.testing.assert_allclose(data[-2:], [2, 3])

    # Select second-to-last element
    np.testing.assert_allclose(data[-2:-1], [2])

    a = np.array([1, 2, 3, 4, 5, 6])
    b = a[3:]
    np.testing.assert_allclose(b, [4, 5, 6])
    # Both arrays share the same underlying memory
    b[0] = 40
    np.testing.assert_allclose(a, [1, 2, 3, 40, 5, 6])

    data = np.array([[1, 2], [3, 4], [5, 6]])
    np.testing.assert_allclose(data[0, 1], 2)
    np.testing.assert_allclose(data[1:3], [[3, 4], [5, 6]])
    np.testing.assert_allclose(data[0:2, 0], [1, 3])


def test_aggregation_operations():
    """
    Test array axes
    """

    data = np.array([[1, 2], [5, 3], [4, 6]])

    assert np.max(data) == 6
    assert np.min(data) == 1
    assert np.sum(data) == 21

    # Finding maximum values along first axis (rows)
    np.testing.assert_allclose(np.max(data, axis=0), [5, 6])

    # Finding maximum values along second axis (columns)
    np.testing.assert_allclose(np.max(data, axis=1), [2, 5, 6])


def test_operations_between_arrays():
    """
    Test some basic array operations
    """

    data = np.array([1, 2])
    ones = np.ones(2)
    # Element-wise operations
    np.testing.assert_allclose(data + ones, [2, 3])
    np.testing.assert_allclose(data - ones, [0, 1])
    np.testing.assert_allclose(data * data, [1, 4])
    np.testing.assert_allclose(data / data, [1, 1])

    x = np.array([[1, 2, 3], [3, 2, -2]])
    y = np.array([[3, 0, 2], [1, 4, -2]])
    # Element-wise product between two matrices (shapes must be identical)
    z = x * y
    assert z.shape == (2, 3)
    np.testing.assert_allclose(z, [[3, 0, 6], [3, 8, 4]])

    x = np.array([[1, 2, 3], [3, 2, 1]])
    y = np.array([[3, 0], [2, 1], [4, -2]])
    # Dot product between two matrices (shapes must be compatible)
    z = np.dot(x, y)
    assert z.shape == (2, 2)
    np.testing.assert_allclose(z, [[19, -4], [17, 0]])


def test_broadcasting():
    """
    Test broadcasting
    """

    a = np.array([1.0, 2.0])
    # Broadcasting between a 1D array and a scalar
    np.testing.assert_allclose(a * 1.6, [1.6, 3.2])

    data = np.array([[1, 2], [3, 4], [5, 6]])
    ones_row = np.array([[1, 1]])
    # Broadcasting between a 2D array and a 1D array
    np.testing.assert_allclose(data + ones_row, [[2, 3], [4, 5], [6, 7]])


def test_formulae():
    """
    Test implementation of math formulae
    """

    predictions = np.array([1, 1, 1])
    labels = np.array([1, 2, 3])

    error = np.square(predictions - labels).mean()
    assert error == 5 / 3


# Standalone execution
if __name__ == "__main__":
    test_array_fundamentals()
    test_array_attributes()
    test_array_creation()
    test_shape_management()
    test_indexing_slicing()
    test_aggregation_operations()
    test_operations_between_arrays()
    test_broadcasting()
    test_formulae()
