"""
NumPy tutorial code
"""

import numpy as np
import math


def test_array_fundamentals():
    """
    Test array fundamentals
    """

    a = np.array([1, 2, 3, 4, 5, 6])
    print(a)
    assert a[0] == 1

    a[0] = 10
    # Equality test between two NumPy arrays
    np.testing.assert_allclose(a, [10, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(a[:3], [10, 2, 3])

    b = a[3:]
    np.testing.assert_allclose(b, [4, 5, 6])
    b[0] = 40
    np.testing.assert_allclose(a, [10, 2, 3, 40, 5, 6])

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

    a = np.zeros(shape=2)
    np.testing.assert_allclose(a, [0, 0])

    a = np.ones(shape=(3, 2))
    np.testing.assert_allclose(a, [[1, 1], [1, 1], [1, 1]])

    a = np.arange(4)
    np.testing.assert_allclose(a, [0, 1, 2, 3])


def test_shape_management():
    """
    Test shape management functions
    """

    a = np.arange(6)
    assert a.shape == (6,)

    b = a.reshape(3, 2)
    assert b.shape == (3, 2)

    # Create a row vector from a 1D array
    c = a[np.newaxis, :]
    assert c.shape == (1, 6)
    np.testing.assert_allclose(c, [[0, 1, 2, 3, 4, 5]])

    # Create a column vector from a 1D array
    d = a[:, np.newaxis]
    assert d.shape == (6, 1)
    np.testing.assert_allclose(d, [[0], [1], [2], [3], [4], [5]])


def test_indexing_slicing():
    """
    Test array indexing and slicing
    """

    data = np.array([1, 2, 3])
    assert data[1] == 2
    np.testing.assert_allclose(data[0:2], [1, 2])
    np.testing.assert_allclose(data[1:], [2, 3])
    np.testing.assert_allclose(data[-2:], [2, 3])

    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    np.testing.assert_allclose(a[a < 5], [1, 2, 3, 4])


# Standalone execution
if __name__ == "__main__":
    test_array_fundamentals()
    test_array_attributes()
    test_array_creation()
    test_shape_management()
    test_indexing_slicing()
