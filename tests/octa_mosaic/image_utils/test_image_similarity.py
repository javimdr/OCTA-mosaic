import numpy as np
import pytest

from octa_mosaic.image_utils import image_similarity

np.random.seed(16)


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (np.array([1, 2, 3]), np.array([1, 2, 3]), 1.0),
        (np.array([1, 2, 3]), np.array([3, 2, 1]), -1.0),
        (np.array([1, 2, 3]), np.array([0, 0, 0]), 0.0),
        (np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]]), 1.0),
        (np.array([[1, 2], [3, 4]]), np.array([[4, 3], [2, 1]]), -1.0),
        (np.array([[1, 2], [3, 4]]), np.array([[1, 1], [1, 1]]), 0.0),
    ],
)
def test_zncc(x, y, expected):
    """
    Test the Zero-Normalized Cross Correlation (ZNCC) function.

    Args:
        x: The first array for testing.
        y: The second array for testing.
        expected: The expected ZNCC result.

    Asserts:
        Checks that the computed ZNCC matches the expected value.
    """
    result = image_similarity.zncc(x, y)
    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "x, y",
    [
        (np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])),
        (np.array([[1, 2], [3, 4]]), np.array([[4, 3], [2, 1]])),
        (np.array([[1, 2], [3, 4]]), np.array([[1, 1], [1, 1]])),
        (np.random.randint(0, 255, (300, 300)), np.random.randint(0, 255, (300, 300))),
        (np.random.randint(0, 255, (300, 300)), np.random.randint(0, 255, (300, 300))),
        (np.random.randint(0, 255, (300, 300)), np.random.randint(0, 255, (300, 300))),
    ],
)
def test_same_result_zncc_and_normxcorr2_valid(x, y):
    zncc_value = image_similarity.zncc(x, y)
    normxcorr2_value = image_similarity.normxcorr2(x, y, mode="valid")
    assert np.isclose(zncc_value, normxcorr2_value)
