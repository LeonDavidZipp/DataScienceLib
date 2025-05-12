import unittest
import numpy as np
from src.data_science_lib.transformers.transposing.TransposerNDim import TransposerNDim


class TestTransposerNDim(unittest.TestCase):
	def test_transpose_2d_array(self):
		X = np.array([[1, 2, 3], [4, 5, 6]])
		transposer = TransposerNDim(transpose_dims=(1, 0))
		transposer.fit(X)
		transformed = transposer.transform(X)
		expected = np.array([[1, 4], [2, 5], [3, 6]])
		np.testing.assert_array_equal(transformed, expected)

	def test_transpose_3d_array(self):
		X = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
		transposer = TransposerNDim(transpose_dims=(2, 0, 1))
		transposer.fit(X)
		transformed = transposer.transform(X)
		expected = np.array([[[1, 3], [5, 7]], [[2, 4], [6, 8]]])
		np.testing.assert_array_equal(transformed, expected)

	def test_invalid_transpose_dims(self):
		X = np.array([[1, 2, 3], [4, 5, 6]])
		transposer = TransposerNDim(transpose_dims=(0, 2))  # Invalid dimensions
		with self.assertRaises(ValueError):
			transposer.transform(X)

	def test_no_transpose(self):
		X = np.array([[1, 2, 3], [4, 5, 6]])
		transposer = TransposerNDim(transpose_dims=(0, 1))  # No change in dimensions
		transposer.fit(X)
		transformed = transposer.transform(X)
		np.testing.assert_array_equal(transformed, X)


if __name__ == "__main__":
	unittest.main()
