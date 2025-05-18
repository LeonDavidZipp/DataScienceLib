import unittest
import numpy as np
from .scores import Scores
import scipy.stats


class TestScores(unittest.TestCase):
	def test_variance(self):
		X = np.array([1, 2, 3, 4, 5])
		expected_variance = 2.0  # Variance formula: sum((X - mean)^2) / n
		self.assertAlmostEqual(Scores.variance(X), expected_variance, places=5)

	def test_std(self):
		X = np.array([1, 2, 3, 4, 5])
		expected_std = np.sqrt(2.0)  # Standard deviation is sqrt(variance)
		self.assertAlmostEqual(Scores.std(X), expected_std, places=5)

	def test_zscore(self):
		X = np.array([1, 2, 3, 4, 5])
		expected_zscores = scipy.stats.zscore(X, axis=0)  # Z-score formula
		np.testing.assert_array_almost_equal(
			Scores.zscore(X), expected_zscores, decimal=5
		)

	def test_variance_single_value(self):
		X = np.array([5])
		expected_variance = 0.0  # Variance of a single value is 0
		self.assertAlmostEqual(Scores.variance(X), expected_variance, places=5)

	def test_std_single_value(self):
		X = np.array([5])
		expected_std = 0.0  # Standard deviation of a single value is 0
		self.assertAlmostEqual(Scores.std(X), expected_std, places=5)

	def test_zscore_single_value(self):
		X = np.array([5])
		expected_zscores = scipy.stats.zscore(
			X, axis=0
		)  # Z-score of a single value is undefined (division by 0)
		np.testing.assert_array_equal(Scores.zscore(X), expected_zscores)

	def test_zscore_empty_array(self):
		X = np.array([])
		expected_zscores = scipy.stats.zscore(
			X, axis=0
		)  # Z-score of an empty array should return an empty array
		np.testing.assert_array_equal(Scores.zscore(X), expected_zscores)


if __name__ == "__main__":
	unittest.main()
