import unittest
import numpy as np
import pandas as pd
import polars as pl
import polars.testing as pl_testing
from OutlierRemover import OutlierRemover


class TestOutlierRemoverCorrectInputs(unittest.TestCase):
	def test_remove_outliers_numpy(self):
		# y of shape (3,)
		X = np.array([[1, 2, 3], [5, 6, 100], [3, 4, 5]])
		y = np.array([4, 4, 6])
		X_pred, y_pred = OutlierRemover.remove_outliers(X, y, threshold=1.4)
		X_exp = np.array([[1, 2, 3], [3, 4, 5]])
		y_exp = np.array([4, 6])
		self.assertIsInstance(X_pred, np.ndarray)
		self.assertIsInstance(y_pred, np.ndarray)
		np.testing.assert_array_equal(X_pred, X_exp)
		np.testing.assert_array_equal(y_pred, y_exp)

		# y of shape (3, 1)
		y = np.array([[4], [4], [6]])
		X_pred, y_pred = OutlierRemover.remove_outliers(X, y, threshold=1.4)
		X_exp = np.array([[1, 2, 3], [3, 4, 5]])
		y_exp = np.array([[4], [6]])
		self.assertIsInstance(X_pred, np.ndarray)
		self.assertIsInstance(y_pred, np.ndarray)
		np.testing.assert_array_equal(X_pred, X_exp)
		np.testing.assert_array_equal(y_pred, y_exp)

	def test_remove_outliers_numpy_no_y(self):
		X = np.array([[1, 2, 3, 4], [5, 6, 100, 4], [3, 4, 5, 6]])
		X_pred = OutlierRemover.remove_outliers(X, threshold=1.4)
		X_exp = np.array([[1, 2, 3, 4], [3, 4, 5, 6]])
		self.assertIsInstance(X_pred, np.ndarray)
		np.testing.assert_array_equal(X_pred, X_exp)

	def test_remove_outliers_pandas(self):
		X = pd.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 100, 4], "C": [3, 4, 5, 6]})
		y = pd.DataFrame({"D": [1, 2, 3, 4]})
		X_pred, y_pred = OutlierRemover.remove_outliers(X, y, threshold=1.4)
		X_exp = pd.DataFrame({"A": [1, 2, 4], "B": [5, 6, 4], "C": [3, 4, 6]})
		y_exp = pd.DataFrame({"D": [1, 2, 4]})
		self.assertIsInstance(X_pred, pd.DataFrame)
		self.assertIsInstance(y_pred, pd.DataFrame)
		pd.testing.assert_frame_equal(X_pred, X_exp)
		pd.testing.assert_frame_equal(y_pred, y_exp)

	def test_remove_outliers_pandas_no_y(self):
		X = pd.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 100, 4], "C": [3, 4, 5, 6]})
		X_pred = OutlierRemover.remove_outliers(X, threshold=1.4)
		X_exp = pd.DataFrame({"A": [1, 2, 4], "B": [5, 6, 4], "C": [3, 4, 6]})
		self.assertIsInstance(X_pred, pd.DataFrame)
		pd.testing.assert_frame_equal(X_pred, X_exp)

	def test_remove_outliers_polars(self):
		X = pl.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 100, 4], "C": [3, 4, 5, 6]})
		y = pl.DataFrame({"D": [1, 2, 3, 4]})
		X_pred, y_pred = OutlierRemover.remove_outliers(X, y, threshold=1.4)
		X_exp = pl.DataFrame({"A": [1, 2, 4], "B": [5, 6, 4], "C": [3, 4, 6]})
		y_exp = pl.DataFrame({"D": [1, 2, 4]})
		self.assertIsInstance(X_pred, pl.DataFrame)
		self.assertIsInstance(y_pred, pl.DataFrame)
		pl_testing.assert_frame_equal(X_pred, X_exp)
		pl_testing.assert_frame_equal(y_pred, y_exp)

	def test_remove_outliers_polars_no_y(self):
		X = pl.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 100, 4], "C": [3, 4, 5, 6]})
		X_pred = OutlierRemover.remove_outliers(X, threshold=1.4)
		X_exp = pl.DataFrame({"A": [1, 2, 4], "B": [5, 6, 4], "C": [3, 4, 6]})
		self.assertIsInstance(X_pred, pl.DataFrame)
		pl_testing.assert_frame_equal(X_pred, X_exp)

	def test_remove_outliers_polars_lazy(self):
		X = pl.LazyFrame({"A": [1, 2, 3, 4], "B": [5, 6, 100, 4], "C": [3, 4, 5, 6]})
		y = pl.LazyFrame({"D": [1, 2, 3, 4]})
		X_pred, y_pred = OutlierRemover.remove_outliers(X, y, threshold=1.4)
		X_exp = pl.LazyFrame({"A": [1, 2, 4], "B": [5, 6, 4], "C": [3, 4, 6]})
		y_exp = pl.LazyFrame({"D": [1, 2, 4]})
		self.assertIsInstance(X_pred, pl.LazyFrame)
		self.assertIsInstance(y_pred, pl.LazyFrame)
		pl_testing.assert_frame_equal(X_pred.collect(), X_exp.collect())
		pl_testing.assert_frame_equal(y_pred.collect(), y_exp.collect())

	def test_remove_outliers_polars_lazy_no_y(self):
		X = pl.LazyFrame({"A": [1, 2, 3, 4], "B": [5, 6, 100, 4], "C": [3, 4, 5, 6]})
		X_pred = OutlierRemover.remove_outliers(X, threshold=1.4)
		X_exp = pl.LazyFrame({"A": [1, 2, 4], "B": [5, 6, 4], "C": [3, 4, 6]})
		self.assertIsInstance(X_pred, pl.LazyFrame)
		pl_testing.assert_frame_equal(X_pred.collect(), X_exp.collect())


class TestOutlierRemoverIncorrectInputs(unittest.TestCase):
	def test_remove_outliers_incorrect_shape(self):
		# Test all combinations of X and y types
		X_types = [
			np.array([[1, 2, 3], [5, 6, 100], [3, 4, 5]]),
			np.array([[1, 2, 3], [5, 6, 100], [3, 4, 5]]),
			pd.DataFrame({"A": [1, 5, 3], "B": [2, 6, 4], "C": [3, 100, 5]}),
			pl.DataFrame({"A": [1, 5, 3], "B": [2, 6, 4], "C": [3, 100, 5]}),
			pl.LazyFrame({"A": [1, 5, 3], "B": [2, 6, 4], "C": [3, 100, 5]}),
		]
		y_types = [
			np.array([4, 4]),
			np.array([[4], [4]]),
			pd.DataFrame({"D": [4, 4, 6, 4]}),
			pl.DataFrame({"D": [4, 4, 6, 4]}),
			pl.LazyFrame({"D": [4, 4, 6, 4]}),
		]

		for X, y in zip(X_types, y_types):
			with self.assertRaises(ValueError):
				OutlierRemover.remove_outliers(X, y, threshold=1.4)

	def test_remove_outliers_incorrect_type(self):
		X_types = [
			np.array([[1, 2, 3], [5, 6, 100], [3, 4, 5]]),
			pd.DataFrame({"A": [1, 5, 3], "B": [2, 6, 4], "C": [3, 100, 5]}),
			pl.DataFrame({"A": [1, 5, 3], "B": [2, 6, 4], "C": [3, 100, 5]}),
			pl.LazyFrame({"A": [1, 5, 3], "B": [2, 6, 4], "C": [3, 100, 5]}),
		]
		y_types = [
			np.array([4, 4, 6]),  # Correct shape but incompatible type
			pd.DataFrame({"D": [4, 4, 6]}),
			pl.DataFrame({"D": [4, 4, 6]}),
			pl.LazyFrame({"D": [4, 4, 6]}),
		]

		for X in X_types:
			for y in y_types:
				if not isinstance(X, type(y)):  # Ensure mismatched types
					with self.assertRaises(ValueError):
						OutlierRemover.remove_outliers(X, y, threshold=1.4)

	# def test_remove_outliers_incorrect_threshold(self):


if __name__ == "__main__":
	unittest.main()
