import polars as pl
import polars.testing as pt
import unittest
from etl.Cleaner import Cleaner

class TestCleaner(unittest.TestCase):
	def setUp(self):
		data = {
			r"A%a": [1 , 2, None, 1, 10, -33],
			"b)": ["a", None, None, "a", "b", "c"],
			"c_": [3.2, 3.2, 2.3, 3.2, 1.0, 2.0],
			"d": ["a", None, "d", "a", "b", None]
		}

		self.cleaner = Cleaner()
		self.lf = pl.LazyFrame(data)
		self.cols = self.lf.collect_schema().names()

	def test_cols_from_provided_cols_(self):
		"""
		Test the cols_from_provided_cols_ method of the Cleaner class.
		"""
		tests = {
			tuple(list(self.cols) + ["e", "f"]): list(self.cols),
			"b)": ["b)"],
			None: [],
			(): []
		}

		for inp, expected in tests.items():
			if isinstance(inp, tuple):
				inp = list(inp)
			outp = self.cleaner.cols_from_provided_cols_(self.lf, inp)
			self.assertEqual(outp, expected)

	def test_adjust_column_names(self):
		"""
		Test the adjust_column_names method of the Cleaner class.
		"""
		expected = ["apercenta", "b", "c", "d"]
		outp = self.cleaner.adjust_column_names(self.lf).collect_schema().names()
		self.assertEqual(outp, expected)

	def test_drop(self):
		"""
		Test the drop method of the Cleaner class.
		"""
		# Drop a single column
		result = self.cleaner.drop(self.lf, "b)")
		self.assertNotIn("b)", result.collect_schema().names())

		# Drop multiple columns
		result = self.cleaner.drop(self.lf, ["A%a", "c_"])
		self.assertNotIn("A%a", result.collect_schema().names())
		self.assertNotIn("c_", result.collect_schema().names())

		# Drop with None (should return the same LazyFrame)
		result = self.cleaner.drop(self.lf, None)
		self.assertEqual(result.collect_schema().names(), self.lf.collect_schema().names())

	def test_remove_nulls(self):
		"""
		Test the remove_nulls method of the Cleaner class.
		"""
		tests = {
			0.0: 6,
			None: 6,
			0.51: 4,
			1.0: 3
		}
		for threshold, expected in tests.items():
			result = self.cleaner.remove_nulls(self.lf, non_null_threshold=threshold)
			self.assertEqual(result.collect().shape[0], expected)

	def test_remove_duplicates(self):
		"""
		Test the remove_duplicates method of the Cleaner class.
		"""
		# Remove duplicates based on all columns
		result = self.cleaner.remove_duplicates(self.lf)
		self.assertEqual(len(result.collect()), 5)  # Only unique rows remain

	def test_fill_nulls(self):
		"""
		Test the fill_nulls method of the Cleaner class.
		"""
		# Fill nulls with a constant value
		result = self.cleaner.fill_nulls(self.lf)
		print(result.null_count().collect())
		self.assertEqual(result.null_count().collect().sum_horizontal().item(), 0)  # No nulls should remain

	def test_sort(self):
		"""
		Test the sort method of the Cleaner class.
		"""

		result = self.cleaner.sort(self.lf, cols=None, descending=False)
		pt.assert_frame_equal(result, self.lf.sort(by=self.lf.collect_schema().names(), descending=False))

		result = self.cleaner.sort(self.lf, cols="A%a", descending=False)
		pt.assert_frame_equal(result, self.lf.sort(by=["A%a"], descending=False))

		result = self.cleaner.sort(self.lf, cols="A%a", descending=True)
		pt.assert_frame_equal(result, self.lf.sort(by=["A%a"], descending=True))

		result = self.cleaner.sort(self.lf, cols=["d", "A%a"], descending=False)
		pt.assert_frame_equal(result, self.lf.sort(by=["d", "A%a"], descending=False))

