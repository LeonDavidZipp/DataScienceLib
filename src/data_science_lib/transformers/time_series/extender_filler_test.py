import unittest
import polars.testing as pltest
import numpy as np
import polars as pl
import datetime as dt
from .casters import ForeCaster, BackCaster

from .extenders import MultiTimeSeriesGapFiller, NumericalTimeSeriesExtender
from .extrapolator import (
	Extrapolator,
)


class TestExtrapolator(unittest.TestCase):
	def setUp(self):
		self.ext = Extrapolator("linear", 0)
		return super().setUp()

	def test_constructor(self):
		self.assertEqual("linear", self.ext.interpolation_method)
		self.assertEqual(0, self.ext.fill_value_if_only_null_values)

	def test_fill_regular(self):
		input = pl.Series(
			name="col", values=[None, None, 5, 6, 7, None, 9, None], dtype=pl.Int8
		)
		expected = pl.Series(
			name="col", values=[3, 4, 5, 6, 7, 8, 9, 10], dtype=pl.Float64
		)
		output = self.ext.fill_regular(input)
		pltest.assert_series_equal(expected, output)

	def test_fill_regular_only_null(self):
		input = pl.Series(name="col", values=[None, None, None, None], dtype=pl.Int8)
		expected = pl.Series(name="col", values=[0, 0, 0, 0], dtype=pl.Float64)
		output = self.ext.fill_regular(input)
		pltest.assert_series_equal(expected, output)

	def test_fill_regular_one_non_null(self):
		input = pl.Series(name="col", values=[None, None, 4, None], dtype=pl.Int8)
		expected = pl.Series(name="col", values=[4, 4, 4, 4], dtype=pl.Float64)
		output = self.ext.fill_regular(input)
		pltest.assert_series_equal(expected, output)

	def test_fill_timeseries_both_none(self):
		"""
		should behave exactly like fill_regular
		"""

		input = pl.Series(
			name="col", values=[None, None, 5, 6, 7, None, 9, None], dtype=pl.Int8
		)
		output1 = self.ext.fill_regular(input)
		output2 = self.ext.fill_timeseries(input)
		pltest.assert_series_equal(output1, output2)

		input = pl.Series(name="col", values=[None, None, 4, None], dtype=pl.Int8)
		output1 = self.ext.fill_regular(input)
		output2 = self.ext.fill_timeseries(input)
		pltest.assert_series_equal(output1, output2)

		input = pl.Series(name="col", values=[None, None, 4, None], dtype=pl.Int8)
		output1 = self.ext.fill_regular(input)
		output2 = self.ext.fill_timeseries(input)
		pltest.assert_series_equal(output1, output2)

	def test_fill_timeseries_front(self):
		input = pl.Series(
			name="col", values=[None, None, 5, 6, 7, 8, 9, 10], dtype=pl.Int8
		)

		self.ext.val_before_first_non_null_val = 0
		self.ext.val_after_last_non_null_val = None
		expected = pl.Series(
			name="col", values=[0, 0, 5, 6, 7, 8, 9, 10], dtype=pl.Float64
		)
		output = self.ext.fill_timeseries(input)
		pltest.assert_series_equal(expected, output)

		self.ext.val_before_first_non_null_val = -99
		self.ext.val_after_last_non_null_val = None
		expected = pl.Series(
			name="col", values=[-99, -99, 5, 6, 7, 8, 9, 10], dtype=pl.Float64
		)
		output = self.ext.fill_timeseries(input)
		pltest.assert_series_equal(expected, output)

	def test_fill_timeseries_back(self):
		input = pl.Series(
			name="col", values=[None, None, 5, 6, 7, 8, None, None], dtype=pl.Int8
		)

		self.ext.val_before_first_non_null_val = None
		self.ext.val_after_last_non_null_val = 0
		expected = pl.Series(
			name="col", values=[3, 4, 5, 6, 7, 8, 0, 0], dtype=pl.Float64
		)
		output = self.ext.fill_timeseries(input)
		pltest.assert_series_equal(expected, output)

		self.ext.val_before_first_non_null_val = None
		self.ext.val_after_last_non_null_val = -99
		expected = pl.Series(
			name="col", values=[3, 4, 5, 6, 7, 8, -99, -99], dtype=pl.Float64
		)
		output = self.ext.fill_timeseries(input)
		pltest.assert_series_equal(expected, output)

	def test_fill_time_series_front_and_back(self):
		input = pl.Series(
			name="col", values=[None, None, 5, 6, 7, 8, None, None], dtype=pl.Int8
		)

		self.ext.val_before_first_non_null_val = 100
		self.ext.val_after_last_non_null_val = 0
		expected = pl.Series(
			name="col", values=[100, 100, 5, 6, 7, 8, 0, 0], dtype=pl.Float64
		)
		output = self.ext.fill_timeseries(input)
		pltest.assert_series_equal(expected, output)


class TestMultiTimeSeriesGapFiller(unittest.TestCase):
	def setUp(self):
		self.filler = MultiTimeSeriesGapFiller(
			b"0", False, "mean", "forward_then_backward", "mean"
		)
		return super().setUp()

	def test_constructor(self):
		self.assertEqual(b"0", self.filler.binary_value)
		self.assertEqual(False, self.filler.boolean_value)
		self.assertIsNone(self.filler.duration_strategy)
		self.assertEqual("forward_then_backward", self.filler.string_strategy)
		self.assertEqual("mean", self.filler.time_strategy)

	def test_fill(self):
		dates = pl.date_range(
			start=dt.date(2023, 1, 1),
			end=dt.date(2023, 6, 1),
			interval="1mo",
			eager=True,
		).filter([True, True, True, True, False, True])
		input = pl.DataFrame(
			{
				"date": dates,  # Skip April
				"a": pl.Series(
					"a", [b"abc", b"def", None, b"stu", b"vwx"], dtype=pl.Binary
				),
				"b": pl.Series(
					"b", [None, False, False, True, False], dtype=pl.Boolean
				),
				"c": pl.Series(
					"c",
					[
						None,
						dt.timedelta(days=3),
						dt.timedelta(days=4),
						dt.timedelta(days=7),
						dt.timedelta(days=10),
					],
					dtype=pl.Duration,
				),
				"d": pl.Series(
					"d", [None, "cherry", "fig", "grape", "lemon"], dtype=pl.String
				),
				"e": pl.Series(
					"e", [None, "cat", "dog", None, "cat"], dtype=pl.Categorical
				),
				"f": pl.Series(
					"f",
					[
						None,
						dt.time(11, 0),
						dt.time(12, 30),
						dt.time(17, 0),
						dt.time(21, 30),
					],
					dtype=pl.Time,
				),
			}
		)

		dates = pl.date_range(
			start=dt.date(2023, 1, 1),
			end=dt.date(2023, 6, 1),
			interval="1mo",
			eager=True,
		)
		expected = pl.DataFrame(
			{
				"date": pl.date_range(
					start=dt.date(2023, 1, 1),
					end=dt.date(2023, 6, 1),
					interval="1mo",
					eager=True,
				),
				"a": pl.Series(
					"a", [b"abc", b"def", b"0", b"stu", b"0", b"vwx"], dtype=pl.Binary
				),
				"b": pl.Series(
					"b", [False, False, False, True, False, False], dtype=pl.Boolean
				),
				"c": pl.Series(
					"c",
					[
						dt.timedelta(days=6),
						dt.timedelta(days=3),
						dt.timedelta(days=4),
						dt.timedelta(days=7),
						dt.timedelta(days=6),
						dt.timedelta(days=10),
					],
					dtype=pl.Duration,
				),
				"d": pl.Series(
					"d",
					["cherry", "cherry", "fig", "grape", "grape", "lemon"],
					dtype=pl.String,
				),
				"e": pl.Series(
					"e",
					["cat", "cat", "dog", "dog", "dog", "cat"],
					dtype=pl.Categorical,
				),
				"f": pl.Series(
					"f",
					[
						dt.time(15, 30),
						dt.time(11, 0),
						dt.time(12, 30),
						dt.time(17, 0),
						dt.time(15, 30),
						dt.time(21, 30),
					],
					dtype=pl.Time,
				),
			}
		)

		output = self.filler.fill(input, "date")
		pltest.assert_frame_equal(expected, output)


class TestForeCaster(unittest.TestCase):
	def setUp(self):
		self.data = np.random.randn(100)
		self.forecaster = ForeCaster(period="monthly", steps=24)

	def test_fit_transform(self):
		transformed_data = self.forecaster.fit_transform(self.data)  # type: ignore
		self.assertEqual(transformed_data.shape[0], 124)


class TestBackCaster(unittest.TestCase):
	def setUp(self):
		self.data = np.random.randn(100)
		self.backcaster = BackCaster(period="monthly", steps=24)

	def test_fit_transform(self):
		transformed_data = self.backcaster.fit_transform(self.data)  # type: ignore
		self.assertEqual(transformed_data.shape[0], 124)


class TestNumericalTimeSeriesExtender(unittest.TestCase):
	def setUp(self):
		self.data = pl.Series(np.random.randn(100))
		self.extender = NumericalTimeSeriesExtender(period="monthly", steps=24)

	def test_forward(self):
		extended_data = self.extender.forward(self.data)
		self.assertEqual(extended_data.shape[0], 124)

	def test_backward(self):
		extended_data = self.extender.backward(self.data)
		self.assertEqual(extended_data.shape[0], 124)


if __name__ == "__main__":
	unittest.main()
