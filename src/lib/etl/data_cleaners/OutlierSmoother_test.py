import unittest
import polars as pl
import polars.testing as pl_testing
from src.lib.etl.data_cleaners.OutlierSmoother import OutlierSmoother


class TestOutlierSmoother(unittest.TestCase):
	def test_smooth_polars_max(self):
		# Test smoothing on a Polars DataFrame
		X = pl.DataFrame(
			{"A": [1, 2, 3, 100], "B": [5, 6, 7, 200], "C": [3, 4, 5, 300]}
		)
		zscore = 1
		smoother = OutlierSmoother(max_zscore=zscore)
		X_smoothed = smoother.smooth(X)

		# Expected output
		a_max = (
			X.select(pl.col("A").mean()).item()
			+ zscore * X.select(pl.col("A").std()).item()
		)
		b_max = (
			X.select(pl.col("B").mean()).item()
			+ zscore * X.select(pl.col("B").std()).item()
		)
		c_max = (
			X.select(pl.col("C").mean()).item()
			+ zscore * X.select(pl.col("C").std()).item()
		)
		X_exp = pl.DataFrame(
			{
				"A": [1, 2, 3, int(a_max)],
				"B": [5, 6, 7, int(b_max)],
				"C": [3, 4, 5, int(c_max)],
			}
		)

		self.assertIsInstance(X_smoothed, pl.DataFrame)
		pl_testing.assert_frame_equal(X_smoothed, X_exp, check_exact=False)

	def test_smooth_polars_min(self):
		# Test smoothing on a Polars DataFrame
		X = pl.DataFrame(
			{"A": [1, 2, 3, -100], "B": [5, 6, 7, -200], "C": [3, 4, 5, -300]}
		)
		zscore = 1
		smoother = OutlierSmoother(max_zscore=zscore)
		X_smoothed = smoother.smooth(X)

		# Expected output
		a_min = (
			X.select(pl.col("A").mean()).item()
			- zscore * X.select(pl.col("A").std()).item()
		)
		b_min = (
			X.select(pl.col("B").mean()).item()
			- zscore * X.select(pl.col("B").std()).item()
		)
		c_min = (
			X.select(pl.col("C").mean()).item()
			- zscore * X.select(pl.col("C").std()).item()
		)
		X_exp = pl.DataFrame(
			{
				"A": [1, 2, 3, int(a_min)],
				"B": [5, 6, 7, int(b_min)],
				"C": [3, 4, 5, int(c_min)],
			}
		)

		self.assertIsInstance(X_smoothed, pl.DataFrame)
		pl_testing.assert_frame_equal(X_smoothed, X_exp, check_exact=False)


if __name__ == "__main__":
	unittest.main()
