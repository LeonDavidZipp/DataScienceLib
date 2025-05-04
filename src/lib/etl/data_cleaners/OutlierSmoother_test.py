import unittest
import polars as pl
import polars.testing as pl_testing
from src.lib.etl.data_cleaners.OutlierSmoother import OutlierSmoother


class TestOutlierSmoother(unittest.TestCase):
	def test_smooth_polars(self):
		# Test smoothing on a Polars DataFrame
		X = pl.DataFrame({
			"A": [1, 2, 3, 100],
			"B": [5, 6, 7, 200],
			"C": [3, 4, 5, 300]
		})
		smoother = OutlierSmoother(max_zscore=2.0)
		X_smoothed = smoother.smooth(X)

		# Expected output
		X_exp = pl.DataFrame({
			"A": [1, 2, 3, 3 + 2 * 1.118033988749895],  # Mean + 2 * StdDev
			"B": [5, 6, 7, 7 + 2 * 1.118033988749895],
			"C": [3, 4, 5, 5 + 2 * 1.118033988749895]
		})

		self.assertIsInstance(X_smoothed, pl.DataFrame)
		pl_testing.assert_frame_equal(X_smoothed, X_exp, check_exact=False)


if __name__ == "__main__":
	unittest.main()