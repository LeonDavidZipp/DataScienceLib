import polars as pl
import polars.selectors as cs


class OutlierSmoother:
	def __init__(self, max_zscore: float = 3.0):
		"""
		Initializes the OutlierSmoother with a specified maximum Z-score.

		Args:
		        max_zscore (float): maximum Z-score for identifying outliers to smooth.
		"""
		self.max_zscore = max_zscore

	def smooth_col_(self, X: pl.Series) -> pl.Series:
		max_val = X.mean() + self.max_zscore * X.std()  # type: ignore
		min_val = X.mean() - self.max_zscore * X.std()  # type: ignore
		if X.dtype.is_integer():
			max_val = int(max_val)  # type: ignore
			min_val = int(min_val)  # type: ignore
		res = (
			pl.DataFrame(X)
			.select(
				pl.when(cs.by_name(X.name) > max_val)  # type: ignore
				.then(max_val)  # type: ignore
				.when(cs.by_name(X.name) < min_val)  # type: ignore
				.then(min_val)  # type: ignore
				.otherwise(cs.by_name(X.name))
			)
			.to_series()
		)
		return res

	def smooth(self, X: pl.DataFrame) -> pl.DataFrame:
		"""
		Smooths numeric outliers in the DataFrame by replacing them with the mean of the column.

		Args:
			X (pl.DataFrame): Input DataFrame.

		Returns:
			pl.DataFrame: DataFrame with smoothed outliers.
		"""
		columns = X.columns
		return X.with_columns(cs.numeric().map_batches(self.smooth_col_)).select(
			columns
		)
