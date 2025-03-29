import numpy as np
import polars as pl
import pandas as pd
import scipy.stats as st
from typing import Tuple, Optional

class OutlierRemover:
	def __init__(self):
		pass

	@staticmethod
	def remove_outliers(
		X: pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray,
		y: pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray = None,
		threshold: float = 3.0
	) -> Tuple[pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, \
			Optional[pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray]]:
		"""
		Removes outliers based on Z-score.

		Args:
			X: Input features (polars.DataFrame, polars.LazyFrame, pandas.DataFrame or numpy.ndarray).
			y: Optional target values (polars.DataFrame, polars.LazyFrame, pandas.DataFrame or numpy.ndarray).
			If not provided, last column is assumed to be the target. X and y must be of the same type.
			threshold: Z-score threshold for identifying outliers.

		Returns:
			Filtered X (and y if provided) without outliers.
		"""
		if y is None:
			if isinstance(X, pl.DataFrame):
				return OutlierRemover._polars_no_y(X, threshold)
			elif isinstance(X, pl.LazyFrame):
				return OutlierRemover._polars_lazy_no_y(X, threshold)
			elif isinstance(X, pd.DataFrame):
				return OutlierRemover._pandas_no_y(X, threshold)
			elif isinstance(X, np.ndarray):
				return OutlierRemover._numpy_no_y(X, threshold)
			else:
				raise ValueError(f"Unsupported data type. Expected polars.DataFrame, pandas.DataFrame or numpy.ndarray twice, instead got {type(X)} and {type(y)}")
		else:
			if isinstance(X, pl.DataFrame) and isinstance(y, pl.DataFrame):
				return OutlierRemover._polars(X, y, threshold)
			elif isinstance(X, pl.LazyFrame) and isinstance(y, pl.LazyFrame):
				return OutlierRemover._polars_lazy(X, y, threshold)
			elif isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
				return OutlierRemover._pandas(X, y, threshold)
			elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
				return OutlierRemover._numpy(X, y, threshold)
			else:
				raise ValueError(f"Unsupported data type. Expected polars.DataFrame, pandas.DataFrame or numpy.ndarray, instead got {type(X)}")
			
	@staticmethod
	def _polars(X: pl.DataFrame, y: pl.DataFrame, threshold):
		if y.shape[1] != 1:
			raise ValueError("y must be a 1D DataFrame.")
		elif X.shape[0] != y.shape[0]:
			raise ValueError("X and y must have the same number of rows.")
		zscores = np.abs(
			st.zscore(
				X.to_numpy(),
				axis=0,
				nan_policy="omit"
			)
		).max(axis=1)
		
		X = (
			X.with_columns(pl.Series(values=zscores).alias("zscores"))
			.filter(pl.col("zscores") <= threshold)
			.drop(["zscores"])
		)
		y = (
			y.with_columns(pl.Series(values=zscores).alias("zscores"))
			.filter(pl.col("zscores") <= threshold)
			.drop(["zscores"])
		)
		return X, y


	@staticmethod
	def _polars_lazy(X: pl.LazyFrame, y: pl.LazyFrame, threshold):
		X, y = OutlierRemover._polars(X.collect(), y.collect(), threshold)
		return X.lazy(), y.lazy()

	@staticmethod
	def _pandas(X: pd.DataFrame, y: pd.DataFrame, threshold):
		X, y = OutlierRemover._polars(pl.from_pandas(X), pl.from_pandas(y), threshold)
		return X.to_pandas(), y.to_pandas()

	@staticmethod
	def _numpy(X: np.ndarray, y: np.ndarray, threshold: float):
		# Ensure y is either 1D or a 2D array with a single column
		if y.ndim == 2 and y.shape[1] == 1:
			y_is_2d = True  # Track if y was originally 2D
			y = y.ravel()  # Flatten to 1D array
		elif y.ndim == 1:
			y_is_2d = False
		else:
			raise ValueError("y must be either a 1D array or a 2D array with a single column.")

		# Check that X and y have compatible shapes
		if X.shape[0] != y.shape[0]:
			raise ValueError("X and y must have the same number of rows.")

		# Calculate Z-scores for X
		zscores: np.ndarray = np.abs(
			st.zscore(
				X,
				axis=0,
				nan_policy="omit"
			)
		).max(axis=1)

		# Filter rows based on the Z-score threshold
		X_filtered = X[zscores <= threshold]
		y_filtered = y[zscores <= threshold]

		# If y was originally 2D, reshape it back to 2D
		if y_is_2d:
			y_filtered = y_filtered.reshape(-1, 1)

		return X_filtered, y_filtered

	@staticmethod
	def _polars_no_y(X: pl.DataFrame, threshold: float)-> pl.DataFrame:
		zscores = np.abs(
			st.zscore(
				X.select(pl.exclude(X.columns[-1])).to_numpy(),
				axis=0,
				nan_policy="omit"
			)
		).max(axis=1)
		return (
			X.with_columns(pl.Series(values=zscores).alias("zscores"))
			.filter(pl.col("zscores") <= threshold)
			.drop(["zscores"])
		)

	@staticmethod
	def _polars_lazy_no_y(X: pl.LazyFrame, threshold: float)-> pl.LazyFrame:
		return OutlierRemover._polars_no_y(
			X.collect(),
			threshold
		).lazy()

	@staticmethod
	def _pandas_no_y(X: pd.DataFrame, threshold: float)-> pd.DataFrame:
		return OutlierRemover._polars_no_y(pl.from_pandas(X), threshold).to_pandas()

	@staticmethod
	def _numpy_no_y(X, threshold: float)-> np.ndarray:
		zscores: np.ndarray = np.abs(
			st.zscore(
				X[:, :-1],
				axis=0,
				nan_policy="omit"
			)
		).max(axis=1)
		return X[zscores <= threshold]
