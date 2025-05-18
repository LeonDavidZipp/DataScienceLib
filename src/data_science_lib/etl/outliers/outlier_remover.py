import numpy as np
import polars as pl
import scipy.stats as st  # type: ignore
from numpy.typing import NDArray
from typing import Any


class OutlierRemover:
	def __init__(self):
		pass

	@staticmethod
	def remove(
		X: pl.DataFrame | pl.LazyFrame | NDArray[Any],
		y: pl.DataFrame | pl.LazyFrame | NDArray[Any] | None = None,
		threshold: float = 3.0,
	) -> (
		tuple[
			pl.DataFrame | pl.LazyFrame | NDArray[Any],
			pl.DataFrame | pl.LazyFrame | NDArray[Any],
		]
		| pl.DataFrame
		| pl.LazyFrame
		| NDArray[Any]
	):
		"""
		Removes outliers based on Z-score. NOT meant to be part of a pipeline, but applied to the data beforehand.

		Args:
		        X: Input features (polars.DataFrame, polars.LazyFrame, pandas.DataFrame or numpy.ndarray).
		        y: Optional target values (polars.DataFrame, polars.LazyFrame, pandas.DataFrame or numpy.ndarray).
		        y_count: Last n columns of X are considered the target. Ignored if y is provided.
		        If not provided, last column is assumed to be the target. X and y must be of the same type.
		        threshold: Z-score threshold for identifying outliers.

		Returns:
		        Filtered X (and y if provided) without outliers.
		"""

		match (X, y):
			case (pl.DataFrame(), pl.DataFrame()) | (pl.DataFrame(), None):
				x_new, y_new = OutlierRemover._polars(X, y, threshold)  # type: ignore
			case (pl.LazyFrame(), pl.LazyFrame()) | (pl.LazyFrame(), None):
				x_new, y_new = OutlierRemover._polars_lazy(X, y, threshold)  # type: ignore
			case (np.ndarray(), np.ndarray()) | (np.ndarray(), None):
				x_new, y_new = OutlierRemover._numpy(X, y, threshold)  # type: ignore
			case (_, _):
				raise ValueError(
					f"Unsupported data type. Expected polars.DataFrame, pandas.DataFrame or numpy.ndarray twice, instead got {type(X)} and {type(y)}"
				)

		if y_new is None:
			return x_new
		else:
			return x_new, y_new

	@staticmethod
	def _polars(
		X: pl.DataFrame, y: pl.DataFrame | None, threshold: float
	) -> tuple[pl.DataFrame, pl.DataFrame | None]:
		"""
		Removes outliers based on Z-score for polars DataFrames.
		Args:
		        X (polars.DataFrame): Input features.
		        y (polars.DataFrame): Target values.
		        threshold (float): Z-score threshold for identifying outliers.
		Returns:
		        Filtered X and y without outliers.
		"""

		zscores = np.abs(st.zscore(X.to_numpy(), axis=0, nan_policy="omit")).max(axis=1)  # type: ignore
		x_new = (
			X.with_columns(pl.Series(values=zscores).alias("zscores"))
			.filter(pl.col("zscores") <= threshold)
			.drop(["zscores"])
		)

		y_new = None
		if y is not None:
			if y.shape[1] != 1:
				raise ValueError("y must be a 1D DataFrame.")
			elif X.shape[0] != y.shape[0]:
				raise ValueError("X and y must have the same number of rows.")
			y_new = (
				y.with_columns(pl.Series(values=zscores).alias("zscores"))
				.filter(pl.col("zscores") <= threshold)
				.drop(["zscores"])
			)
		return x_new, y_new

	@staticmethod
	def _polars_lazy(
		X: pl.LazyFrame, y: pl.LazyFrame | None, threshold: float
	) -> tuple[pl.LazyFrame, pl.LazyFrame | None]:
		"""
		Removes outliers based on Z-score for polars LazyFrames.
		Args:
		        X (polars.LazyFrame): Input features.
		        y (polars.LazyFrame): Target values.
		        threshold (float): Z-score threshold for identifying outliers.
		Returns:
		        Filtered X and y without outliers.
		"""

		if y is not None:
			x_new, y_new = OutlierRemover._polars(X.collect(), y.collect(), threshold)  # type: ignore
		else:
			x_new, y_new = OutlierRemover._polars(X.collect(), None, threshold)  # type: ignore
		if y_new is not None:
			y_new = y_new.lazy()
		return x_new.lazy(), y_new

	@staticmethod
	def _numpy(
		X: NDArray[Any], y: NDArray[Any] | None, threshold: float
	) -> tuple[NDArray[Any], NDArray[Any] | None]:
		"""
		Removes outliers based on Z-score for numpy ndarray.
		Args:
		        X (numpy.ndarray): Input features.
		        y (numpy.ndarray): Target values.
		        threshold (float): Z-score threshold for identifying outliers.
		Returns:
		        Filtered X and y without outliers.
		"""

		# Calculate Z-scores for X
		zscores: NDArray[Any] = np.abs(st.zscore(X, axis=1, nan_policy="omit")).max(  # type: ignore
			axis=1
		)
		print("scores:", zscores)

		# Filter rows based on the Z-score threshold
		X_filtered = X[zscores <= threshold]
		y_filtered = None

		if y is not None:
			# Ensure y is either 1D or a 2D array with a single column
			if y.ndim == 2 and y.shape[1] == 1:
				y_is_2d = True  # Track if y was originally 2D
				y = y.ravel()  # Flatten to 1D array
			elif y.ndim == 1:
				y_is_2d = False
			else:
				raise ValueError(
					"y must be either a 1D array or a 2D array with a single column."
				)

			# Check that X and y have compatible shapes
			if X.shape[0] != y.shape[0]:
				raise ValueError("X and y must have the same number of rows.")

			y_filtered = y[zscores <= threshold]

			# If y was originally 2D, reshape it back to 2D
			if y_is_2d:
				y_filtered = y_filtered.reshape(-1, 1)

		return X_filtered, y_filtered
