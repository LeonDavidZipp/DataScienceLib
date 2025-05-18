import polars as pl
from typing import Literal


class Extrapolator:
	def __init__(
		self,
		interpolation_method: Literal["linear", "nearest"] = "linear",
		fill_value_if_only_null_values: int | float | None = None,
		val_before_first_non_null_val: int | float | None = None,
		val_after_last_non_null_val: int | float | None = None,
	):
		"""
		Args:
			interpolation_method (Literal["linear", "nearest"]): whether missing values inbetween non-null values
				should be filled linearly or with the nearest non-null value
			fill_value_if_only_null_values (int | float | None): The default fill value if the series consists of
				only Null values
			val_before_first_non_null_val (int | float | None): The value to insert before the first non_null_value. If None,
				this function linearly fills all values up to the first non-null value.
			val_after_last_non_null_val (int | float | None): The value to insert after the last non_null_value. If None,
				this function linearly fills all values after the last non-null value.

		"""
		self.interpolation_method = interpolation_method
		self.fill_value_if_only_null_values = fill_value_if_only_null_values
		self.val_before_first_non_null_val = val_before_first_non_null_val
		self.val_after_last_non_null_val = val_after_last_non_null_val

	def fill_regular(self, col: pl.Series) -> pl.Series:
		"""
		Simply fills the time series linearly from start to finish. Interpolates first, then extrapolates missing
		outside values.

		Args:
			col (pl.Series): The series to be extrapolated.
			fill_value_if_only_null_values (int | float | None): The default fill value if the series consists of
				only Null values

		Returns:
			pl.Series: The extrapolated series.

		Example:
			>>> import polars as pl
			>>> from your_module import extrapolate_series
			>>> series = pl.Series([None, None, 5, 6, 7, None, 9, None])
			>>> extrapolate_series(series)
			shape: (8,)
			Series: '' [f64]
			[
				3.0
				4.0
				5.0
				6.0
				7.0
				8.0
				9.0
				10.0
			]
		"""

		# first interpolate so a trend can actually be inferred
		col = col.interpolate(method=self.interpolation_method)  # type: ignore
		no_nulls = col.drop_nulls()
		if no_nulls.len() == 0:
			return col.fill_null(value=self.fill_value_if_only_null_values)
		elif no_nulls.len() == 1:
			return col.fill_null(value=col.median())

		avg_increase = col.diff(n=1).mean()
		first_non_null_val = no_nulls.first()
		first_non_null_idx = col.is_not_null().arg_true().first()
		last_non_null_val = no_nulls.last()
		last_non_null_idx = col.is_not_null().arg_true().last()

		first_val = first_non_null_val - avg_increase * first_non_null_idx  # type: ignore
		last_val = last_non_null_val + avg_increase * (  # type: ignore
			max(0, col.len() - 1 - last_non_null_idx)  # type: ignore
		)

		if (first_non_null_idx is None) or (last_non_null_idx is None):
			return col

		col = (
			pl.DataFrame({"col": col})
			.with_row_index(name="index")
			.select(
				pl.when((pl.col("index") == 0) & (first_non_null_idx > 0))  # type: ignore
				.then(pl.lit(first_val))
				.when(
					(pl.col("index") == (col.len() - 1))
					& (last_non_null_idx < (col.len() - 1))  # type: ignore
				)
				.then(pl.lit(last_val))
				.otherwise(pl.col("col"))
				.alias("col"),
			)
			.to_series()
			.interpolate(method="linear")
		)

		return col

	def fill_timeseries(
		self,
		col: pl.Series,
	) -> pl.Series:
		"""
		Linearly fills a time series respecting missing values before the first non-null value or after the last non-null value
		can indicate a time series has started or ended. Interpolates first, then extrapolates missing outside values. I all
		val_[...]_val variables are None, behaves exactly like fill_regular

		Args:
			col (pl.Series): The series to be extrapolated.
			val_before_first_non_null_val: (int | float | None): The value to insert before the first non_null_value. If None,
				this function linearly fills all values up to the first non-null value.
			val_after_last_non_null_val: (int | float | None): The value to insert after the last non_null_value. If None,
				this function linearly fills all values after the last non-null value.

		Returns:
			pl.Series: The extrapolated series.

		Examples:
			>>> import polars as pl
			>>> from your_module import extrapolate_series
			>>> series = pl.Series([None, None, 5, 6, 7, None, 9, None, None])
			>>> extrapolate_series(series, val_before_first_non_null_val=0)
			shape: (9,)
			Series: '' [f64]
			[
				0.0
				0.0
				5.0
				6.0
				7.0
				8.0
				9.0
				10.0
				11.0
			]

			>>> series = pl.Series([None, None, 5, 6, 7, None, 9, None, None])
			>>> extrapolate_series(series, val_after_last_non_null_val=100)
			shape: (9,)
			Series: '' [f64]
			[
				3.0
				4.0
				5.0
				6.0
				7.0
				8.0
				9.0
				100.0
				100.0
			]

			>>> series = pl.Series([None, None, 5, 6, 7, None, 9, None, None])
			>>> extrapolate_series(
			...     series,
			...     val_before_first_non_null_val=99,
			...     val_after_last_non_null_val=100,
			... )
			shape: (9,)
			Series: '' [f64]
			[
				99.0
				99.0
				5.0
				6.0
				7.0
				8.0
				9.0
				100.0
				100.0
			]
		"""

		if (
			self.val_before_first_non_null_val is None
			and self.val_after_last_non_null_val is None
		):
			return self.fill_regular(col)

		col = col.interpolate(method=self.interpolation_method)  # type: ignore
		no_nulls = col.drop_nulls()
		if no_nulls.len() == 0:
			return col.fill_null(value=self.fill_value_if_only_null_values)
		elif no_nulls.len() == 1:
			return col.fill_null(value=col.mean())

		avg_increase = col.diff(n=1).mean()
		first_non_null_idx = col.is_not_null().arg_true().first()
		first_non_null_val = no_nulls.first()
		last_non_null_val = no_nulls.last()
		last_non_null_idx = col.is_not_null().arg_true().last()
		first_val = first_non_null_val - avg_increase * first_non_null_idx  # type: ignore
		last_val = last_non_null_val + avg_increase * (  # type: ignore
			max(0, col.len() - 1 - last_non_null_idx)  # type: ignore
		)

		df = pl.DataFrame({"col": col}).with_row_index(name="index")

		if isinstance(self.val_before_first_non_null_val, (int, float)):
			df = df.with_columns(
				pl.when(
					(pl.col("index") < first_non_null_idx) & (pl.col("col").is_null())
				)
				.then(self.val_before_first_non_null_val)
				.otherwise(pl.col("col"))
				.alias("col")
			)
		elif self.val_before_first_non_null_val is None:
			df = df.with_columns(
				pl.when(
					(pl.col("index") <= first_non_null_idx) & (pl.col("col").is_null())
				)
				.then(first_val + avg_increase * pl.col("index"))  # type: ignore
				.otherwise(pl.col("col"))
				.alias("col")
			)

		if isinstance(self.val_after_last_non_null_val, (int, float)):
			df = df.with_columns(
				pl.when(
					(pl.col("index") >= last_non_null_idx) & (pl.col("col").is_null())
				)
				.then(self.val_after_last_non_null_val)
				.otherwise(pl.col("col"))
				.alias("col")
			)
		elif self.val_after_last_non_null_val is None:
			df = df.with_columns(
				pl.when(
					(pl.col("index") >= last_non_null_idx) & (pl.col("col").is_null())
				)
				.then(last_val - avg_increase * (col.len() - 1 - pl.col("index")))  # type: ignore
				.otherwise(pl.col("col"))
				.alias("col")
			)

		return df.select(pl.col("col")).to_series()
