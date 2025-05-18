import polars as pl
import polars.selectors as cs
from typing import Literal
from .casters import ForeCaster, BackCaster
from .fillers import (
	MultiTimeSeriesGapFiller,
)


class NumericalTimeSeriesExtender:
	def __init__(
		self,
		period: Literal["daily", "monthly", "yearly"] = "monthly",
		steps: int = 24,
		only_return_extended: bool = False,
	):
		if period not in ["daily", "monthly", "yearly"]:
			raise ValueError(
				"Expected period to be of type Literal['daily', 'monthly', 'yearly']."
			)
		self.period = period
		self.steps = steps
		self.only_return_extended = only_return_extended

	def forward(self, time_series: pl.Series) -> pl.Series:
		"""
		Tries to linearly predict time series into the future.

		Args:
			time_series (pl.Series): Time series data.

		Returns:
			pl.Series: Extended time series data.
		"""

		time_series = time_series.interpolate(method="linear")
		data = ForeCaster(
			self.period,  # type: ignore
			self.steps,
			self.only_return_extended,  # type: ignore
		).fit_transform(time_series.to_numpy())  # type: ignore
		return pl.Series(data, nan_to_null=True)

	def backward(self, time_series: pl.Series) -> pl.Series:
		"""
		Tries to linearly predict time series into the past.

		Args:
			time_series (pl.Series): Time series data.

		Returns:
			pl.Series: Extended time series data.
		"""

		time_series = time_series.interpolate(method="linear")
		data = BackCaster(
			self.period,  # type: ignore
			self.steps,
			self.only_return_extended,  # type: ignore
		).fit_transform(time_series.to_numpy())  # type: ignore
		return pl.Series(data, nan_to_null=True)


class DateTimeSeriesExtender:
	def __init__(
		self,
		period: Literal["daily", "monthly", "yearly"] = "monthly",
		steps: int = 24,
		only_return_extended: bool = False,
	):
		if period not in ["daily", "monthly", "yearly"]:
			raise ValueError(
				"Expected period to be of type Literal['daily', 'monthly', 'yearly']."
			)
		self.period = period
		self.steps = steps
		self.only_return_extended = only_return_extended
		self.period_to_offset = {"daily": "d", "monthly": "mo", "yearly": "y"}

	def forward(self, time_series: pl.Series) -> pl.Series:
		"""
		Tries to linearly predict time series into the future.

		Args:
			time_series (pl.Series): Time series data.

		Returns:
			pl.Series: Extended time series data.
		"""

		offset = self.period_to_offset.get(self.period)
		if self.only_return_extended:
			start = time_series.tail(1).dt.offset_by(f"1{offset}").first()
		else:
			start = time_series.min()  # type: ignore
		end = time_series.tail(1).dt.offset_by(f"{self.steps}{offset}").first()  # type: ignore
		return pl.date_range(start=start, end=end, interval=f"1{offset}", eager=True)  # type: ignore

	def backward(self, time_series: pl.Series) -> pl.Series:
		"""
		Tries to linearly predict time series into the past.

		Args:
			time_series (pl.Series): Time series data.

		Returns:
			pl.Series: Extended time series data.
		"""

		offset = self.period_to_offset.get(self.period)
		if self.only_return_extended:
			end = time_series.head(1).dt.offset_by(f"-1{offset}").first()
		else:
			end = time_series.max()  # type: ignore
		start = time_series.head(1).dt.offset_by(f"-{self.steps}{offset}").first()  # type: ignore
		return pl.date_range(start=start, end=end, interval=f"1{offset}", eager=True)  # type: ignore


class NonNumericalTimeSeriesExtender:
	def __init__(
		self,
		strategy: Literal[
			"forward", "backward", "min", "max", "mean", "zero", "one"
		] = "forward",
		steps: int = 24,
		only_return_extended: bool = False,
	):
		if strategy not in ["forward", "backward", "min", "max", "mean", "zero", "one"]:
			raise ValueError(
				"Expected period to be of type Literal['daily', 'monthly', 'yearly']."
			)
		self.strategy = strategy
		self.steps = steps
		self.only_return_extended = only_return_extended

	def forward(self, time_series: pl.Series) -> pl.Series:
		"""
		Tries to linearly predict time series into the future.

		Args:
			time_series (pl.Series): Time series data.

		Returns:
			pl.Series: Extended time series data.
		"""

		ext_series = pl.Series(
			values=[None] * self.steps, dtype=time_series.dtype
		).fill_null(strategy=self.strategy)  # type: ignore
		if self.only_return_extended:
			return ext_series
		else:
			return pl.concat([time_series, ext_series])

	def backward(self, time_series: pl.Series) -> pl.Series:
		"""
		Tries to linearly predict time series into the past.

		Args:
			time_series (pl.Series): Time series data.

		Returns:
			pl.Series: Extended time series data.
		"""

		ext_series = pl.Series(
			values=[None] * self.steps, dtype=time_series.dtype
		).fill_null(strategy=self.strategy)  # type: ignore
		if self.only_return_extended:
			return ext_series
		else:
			return pl.concat([ext_series, time_series])


class MultiTimeSeriesExtender:
	def __init__(
		self,
		period: Literal["daily", "monthly", "yearly"] = "monthly",
		steps: int = 24,
		binary_value: Literal[0, 1] = 0,
		boolean_value: Literal[0, 1] = 0,
		duration_strategy: Literal[
			"forward", "backward", "min", "max", "mean", "zero", "one"
		] = "forward",
		string_strategy: Literal["forward", "backward", "zero", "one"] = "forward",
		time_strategy: Literal[
			"forward", "backward", "min", "max", "mean", "zero", "one"
		] = "mean",
		only_return_extended: bool = False,
	):
		if period not in ["daily", "monthly", "yearly"]:
			raise ValueError(
				"Expected period to be of type Literal['daily', 'monthly', 'yearly']."
			)
		self.period = period
		self.period_to_offset = {"daily": "d", "monthly": "mo", "yearly": "y"}
		self.steps = steps
		self.binary_value = binary_value
		self.boolean_value = boolean_value
		self.duration_strategy = duration_strategy
		self.string_strategy = string_strategy
		self.time_strategy = time_strategy
		self.only_return_extended = only_return_extended

	def forward(self, X: pl.DataFrame, date_col: str) -> pl.DataFrame:
		"""
		Tries to linearly predict time series into the future.

		Args:
			X (pl.DataFrame): Time series data.
			date_col (str): Name of the date column

		Returns:
			pl.DataFrame: Extended time series data.
		"""

		x_new = MultiTimeSeriesGapFiller(
			self.binary_value,  # type: ignore
			self.boolean_value,  # type: ignore
			self.duration_strategy,  # type: ignore
			self.string_strategy,  # type: ignore
			self.time_strategy,  # type: ignore
		).fill(X, date_col)

		cols = x_new.columns

		nts = NumericalTimeSeriesExtender(
			self.period,  # type: ignore
			self.steps,
			self.only_return_extended,  # type: ignore
		)  # type: ignore
		dts = DateTimeSeriesExtender(self.period, self.steps)  # type: ignore
		nnts = NonNumericalTimeSeriesExtender("zero", self.steps)
		nnts2 = NonNumericalTimeSeriesExtender("mean", self.steps)
		nnts3 = NonNumericalTimeSeriesExtender("forward", self.steps)
		return (
			pl.DataFrame()
			.with_columns(
				x_new.select(cs.date().map_batches(dts.forward)),
			)
			.with_columns(
				x_new.select(cs.numeric().map_batches(nts.forward)),
			)
			.with_columns(
				x_new.select(cs.binary().map_batches(nnts.forward)),
			)
			.with_columns(
				x_new.select(cs.boolean().map_batches(nnts.forward)),
			)
			.with_columns(
				x_new.select(cs.duration().map_batches(nnts2.forward)),
			)
			.with_columns(
				x_new.select(
					cs.string(include_categorical=True).map_batches(nnts3.forward)
				),
			)
			.with_columns(
				x_new.select(cs.time().map_batches(nnts2.forward)),
			)
			.select(cols)
		)

	def backward(self, X: pl.DataFrame, date_col: str) -> pl.DataFrame:
		"""
		Tries to linearly predict time series into the past.

		Args:
			X (pl.DataFrame): Time series data.
			date_col (str): Name of the date column

		Returns:
			pl.DataFrame: Extended time series data.
		"""

		# interpolate dataframe
		x_new = MultiTimeSeriesGapFiller(
			self.binary_value,  # type: ignore
			self.boolean_value,  # type: ignore
			self.duration_strategy,  # type: ignore
			self.string_strategy,  # type: ignore
			self.time_strategy,  # type: ignore
		).fill(X, date_col)

		cols = x_new.columns

		nts = NumericalTimeSeriesExtender(
			self.period,  # type: ignore
			self.steps,
			self.only_return_extended,
		)  # type: ignore
		dts = DateTimeSeriesExtender(self.period, self.steps, self.only_return_extended)  # type: ignore
		nnts = NonNumericalTimeSeriesExtender(
			"zero", self.steps, self.only_return_extended
		)
		nnts2 = NonNumericalTimeSeriesExtender(
			"mean", self.steps, self.only_return_extended
		)
		nnts3 = NonNumericalTimeSeriesExtender(
			"backward", self.steps, self.only_return_extended
		)
		print(f"df shape before: {x_new.shape}")
		x_new = (
			pl.DataFrame()
			.with_columns(
				x_new.select(cs.date().map_batches(dts.backward)),
			)
			.with_columns(
				x_new.select(cs.numeric().map_batches(nts.backward)),
			)
			.with_columns(
				x_new.select(cs.binary().map_batches(nnts.backward)),
			)
			.with_columns(
				x_new.select(cs.boolean().map_batches(nnts.backward)),
			)
			.with_columns(
				x_new.select(cs.duration().map_batches(nnts2.backward)),
			)
			.with_columns(
				x_new.select(
					cs.string(include_categorical=True).map_batches(nnts3.backward)
				),
			)
			.with_columns(
				x_new.select(cs.time().map_batches(nnts2.backward)),
			)
			.select(cols)
		)
		print(f"df shape after: {x_new.shape}")
		return x_new
