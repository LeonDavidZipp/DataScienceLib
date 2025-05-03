import polars as pl
from typing import Literal
import src.lib.transformers.time_series.caster as cast


class Extender:
	def __init__(
		self,
		period: Literal["daily", "monthly", "yearly"],
		steps: int = 24,
	):
		self.period = period
		self.steps = steps

	def forward(self, time_series: pl.Series) -> pl.Series:
		"""
		Tries to linearly predict time series into the future.

		Args:
			time_series (pl.Series): Time series data.
			period (str): Period of the time series. Can be "daily", "monthly", or "yearly".
			steps (int): Number of steps to predict.

		Returns:
			pl.Series: Extended time series data.
		"""

		time_series = time_series.interpolate(method="linear")
		return cast.ForeCaster(self.period, self.steps).fit_transform(time_series)

	def backward(self, time_series: pl.Series) -> pl.Series:
		"""
		Tries to linearly predict time series into the past.

		Args:
			time_series (pl.Series): Time series data.
			period (str): Period of the time series. Can be "daily", "monthly", or "yearly".
			steps (int): Number of steps to predict.

		Returns:
			pl.Series: Extended time series data.
		"""

		time_series = time_series.interpolate(method="linear")
		return cast.BackCaster(self.period, self.steps).fit_transform(time_series)
