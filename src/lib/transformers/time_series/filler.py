import polars as pl
import polars.selectors as cs
from typing import Literal


class TimeSeriesGapFiller:
	"""
	Class for filling missing values in time series data.
	"""

	def __init__(
		self,
		binary_value: Literal[0, 1] = 0,
		boolean_value: Literal[0, 1] = 0,
		duration_strategy: Literal[
			"forward", "backward", "min", "max", "mean", "zero", "one"
		] = "mean",
		string_value: str = "missing",
		time_strategy: Literal[
			"forward", "backward", "min", "max", "mean", "zero", "one"
		] = "mean",
	):
		"""
		Initializes the TimeSeriesGapFiller with specified strategies for filling missing values.

		Args:
			binary_value (int): Value to fill for binary columns.
			boolean_value (int): Value to fill for boolean columns.
			duration_strategy (str): Strategy for filling duration columns.
			string_value (str): Strategy for filling string columns.
			time_strategy (str): Strategy for filling time columns.
		"""
		self.binary_value = binary_value
		self.boolean_value = boolean_value
		self.duration_strategy = duration_strategy
		self.string_value = string_value
		self.time_strategy = time_strategy

	def fill(self, X: pl.DataFrame, date_col: str) -> pl.DataFrame:
		cols = X.columns
		X = (
			X.upsample(time_column=date_col, every="1mo")
			.with_columns(
				cs.by_name(date_col),
				cs.numeric().interpolate(method="linear"),
				cs.binary().fill_null(value=self.binary_value),
				cs.boolean().fill_null(value=self.boolean_value),
				cs.duration().fill_null(strategy=self.duration_strategy),
				cs.string(include_categorical=True).fill_null(value="missing"),
				cs.time().fill_null(strategy=self.time_strategy),
			)
			.select(cols)
		)
