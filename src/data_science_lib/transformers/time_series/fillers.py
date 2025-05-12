import polars as pl
import polars.selectors as cs
from typing import Literal


class MultiTimeSeriesGapFiller:
	"""
	Class for filling missing values in time series data.
	"""

	def __init__(
		self,
		binary_value: Literal[0, 1] = 0,
		boolean_value: Literal[0, 1] = 0,
		duration_strategy: Literal[
			"forward", "backward", "min", "max", "mean", "zero", "one"
		] = "forward",
		string_strategy: Literal["forward", "backward", "zero", "one"] = "forward",
		time_strategy: Literal[
			"forward", "backward", "min", "max", "mean", "zero", "one"
		] = "mean",
	):
		"""
		Initializes the MultiTimeSeriesGapFiller with specified strategies for filling missing values.

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
		self.string_strategy = string_strategy
		self.time_strategy = time_strategy

	def fill(self, X: pl.DataFrame, date_col: str) -> pl.DataFrame:
		cols = X.columns
		return (
			X.sort(by=[date_col])
			.upsample(time_column=date_col, every="1mo")
			.with_columns(
				cs.by_name(date_col),
				cs.numeric().interpolate(method="linear"),
				cs.binary().fill_null(value=self.binary_value),
				cs.boolean().fill_null(value=self.boolean_value),
				cs.duration().fill_null(strategy=self.duration_strategy),  # type: ignore
				cs.string(include_categorical=True).fill_null(
					strategy=self.string_strategy  # type: ignore
				),
				cs.time().fill_null(strategy=self.time_strategy),  # type: ignore
			)
			.select(cols)
		)
