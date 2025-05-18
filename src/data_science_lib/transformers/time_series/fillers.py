import polars as pl
import polars.selectors as cs
from typing import Literal
from .extrapolator import Extrapolator


class MultiTimeSeriesGapFiller:
	"""
	Class for filling missing values in time series data.
	"""

	def __init__(
		self,
		binary_value: bytes = b"0",
		boolean_value: bool = False,
		duration_strategy: Literal[
			"forward", "backward", "min", "max", "mean", "zero", "one"
		] = "mean",
		string_strategy: Literal[
			"forward_then_backward", "backward_then_forward"
		] = "backward_then_forward",
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

		if not isinstance(binary_value, bytes):
			raise TypeError(
				f"binary_value must be of type bytes. Got {type(binary_value)} instead."
			)
		if not duration_strategy in [
			"forward",
			"backward",
			"min",
			"max",
			"mean",
			"zero",
			"one",
		]:
			raise ValueError(
				"duration_strategy must be one of Literal['forward', 'backward', 'min', 'max', 'mean', 'zero', 'one']."
			)
		if not string_strategy in [
			"forward_then_backward",
			"backward_then_forward",
		]:
			raise ValueError(
				"string_strategy must be one of Literal['forward_then_backward', 'backward_then_forward']."
			)
		if not time_strategy in [
			"forward",
			"backward",
			"min",
			"max",
			"mean",
			"zero",
			"one",
		]:
			raise ValueError(
				"time_strategy must be one of Literal['forward', 'backward', 'min', 'max', 'mean', 'zero', 'one']."
			)

		self.binary_value = binary_value
		self.boolean_value = boolean_value
		self.duration_strategy = duration_strategy
		self.string_strategy = string_strategy
		self.time_strategy = time_strategy

	def fill(self, df: pl.DataFrame, date_col: str) -> pl.DataFrame:
		cols = df.columns

		df = (
			df.sort(by=[date_col])
			.upsample(time_column=date_col, every="1mo")
			.select(cols)
		)

		ext = Extrapolator(
			interpolation_method="linear",
			fill_value_if_only_null_values=0,
			val_before_first_non_null_val=0,
			val_after_last_non_null_val=None,
		)

		df = df.with_columns(
			cs.by_name(date_col),
			cs.numeric().map_batches(ext.fill_timeseries),
			cs.binary().fill_null(value=self.binary_value),
			cs.boolean().fill_null(value=self.boolean_value),
			cs.duration().fill_null(strategy=self.duration_strategy),  # type: ignore
			cs.time().fill_null(strategy=self.time_strategy),  # type: ignore
		)

		match self.string_strategy:
			case "forward_then_backward":
				df = df.with_columns(
					cs.duration()
					.fill_null(strategy="forward")
					.fill_null(strategy="backward")
				)
			case "backward_then_forward":
				df = df.with_columns(
					cs.duration()
					.fill_null(strategy="backward")
					.fill_null(strategy="forward")
				)
			case _:
				raise ValueError(
					"Expected string_strategy to be of type Literal['forward_then_backward', 'backward_then_forward']."
				)

		return df
