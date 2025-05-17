import polars as pl
import polars.selectors as cs
from ...helpers.renamer import Renamer
from abc import abstractmethod


class Cleaner:
	"""
	Class for basic data cleanup steps
	"""

	def __init__(self):
		pass

	@staticmethod
	def cols_from_provided_cols_(
		lf: pl.LazyFrame, cols: str | list[str] | None
	) -> list[str]:
		"""
		Helper method filtering the provided columns to only include those that are in the LazyFrame

		Args:
		        lf (pl.LazyFrame): LazyFrame to check columns of
		        cols (str | list[str]): columns to check for in the LazyFrame

		Returns:
		        list[str] | None: list of columns that are in the LazyFrame, or None if no columns were provided
		"""

		try:
			match cols:
				case str():
					lf_cols = lf.collect_schema().names()
					return [col for col in lf_cols if col == cols]
				case list():
					lf_cols = lf.collect_schema().names()
					return [col for col in cols if col in lf_cols]
				case None:
					return []
				case _:
					raise ValueError("cols must be one of str, list[str], or None")
		except Exception as e:
			raise ValueError(f"Error checking columns: {e}")

	@staticmethod
	def format_column_names(lf: pl.LazyFrame) -> pl.LazyFrame:
		"""
		Adjusts the column names to be valid python variable names.

		Args:
		        lf (pl.LazyFrame): LazyFrame to adjust column names of

		Returns:
		        pl.LazyFrame: the modified LazyFrame
		"""

		try:
			rn = Renamer()
			return lf.select(pl.all().name.map(rn.rename))
		except Exception as e:
			raise ValueError(f"Error renaming columns: {e}")

	@staticmethod
	def rename_columns(
		lf: pl.LazyFrame, names_to_new_names: dict[str, str]
	) -> pl.LazyFrame:
		"""
		Renames the columns in the LazyFrame.

		Args:
		        lf (pl.LazyFrame): LazyFrame to rename columns of
		        names_to_new_names (dict[str, str]): dictionary mapping old column names to new column names

		Returns:
		        pl.LazyFrame: the modified LazyFrame
		"""

		try:
			return lf.rename(names_to_new_names, strict=False)
		except Exception as e:
			raise ValueError(f"Error renaming columns: {e}")

	@staticmethod
	def reorder_columns(
		lf: pl.LazyFrame, column_names: list[str] | None = None
	) -> pl.LazyFrame:
		"""
		Reorders the columns in the LazyFrame.

		Args:
		        lf (pl.LazyFrame): LazyFrame to reorder columns of
		        column_names (list[str]): list of new column names

		Returns:
		        pl.LazyFrame: the modified LazyFrame
		"""

		if column_names is None:
			return lf

		try:
			return lf.select(
				cs.by_name(column_names),
				~cs.by_name(column_names),
			)
		except Exception as e:
			raise ValueError(f"Error reordering columns: {e}")

	@staticmethod
	def drop(lf: pl.LazyFrame, cols: str | list[str] | None = None) -> pl.LazyFrame:
		"""
		Drops the given columns from the LazyFrame.

		Args:
		        lf (pl.LazyFrame): LazyFrame to drop columns from
		        cols (str | list[str]): columns to drop

		Returns:
		        pl.LazyFrame: the modified LazyFrame
		"""

		if cols is None:
			return lf

		try:
			cols = Cleaner.cols_from_provided_cols_(lf, cols)
			if not cols:
				return lf

			if len(cols) <= 0:
				pl_cols = pl.all()
			else:
				pl_cols = pl.exclude(cols)

			return lf.select(pl_cols)
		except Exception as e:
			raise ValueError(f"Error dropping columns: {e}")

	@staticmethod
	def remove_nulls(
		lf: pl.LazyFrame,
		cols_to_ignore: str | list[str] | None = None,
		non_null_threshold: float | None = None,
	) -> pl.LazyFrame | None:
		"""
		Removes rows that have too high a percentage of null values.

		Args:
		        lf (pl.LazyFrame): LazyFrame to remove rows from
		        cols_to_ignore (str | list[str]): columns to not consider when determining relative amount of null values
		        non_null_threshold (float): percentage of columns that must be non-null in range [0, 1] to keep the row

		Returns:
		        pl.LazyFrame | None: the modified LazyFrame, or None if no columns were provided
		"""

		if non_null_threshold is None:
			return lf
		elif non_null_threshold < 0 or non_null_threshold > 1:
			raise ValueError("non_null_threshold must be in range [0, 1]")

		try:
			if cols_to_ignore is None:
				cols_to_ignore = []
			cols = Cleaner.cols_from_provided_cols_(lf, cols_to_ignore)
			if cols is None:
				return lf

			if len(cols) <= 0:
				pl_cols = pl.all()
			else:
				pl_cols = pl.exclude(cols)

			return lf.filter(
				(
					pl.sum_horizontal(pl_cols.is_not_null())
					/ max(1, lf.width - len(cols))
				)
				>= non_null_threshold
			)
		except Exception as e:
			raise ValueError(f"Error removing nulls: {e}")

	@staticmethod
	def remove_duplicates(lf: pl.LazyFrame) -> pl.LazyFrame:
		"""
		Removes duplicates from the LazyFrame.

		Args:
		        lf (pl.LazyFrame): LazyFrame to remove duplicates from
		        cols (str | list[str]): columns to check for duplicates

		Returns:
		        pl.LazyFrame: the modified LazyFrame
		"""

		try:
			return lf.unique()
		except Exception as e:
			raise ValueError(f"Error removing duplicates: {e}")

	@staticmethod
	def fill_nulls(
		lf: pl.LazyFrame,
		str_fill_val: str | None = "unknown",
		str_fill_strat: str | None = None,
		cat_fill_val: pl.Categorical | None = pl.lit("unknown", dtype=pl.Categorical),
		cat_fill_strat: str | None = None,
		num_fill_val: int | float | None = None,
		num_fill_strat: str | None = "mean",
		datetime_fill_val: pl.Date | pl.Datetime | None = None,
		datetime_fill_strat: str | None = "backward",
		time_fill_val: pl.Time | None = None,
		time_fill_strat: str | None = "mean",
		bool_fill_val: bool | None = False,
		bool_fill_strat: str | None = None,
	) -> pl.LazyFrame:
		# TODO: fix documentation
		"""
		Fills null values in the given columns with given strategies.

		Args:
		        lf (pl.LazyFrame): LazyFrame to fill nulls in
		        cols (str | list[str]): columns to fill nulls in
		        fill_strategy (str): strategy to use to fill nulls. Options are 'mean', 'median', 'mode', 'zero', 'ffill', 'bfill', or a constant value

		Returns:
		        pl.LazyFrame: the modified LazyFrame
		"""

		if str_fill_val is not None and str_fill_strat is not None:
			raise ValueError(
				f"Only str_fill_val or str_fill_strat can be not None, instead got {str_fill_val} and {str_fill_strat}"
			)
		elif num_fill_val is not None and num_fill_strat is not None:
			raise ValueError(
				f"Only num_fill_val or num_fill_strat can be not None, instead got {num_fill_val} and {num_fill_strat}"
			)
		elif datetime_fill_val is not None and datetime_fill_strat is not None:
			raise ValueError(
				f"Only date_fill_val or date_fill_strat can be not None, instead got {datetime_fill_val} and {datetime_fill_strat}"
			)
		elif time_fill_val is not None and time_fill_strat is not None:
			raise ValueError(
				f"Only time_fill_val or time_fill_strat can be not None, instead got {time_fill_val} and {time_fill_strat}"
			)
		elif bool_fill_val is not None and bool_fill_strat is not None:
			raise ValueError(
				f"Only bool_fill_val or bool_fill_strat can be not None, instead got {bool_fill_val} and {str_fill_strat}"
			)
		elif cat_fill_val is not None and cat_fill_strat is not None:
			raise ValueError(
				f"Only cat_fill_val or cat_fill_strat can be not None, instead got {cat_fill_val} and {str_fill_strat}"
			)

		try:
			return lf.with_columns(
				[
					cs.string().fill_null(value=str_fill_val, strategy=str_fill_strat),
					cs.numeric().fill_null(value=num_fill_val, strategy=num_fill_strat),
					cs.boolean().fill_null(
						value=bool_fill_val, strategy=bool_fill_strat
					),
					cs.date().fill_null(
						value=datetime_fill_val, strategy=datetime_fill_strat
					),
					cs.categorical().fill_null(
						value=cat_fill_val, strategy=cat_fill_strat
					),
					cs.time().fill_null(value=time_fill_val, strategy=time_fill_strat),
				]
			)
		except Exception as e:
			raise ValueError(f"Error filling nulls: {e}")

	@staticmethod
	@abstractmethod
	def custom_fill_nulls(
		lf: pl.LazyFrame,
		cols: str | list[str] | None = None,
		fill_strategy: str | None = None,
	) -> pl.LazyFrame:
		"""
		Fills null values in the given columns with custom strategies.

		Args:
		        lf (pl.LazyFrame): LazyFrame to fill nulls in
		        cols (str | list[str]): columns to fill nulls in
		        fill_strategy (str): strategy to use to fill nulls. Options are 'mean', 'median', 'mode', 'zero', 'ffill', 'bfill', or a constant value

		Returns:
		        pl.LazyFrame: the modified LazyFrame
		"""

		raise NotImplementedError(
			"custom_fill_nulls must be implemented by the child class"
		)

	@staticmethod
	def sort(
		lf: pl.LazyFrame, cols: str | list[str] | None = None, descending: bool = False
	) -> pl.LazyFrame:
		"""
		Sorts the LazyFrame by the given columns.

		Args:
		        lf (pl.LazyFrame): LazyFrame to sort
		        cols (str | list[str] | None): columns to sort by; if None but descending is not None, will sort by all columns
		        descending (bool): whether to sort in descending order; if None and cols is None, will not sort

		Returns:
		        pl.LazyFrame: the modified LazyFrame
		"""

		if cols is None and descending is None:
			return lf

		try:
			cols = Cleaner.cols_from_provided_cols_(lf, cols)
			if len(cols) <= 0:
				cols = lf.collect_schema().names()

			return lf.sort(by=cols, descending=descending)
		except Exception as e:
			raise ValueError(f"Error sorting columns: {e}")
