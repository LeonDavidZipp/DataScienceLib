import polars as pl
from abc import ABC, abstractmethod
from etl.Cleaner import Cleaner

class BaseDataSource(ABC):
	def __init__(self):
		pass

class BaseETLPipeline(ABC):
	def __init__(self):
		pass

	def extract(self, src: BaseDataSource) -> pl.LazyFrame:
		return self.extract_(src)

	@abstractmethod
	def extract_(self, src: BaseDataSource) -> pl.LazyFrame:
		raise NotImplementedError("Subclasses must implement the extract_ method")

	def transform(self, lf: pl.LazyFrame):
		return self.transform_(lf)

	@abstractmethod
	def transform_(self, lf: pl.LazyFrame):
		raise NotImplementedError("Subclasses must implement the transform_ method")

	def load(self, lf: pl.LazyFrame):
		self.load_(lf)

	@abstractmethod
	def load_(self, lf: pl.LazyFrame):
		raise NotImplementedError("Subclasses must implement the load_ method")

	def run(self):
		lf = self.extract()
		lf = self.transform(lf)
		self.load(lf)

class CleanerPipeline(BaseETLPipeline):
	def __init__(
			self,
			non_null_threshold: float = 0.0,
			descending: bool = False,
			str_fill_val: str | None = "unknown",
			str_fill_strat: str | None = None,
			cat_fill_val: pl.Categorical | None = pl.lit("unknown", dtype=pl.Categorical),
			cat_fill_strat: str | None = None,
			num_fill_val: int | float | None = None,
			num_fill_strat: str | None = "mean",
			datetime_fill_val: pl.Date | pl.Datetime | None = None ,
			datetime_fill_strat: str | None = "backward",
			time_fill_val: pl.Time | None = None,
			time_fill_strat: str | None = "mean",
			bool_fill_val: bool | None = False,
			bool_fill_strat: str | None = None,
		):
		"""
		Args:
			non_null_threshold (float): The threshold for non-null values to keep a column.
			descending (bool): Whether to sort the DataFrame in descending order.
			str_fill_val (str | None): The value to fill for string columns.
			str_fill_strat (str | None): The strategy for filling string columns.
			cat_fill_val (pl.Categorical | None): The value to fill for categorical columns.
			cat_fill_strat (str | None): The strategy for filling categorical columns.
			num_fill_val (int | float | None): The value to fill for numeric columns.
			num_fill_strat (str | None): The strategy for filling numeric columns.
			datetime_fill_val (pl.Date | pl.Datetime | None): The value to fill for datetime columns.
			datetime_fill_strat (str | None): The strategy for filling datetime columns.
			time_fill_val (pl.Time | None): The value to fill for time columns.
			time_fill_strat (str | None): The strategy for filling time columns.
			bool_fill_val (bool | None): The value to fill for boolean columns.
			bool_fill_strat (str | None): The strategy for filling boolean columns.
		"""

		super().__init__()
		self.non_null_threshold = non_null_threshold
		self.descending = descending
		self.str_fill_val = str_fill_val
		self.str_fill_strat = str_fill_strat
		self.cat_fill_val = cat_fill_val
		self.cat_fill_strat = cat_fill_strat
		self.num_fill_val = num_fill_val
		self.num_fill_strat = num_fill_strat
		self.datetime_fill_val = datetime_fill_val
		self.datetime_fill_strat = datetime_fill_strat
		self.time_fill_val = time_fill_val
		self.time_fill_strat = time_fill_strat
		self.bool_fill_val = bool_fill_val
		self.bool_fill_strat = bool_fill_strat

	def extract_(self, src: BaseDataSource):
		return super().extract_(src)

	def transform_(self, lf: pl.LazyFrame):
		lf = Cleaner.adjust_column_names(lf)
		lf = Cleaner.remove_nulls(lf, self.non_null_threshold)
		lf = Cleaner.remove_duplicates(lf)
		lf = Cleaner.fill_nulls(
			lf,
			self.str_fill_val,
			self.str_fill_strat,
			self.cat_fill_val,
			self.cat_fill_strat,
			self.num_fill_val,
			self.num_fill_strat,
			self.datetime_fill_val,
			self.datetime_fill_strat,
			self.time_fill_val,
			self.time_fill_strat,
			self.bool_fill_val,
			self.bool_fill_strat,
		)
		return Cleaner.sort(lf, self.descending)
	
	def load_(self, lf: pl.LazyFrame):
		return super().load_(lf)