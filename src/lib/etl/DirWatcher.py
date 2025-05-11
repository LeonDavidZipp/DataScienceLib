from src.lib.etl.data_readers.Readers import CSVReader
from src.lib.etl.cleaners.Cleaner import Cleaner
import watchdog.observers as obs
import watchdog.events as ev
import os

import polars as pl
import sqlalchemy


class BaseETLPipeline:
	def __init__(self):
		pass


class ETLHandler(ev.FileSystemEventHandler):
	def __init__(
		self,
		engine: sqlalchemy.engine.Engine,
		ending_to_schema_mapping: dict[str, str] | None = None,
		non_null_threshold: float | None = None,
		null_fill_strategy: str | None = None,
		descending: bool | None = None,
	):
		"""
		Args:
		        engine: the ALREADY CONNECTED engine to interact with the database
		        ending_to_schema__mapping: a dictionary mapping file endings to schema names
		        non_null_threshold: the threshold for non-null values
		        null_fill_strategy: the strategy for filling null values
		        descending: whether to sort in descending order
		"""
		self.engine = engine
		self.ending_to_schema_mapping = ending_to_schema_mapping
		self.non_null_threshold = non_null_threshold
		self.null_fill_strategy = null_fill_strategy
		self.descending = descending

	def extract(self, path: str) -> pl.LazyFrame:
		name, ext = os.path.splitext(path)
		if ext == ".csv":
			return CSVReader(path).read()

	def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
		lf = Cleaner.adjust_column_names(lf)
		lf = Cleaner.remove_nulls(lf, self.non_null_threshold)
		lf = Cleaner.remove_duplicates(lf)
		lf = Cleaner.fill_nulls(lf, self.null_fill_strategy)
		return Cleaner.sort(lf, self.descending)

	def load(
		self, lf: pl.LazyFrame, table_name: str, schema_name: str = "public"
	) -> None:
		"""
		Load the transformed data into the target system.
		"""
		if lf is None:
			return
		lf.collect().write_database(
			table_name=f"{schema_name}.{table_name}",
			connection=self.engine,
			engine="sqlalchemy",
			if_table_exists="append",
		)

	def on_created(self, event: ev.DirCreatedEvent | ev.FileCreatedEvent):
		if event.is_directory:
			return
		lf = self.extract(event.src_path)
		lf = self.transform(lf)


class DirWatcher:
	def __init__(self, path: str):
		self.path = path
		self.observer = obs.Observer()
		handler = ETLHandler()
		self.observer.schedule(
			event_handler=handler,
			path=path,
			recursive=True,
			event_filter=[ev.FileCreatedEvent],
		)

	# def run
