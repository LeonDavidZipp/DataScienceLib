import polars as pl
from abc import ABC, abstractmethod
from src.data_science_lib.etl.pipelines.BaseDataSource import BaseDataSource


class BaseETLPipeline(ABC):
	def __init__(self):
		super().__init__()

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
