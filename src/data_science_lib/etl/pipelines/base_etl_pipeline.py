import polars as pl
from abc import ABC, abstractmethod
from .base_data_source import BaseDataSource


class BaseETLPipeline(ABC):
	def __init__(self):
		super().__init__()

	def extract(self, src: BaseDataSource) -> pl.LazyFrame:
		return self.extract_(src)

	@abstractmethod
	def extract_(self, src: BaseDataSource) -> pl.LazyFrame:
		raise NotImplementedError("Subclasses must implement the extract_ method")

	def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
		return self.transform_(lf)

	@abstractmethod
	def transform_(self, lf: pl.LazyFrame) -> pl.LazyFrame:
		raise NotImplementedError("Subclasses must implement the transform_ method")

	def load(self, lf: pl.LazyFrame) -> bool:
		return self.load_(lf)

	@abstractmethod
	def load_(self, lf: pl.LazyFrame) -> bool:
		raise NotImplementedError("Subclasses must implement the load_ method")

	def run(self, src: str) -> bool:
		lf = self.extract(src)
		lf = self.transform(lf)
		return self.load(lf)
