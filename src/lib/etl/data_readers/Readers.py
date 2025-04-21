import polars as pl
import os
import csv
from abc import ABC, abstractmethod
from typing import List, Tuple
from src.lib.helpers.Renamer import Renamer


class Reader(ABC):
	def __init__(self, path: str, schema: List[Tuple[str, str]] = None):
		"""
		Initializes the Reader class with the given path.

		Args:
		        path (str): The path to the data file.
		        schema (List[Tuple[str, str]], optional): The schema to enforce if any. If not provided, the schema will be inferred from the file.
		"""

		rn = Renamer()
		names = [rn.rename(str(name)) for name, _ in schema] if schema else None
		dtypes = [dtype for _, dtype in schema] if schema else None

		self.schema = (
			[(name, dtype) for name, dtype in zip(names, dtypes)]
			if names and dtypes
			else None
		)
		self.path = path

	def read(self) -> pl.LazyFrame:
		"""
		Reads data from the given path and returns it as a list of dictionaries.

		Args:
		        path (str): The path to the data file.

		Returns:
		        pl.LazyFrame: A lazy frame containing the data read from the file.
		"""

		if not os.path.exists(self.path):
			raise FileNotFoundError(f"File not found: {self.path}")

		try:
			return self.read_(self.path, self.schema)
		except Exception as e:
			raise RuntimeError(f"Failed to read file: {e}") from e

	@abstractmethod
	def read_() -> pl.LazyFrame:
		"""
		Reads data from the given path and returns it as a list of dictionaries.

		Args:
		        path (str): The path to the data file.

		Returns:
		        pl.LazyFrame: A lazy frame containing the data read from the file.
		"""

		raise NotImplementedError("Subclasses must implement this method")


class CSVReader(Reader):
	def __init__(self, path: str, schema: List[Tuple[str, str]] = None):
		"""
		Initializes the CSVReader class with the given path.
		Args:
		        path (str): The path to the CSV file.
		        schema (List[Tuple[str, str]], optional): The schema to enforce if any. If not provided, the schema will be inferred from the file.
		"""
		super().__init__(path, schema)

	def read_(self) -> pl.LazyFrame:
		"""
		Reads data from a CSV file and returns it as a lazy frame.

		Args:
		        path (str): The path to the CSV file.

		Returns:
		        pl.LazyFrame: A lazy frame containing the data read from the CSV file.
		"""
		try:
			# Detect delimiter and other dialect attributes
			with open(self.path, "r") as file:
				for s in [1024, 2048, 4096]:
					sample = file.read(s)
					sniffer = csv.Sniffer()
					res = sniffer.sniff(sample)
					if res is not None:
						break
				else:
					raise ValueError("Could not detect CSV dialect")

			# Extract dialect attributes
			delim = res.delimiter
			quote_char = res.quotechar or '"'

			# Read the CSV file dynamically using detected attributes
			return pl.scan_csv(
				source=self.path,
				has_header=True,  # Assume the file has a header
				separator=delim,  # Use the detected delimiter
				quote_char=quote_char,  # Use the detected quote character
				null_values=None,  # Handle null values dynamically
				encoding="utf8",  # Default encoding
			)
		except Exception as e:
			raise RuntimeError(f"Failed to read unknown CSV format: {e}") from e
