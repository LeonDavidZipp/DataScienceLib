import polars as pl
import polars.selectors as cs
import numpy as np
from typing import List
from functools import lru_cache
import logging

class Renamer:
	def __init__(self):
		self.counts = dict()

	def rename(self, name: str) -> str:
		name = self.rename_inner_(name)
		if self.counts.get(name) is None:
			self.counts[name] = 1
		else:
			self.counts[name] += 1
			name = f"{name}_{self.counts[name]}"
		return name

	@staticmethod
	@lru_cache(maxsize=1024)
	def rename_inner_(name: str) -> str:
		chars = {
			"!": "",
			"\"": "",
			"#": "number",
			"$": "dollar",
			"%": "percent",
			"&": "and",
			" ": "_",
			"'": "",
			"(": "",
			")": "",
			"*": "star",
			"+": "plus",
			",": "-",
			"-": "_",
			".": "_",
			"/": "_",
			":": "_",
			";": "_",
			"<": "lessthan",
			"=": "equals",
			">": "greaterthan",
			"?": "question",
			"@": "at",
			"[": "_",
			"\\": "_",
			"]": "_",
			"^": "caret",
			"`": "_",
			"{": "_",
			"|": "pipe",
			"}": "_",
			"~": "tilde"
		}

		name = name.translate(str.maketrans(chars))
		name = name.strip("_").lower()
		if name is None:
			return "_"
		else:
			return name

class Cleaner:
	"""
	class for basic data cleanup steps
	"""
	def __init__(self):
		pass

	def cols_from_provided_cols_(self, lf: pl.LazyFrame, cols: str | List[str]) -> list[str] | None:
		"""
		Filters the provided columns to only include those that are in the LazyFrame

		Args:
			lf (pl.LazyFrame): LazyFrame to check columns of
			cols (str | List[str]): columns to check for in the LazyFrame

		Returns:
			list[str] | None: list of columns that are in the LazyFrame, or None if no columns were provided
		"""
		try:
			if type(cols) is str:
				return list((lf.select(cs.string())).columns & [cols])
			elif type(cols) is List[str]:
				return list((lf.select(cs.string())).columns & cols)
			else:
				return None
		except Exception as e:
			raise ValueError(f"Error checking columns: {e}")

	def adjust_col_names(self, lf: pl.LazyFrame) -> pl.LazyFrame:
		"""
		renames all columns to camel case
		lf: LazyFrame to rename columns of
		:return: the modified LazyFrame
		"""

		try:
			rn = Renamer()
			return lf.select(pl.all().name.map(rn.rename))
		except Exception as e:
			raise ValueError(f"Error renaming columns: {e}")

	def remove_nulls(
		self,
		lf: pl.LazyFrame,
		cols_to_ignore: str | List[str] = [],
		non_null_threshold: float = 1
	) -> pl.LazyFrame | None:
		"""
		Removes rows that have too high a percentage of null values.

		Args:
			lf (pl.LazyFrame): LazyFrame to remove rows from
			cols_to_ignore (str | List[str]): columns to not consider when determining relative amount of null values
			non_null_threshold (float): percentage of columns that must be non-null in range [0, 1] to keep the row

		Returns:
			pl.LazyFrame | None: the modified LazyFrame, or None if no columns were provided
		"""

		if non_null_threshold < 0 or non_null_threshold > 1:
			raise ValueError("non_null_threshold must be in range [0, 1]")

		try:
			cols = self.cols_from_provided_cols_(lf, cols_to_ignore)
			if cols is None:
				return lf

			if len(cols) <= 0:
				pl_cols = pl.all()
			else:
				pl_cols = pl.exclude(cols)

			return (
				lf.filter(
					(pl.sum_horizontal(pl_cols.is_not_null()) / max(1, lf.width - len(cols)))
					>= non_null_threshold
				)
			)
		except Exception as e:
			raise ValueError(f"Error removing nulls: {e}")

	def cast_to_categorical(self, lf: pl.LazyFrame, cols: str | List[str]) -> pl.LazyFrame:
		"""
		Turns all possible string columns into categoricals.

		Args:
			lf (pl.LazyFrame): LazyFrame to modify
			cols (str | List[str]): columns to cast to categorical

		Returns:
			pl.LazyFrame: the modified LazyFrame
		"""

		try:
			cols = self.cols_from_provided_cols_(lf, cols)
			if cols is None:
				return lf

			return lf.with_columns(pl.col(cols).cast(pl.Categorical))
		except Exception as e:
			raise ValueError(f"Error converting columns to categorical: {e}")

	def cols_to_dummies(self, lf: pl.LazyFrame, cols: str | List[str]) -> pl.LazyFrame:
		"""
		Turns all possible string columns into dummies.
		Args:
			lf (pl.LazyFrame): LazyFrame to modify
			cols (str | List[str]): columns to dummify

		Returns:
			pl.LazyFrame: the modified LazyFrame
		"""

		try:
			cols = self.cols_from_provided_cols_(lf, cols)
			if cols is None:
				return lf

			return lf.collect().to_dummies(cols).lazy()
		except Exception as e:
			raise ValueError(f"Error converting columns to dummies: {e}")
		
	def remove_duplicates(self, lf: pl.LazyFrame, cols_to_ignore: str | List[str] = []) -> pl.LazyFrame:
		"""
		Removes duplicates from the LazyFrame.

		Args:
			lf (pl.LazyFrame): LazyFrame to remove duplicates from
			cols (str | List[str]): columns to check for duplicates

		Returns:
			pl.LazyFrame: the modified LazyFrame
		"""

		try:
			cols = self.cols_from_provided_cols_(lf, cols_to_ignore)
			if cols is None:
				return lf
			
			if len(cols) <= 0:
				pl_cols = pl.all()
			else:
				pl_cols = pl.exclude(cols)
			return lf.unique(subset=pl_cols)
		except Exception as e:
			raise ValueError(f"Error removing duplicates: {e}")
