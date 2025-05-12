from typing import Literal
import numpy as np
from typing import override
import statsmodels
import abc
import sklearn.base as skbase

import statsmodels.tsa
import statsmodels.tsa.seasonal


class BaseCaster(abc.ABC, skbase.TransformerMixin):
	def __init__(
		self,
		period: Literal["daily", "monthly", "yearly"] = "monthly",
		steps: int = 24,
		only_return_extended: bool = False,
	):
		match period:
			case "daily":
				self.period = 365
			case "monthly":
				self.period = 12
			case "yearly":
				self.period = 1
		self.steps = steps
		self.only_return_extended = only_return_extended
		self.is_fitted = False
		self.trends = None
		self.cycles = None

	def fit(self, X: np.ndarray, y=None) -> "BaseCaster":
		"""
		Removes outliers based on Z-score. NOT meant to be part of a pipeline, but applied to the data beforehand.

		Args:
		        X (np.ndarray): (multiple) time series data.
		        y: Does nothing, but is required for sklearn compatibility.

		Returns:
		        BaseCaster
		"""

		if X.ndim == 1:
			X = X.reshape(-1, 1)

		self.trends = np.zeros((X.shape[1],))
		self.cycles = np.zeros((X.shape[1], self.period))
		for i in range(X.shape[1]):
			stl_res = statsmodels.tsa.seasonal.STL(
				endog=X, period=self.period, robust=True
			).fit()
			self.trends[i] = (
				stl_res.trend[0] - stl_res.trend[-1]
			) / stl_res.trend.shape[0]
			self.cycles[i] = np.array(
				[stl_res.seasonal[j :: j + 1].mean() for j in range(self.period)]
			)

		self.is_fitted = True
		return self

	@abc.abstractmethod
	def transform(self, X: np.ndarray) -> np.ndarray:
		"""
		Transforms the input data.

		Args:
		        X (np.ndarray): (multiple) time series data.

		Returns:
		        np.ndarray: Transformed data.
		"""

		raise NotImplementedError("Subclasses should implement this method")

	@override
	def fit_transform(self, X: np.ndarray, y=None, **fit_params) -> np.ndarray:  # type: ignore
		"""
		Fits the model to the data and then transforms it.

		Args:
		        X (np.ndarray): (multiple) time series data.
		        y: Does nothing, but is required for sklearn compatibility.

		Returns:
		        np.ndarray: Transformed data.
		"""

		self.fit(X)
		return self.transform(X)


class ForeCaster(BaseCaster):
	def __init__(
		self,
		period: Literal["daily", "monthly", "yearly"] = "monthly",
		steps: int = 24,
		only_return_extended: bool = False,
	):
		super().__init__(period, steps, only_return_extended)

	@override
	def transform(self, X: np.ndarray) -> np.ndarray:
		"""
		Transforms the input data.

		Args:
		        X (np.ndarray): (multiple) time series data.

		Returns:
		        np.ndarray: Transformed data.
		"""

		if not self.is_fitted or self.trends is None or self.cycles is None:
			raise ValueError(
				"ForeCaster is not fitted yet. Please call fit() before calling transform()."
			)

		if X.ndim == 1:
			X = X.reshape(-1, 1)

		X_ext = np.array(
			[
				[
					i * self.trends[j] + self.cycles[j, i % self.period]
					for j in range(X.shape[1])
				]
				for i in range(self.steps)
			]
		)

		if self.only_return_extended:
			X = X_ext
		else:
			X = np.concat([X, X_ext], axis=0)

		if X.ndim == 2 and X.shape[1] == 1:
			return X.reshape(
				-1,
			)
		return X


class BackCaster(BaseCaster):
	def __init__(
		self,
		period: Literal["daily", "monthly", "yearly"] = "monthly",
		steps: int = 24,
		only_return_extended: bool = False,
	):
		super().__init__(period, steps, only_return_extended)

	@override
	def transform(self, X: np.ndarray) -> np.ndarray:
		"""
		Transforms the input data.

		Args:
		        X (np.ndarray): (multiple) time series data.

		Returns:
		        np.ndarray: Transformed data.
		"""

		if not self.is_fitted or self.trends is None or self.cycles is None:
			raise ValueError(
				"BackCaster is not fitted yet. Please call fit() before calling transform()."
			)

		if X.ndim == 1:
			X = X.reshape(-1, 1)

		# single series:
		X_ext = np.array(
			[
				[
					(self.steps - i) * self.trends[j] + self.cycles[j, i % self.period]
					for j in range(X.shape[1])
				]
				for i in range(self.steps)
			]
		)

		if self.only_return_extended:
			X = X_ext
		else:
			X = np.concat([X_ext, X], axis=0)

		if X.ndim == 2 and X.shape[1] == 1:
			return X.reshape(
				-1,
			)
		return X
