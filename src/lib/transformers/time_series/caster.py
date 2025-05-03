import numpy as np
from typing import Literal, override
import statsmodels
import abc
import sklearn.base as skbase


class BaseCaster(abc.ABC, skbase.TransformerMixin):
	def __init__(
		self,
		period: Literal["daily", "monthly", "yearly"],
		steps: int = 24,
	):
		match period:
			case "daily":
				self.period = 365
			case "monthly":
				self.period = 12
			case "yearly":
				self.period = 1
		self.steps = steps
		self.trends = None
		self.means = None

	def fit(self, X: np.ndarray, y=None) -> "BaseCaster":
		"""
		Removes outliers based on Z-score. NOT meant to be part of a pipeline, but applied to the data beforehand.

		Args:
		        X (np.ndarray): (multiple) time series data.
		        y: Does nothing, but is required for sklearn compatibility.

		Returns:
		        None
		"""
		self.trends = np.zeros(1, X.shape[0])
		self.means = np.zeros(X.shape[0], self.period)
		for i in range(X.shape[0]):
			stl_res = statsmodels.tsa.seasonal.STL(
				endog=X, period=self.period, robust=True
			).fit()
			self.trends[i] = (
				stl_res.trend[0] - stl_res.trend[-1]
			) / stl_res.trend.shape[0]
			self.means[i] = np.array(
				[stl_res.seasonal[j :: j + 1].mean() for j in range(self.period)]
			)

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

	def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
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
		period: Literal["daily", "monthly", "yearly"],
		steps: int = 24,
	):
		super().__init__(period, steps)

	@override
	def transform(self, X: np.ndarray) -> np.ndarray:
		"""
		Transforms the input data.

		Args:
		        X (np.ndarray): (multiple) time series data.

		Returns:
		        np.ndarray: Transformed data.
		"""

		X_ext = np.array(
			[i * self.trends + self.means[i % self.period] for i in range(self.steps)]
		)
		return np.concat([X, X_ext], axis=0)


class BackCaster(BaseCaster):
	def __init__(
		self,
		period: Literal["daily", "monthly", "yearly"],
		steps: int = 24,
	):
		super().__init__(period, steps)

	@override
	def transform(self, X: np.ndarray) -> np.ndarray:
		"""
		Transforms the input data.

		Args:
		        X (np.ndarray): (multiple) time series data.

		Returns:
		        np.ndarray: Transformed data.
		"""

		X_ext = np.array(
			[
				(self.steps - i) * self.trends + self.means[i % self.period]
				for i in range(self.steps)
			]
		)
		return np.concat([X, X_ext], axis=0)
