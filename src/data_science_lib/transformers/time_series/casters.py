from typing import Literal
import numpy as np
from typing import override, Any
from numpy.typing import NDArray
import statsmodels  # type: ignore
import abc
import sklearn.base as skbase


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

	def fit(self, X: NDArray[Any], y=None) -> "BaseCaster":  # type: ignore
		"""
		Removes outliers based on Z-score. NOT meant to be part of a pipeline, but applied to the data beforehand.

		Args:
		        X (NDArray[Any]): (multiple) time series data.
		        y: Does nothing, but is required for sklearn compatibility.

		Returns:
		        BaseCaster
		"""

		if X.ndim == 1:
			x_new = X.reshape(-1, 1)
		else:
			x_new = X

		self.trends = np.zeros((x_new.shape[1],))
		self.cycles = np.zeros((x_new.shape[1], self.period))
		for i in range(x_new.shape[1]):
			stl_res = statsmodels.tsa.seasonal.STL(  # type: ignore
				endog=x_new, period=self.period, robust=True
			).fit()
			self.trends[i] = (
				stl_res.trend[0] - stl_res.trend[-1]  # type: ignore
			) / stl_res.trend.shape[0]  # type: ignore
			self.cycles[i] = np.array(
				[stl_res.seasonal[j :: j + 1].mean() for j in range(self.period)]  # type: ignore
			)

		self.is_fitted = True
		return self

	@abc.abstractmethod
	def transform(self, X: NDArray[Any]) -> NDArray[Any]:
		"""
		Transforms the input data.

		Args:
		        X (NDArray[Any]): (multiple) time series data.

		Returns:
		        NDArray[Any]: Transformed data.
		"""

		raise NotImplementedError("Subclasses should implement this method")

	@override
	def fit_transform(self, X: NDArray[Any], y=None, **fit_params) -> NDArray[Any]:  # type: ignore
		"""
		Fits the model to the data and then transforms it.

		Args:
		        X (NDArray[Any]): (multiple) time series data.
		        y: Does nothing, but is required for sklearn compatibility.

		Returns:
		        NDArray[Any]: Transformed data.
		"""

		self.fit(X)  # type: ignore
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
	def transform(self, X: NDArray[Any]) -> NDArray[Any]:
		"""
		Transforms the input data.

		Args:
		        X (NDArray[Any]): (multiple) time series data.

		Returns:
		        NDArray[Any]: Transformed data.
		"""

		if not self.is_fitted or self.trends is None or self.cycles is None:
			raise ValueError(
				"ForeCaster is not fitted yet. Please call fit() before calling transform()."
			)

		if X.ndim == 1:
			x_new = X.reshape(-1, 1)
		else:
			x_new = X

		x_ext = np.array(
			[
				[
					i * self.trends[j] + self.cycles[j, i % self.period]
					for j in range(x_new.shape[1])
				]
				for i in range(self.steps)
			]
		)

		if self.only_return_extended:
			x_new = x_ext
		else:
			x_new = np.concat([x_new, x_ext], axis=0)

		if x_new.ndim == 2 and x_new.shape[1] == 1:
			return x_new.reshape(
				-1,
			)
		return x_new


class BackCaster(BaseCaster):
	def __init__(
		self,
		period: Literal["daily", "monthly", "yearly"] = "monthly",
		steps: int = 24,
		only_return_extended: bool = False,
	):
		super().__init__(period, steps, only_return_extended)

	@override
	def transform(self, X: NDArray[Any]) -> NDArray[Any]:
		"""
		Transforms the input data.

		Args:
		        X (NDArray[Any]): (multiple) time series data.

		Returns:
		        NDArray[Any]: Transformed data.
		"""

		if not self.is_fitted or self.trends is None or self.cycles is None:
			raise ValueError(
				"BackCaster is not fitted yet. Please call fit() before calling transform()."
			)

		if X.ndim == 1:
			x_new = X.reshape(-1, 1)
		else:
			x_new = X

		# single series:
		x_ext = np.array(
			[
				[
					(self.steps - i) * self.trends[j] + self.cycles[j, i % self.period]
					for j in range(x_new.shape[1])
				]
				for i in range(self.steps)
			]
		)

		if self.only_return_extended:
			x_new = x_ext
		else:
			x_new = np.concat([x_ext, x_new], axis=0)

		if x_new.ndim == 2 and x_new.shape[1] == 1:
			return x_new.reshape(
				-1,
			)
		return x_new
