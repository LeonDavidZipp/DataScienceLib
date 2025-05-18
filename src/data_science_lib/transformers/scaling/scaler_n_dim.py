import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from numpy.typing import NDArray
from typing import Any


class ScalerNDim(BaseEstimator, TransformerMixin):
	def __init__(
		self,
		divisor: int | float | None = None,
		prevent_div_by_zero_val: int | float = 0.01,
	):
		if divisor is not None and divisor == 0:
			raise ValueError("divisor cannot be 0")
		self.divisor = divisor
		self.prevent_div_by_zero_val = prevent_div_by_zero_val

	def fit(self, X: NDArray[Any], y=None) -> "ScalerNDim":  # type: ignore
		"""
		Fit the scaler to the data.

		Args:
			X (np.ndarray): Input data.
			y: Does nothing, but is required for sklearn compatibility.

		Returns:
			self: Fitted scaler.
		"""

		if self.divisor is None:
			max_val = X.max()
			if max_val == 0 or max_val == 0.0:
				max_val += self.prevent_div_by_zero_val
			self.divisor = max_val
		return self

	def transform(self, X: np.ndarray, y=None) -> NDArray[Any]:  # type: ignore
		"""
		Transform the data using the fitted scaler.

		Args:
			X (np.ndarray): Input data.
			y: Does nothing, but is required for sklearn compatibility.

		Returns:
			np.ndarray: Transformed data.
		"""

		return X / self.divisor

	def fit_transform(self, X: NDArray[Any], y=None) -> NDArray[Any]:  # type: ignore
		"""
		Fit the scaler to the data and then transform it.
		Args:
			X (np.ndarray): Input data.
			y: Does nothing, but is required for sklearn compatibility.
		Returns:
			np.ndarray: Transformed data.
		"""

		self.fit(X)  # type: ignore
		return self.transform(X)  # type: ignore
