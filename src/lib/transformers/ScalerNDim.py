import numpy as np
import sklearn as sk


class ScalerNDim(sk.base.BaseEstimator, sk.base.TransformerMixin):
	def __init__(
		self,
		divisor: int | float | None = None,
		prevent_div_by_zero_val: int | float = 0.01,
	):
		if divisor is not None and divisor == 0:
			raise ValueError("divisor cannot be 0")
		self.divisor = divisor
		self.prevent_div_by_zero_val = prevent_div_by_zero_val

	def fit(self, X: np.ndarray, y=None):
		if self.divisor is None:
			max_val = X.max()
			if max_val == 0:
				max_val += self.prevent_div_by_zero_val
			self.divisor = max_val
		return self

	def transform(self, X: np.ndarray, y=None):
		return X / self.divisor
