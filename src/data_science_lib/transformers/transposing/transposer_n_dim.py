import numpy as np
import sklearn as sk


class TransposerNDim(sk.base.BaseEstimator, sk.base.TransformerMixin):
	def __init__(self, transpose_dims: tuple[int, ...]):
		self.transpose_dims = transpose_dims

	def fit(self, X: np.ndarray, y=None) -> "TransposerNDim":  # type: ignore
		"""
		Only provided for sklearn compatibility.
		Args:
			X (np.ndarray): Does nothing, but is required for sklearn compatibility.
			y: Does nothing, but is required for sklearn compatibility.
		Returns:
			self: Fitted transposer.
		"""
		return self

	def transform(self, X: np.ndarray, y=None) -> np.ndarray:  # type: ignore
		return np.transpose(X, self.transpose_dims)  # type: ignore
