import numpy as np
import sklearn as sk

class TransposerNDim(sk.base.BaseEstimator, sk.base.TransformerMixin):
	def __init__(self, transpose_dims: tuple):
		self.transpose_dims = transpose_dims

	def fit(self, X: np.ndarray, y=None):
		return self

	def transform(self, X: np.ndarray, y=None):
		return np.transpose(X, self.transpose_dims)
