import numpy as np
import math

class Scores:
	@staticmethod
	def variance(X: np.ndarray):
		mean = X.mean()
		n = X.shape[0]
		return np.sum((X - mean) ** 2) / n

	@staticmethod
	def std(X: np.ndarray):
		return np.sqrt(Scores.variance(X))

	@staticmethod
	def zscore(X: np.ndarray):
		std = Scores.std(X)
		mean = X.mean()
		return (X - mean) / std