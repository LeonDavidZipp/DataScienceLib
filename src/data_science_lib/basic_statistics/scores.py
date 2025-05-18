import numpy as np
import math


class Scores:
	@staticmethod
	def variance(X: np.ndarray) -> float:
		mean = X.mean()
		n = X.shape[0]
		return np.sum((X - mean) ** 2) / n

	@staticmethod
	def std(X: np.ndarray):
		return np.sqrt(Scores.variance(X))

	@staticmethod
	def mad(X: np.ndarray):
		mean = X.mean()
		n = X.shape[0]
		return np.sum(np.abs(X - mean)) / n

	@staticmethod
	def zscore(X: np.ndarray):
		std = Scores.std(X)
		mean = X.mean()
		return (X - mean) / std


class Poisson:
	def __init__(self, lambd: float):
		self.lambd = lambd

	def prob(self, x):
		return (self.lambd**x) * (math.e**-self.lambd) / math.factorial(x)
