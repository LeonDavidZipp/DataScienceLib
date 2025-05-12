import unittest
import numpy as np
import polars as pl
from src.etl.extender_filler import ForeCaster, BackCaster, NumericalTimeSeriesExtender


class TestForeCaster(unittest.TestCase):
	def setUp(self):
		self.data = np.random.randn(100)
		self.forecaster = ForeCaster(period="monthly", steps=24)

	def test_fit_transform(self):
		transformed_data = self.forecaster.fit_transform(self.data)
		self.assertEqual(transformed_data.shape[0], 124)


class TestBackCaster(unittest.TestCase):
	def setUp(self):
		self.data = np.random.randn(100)
		self.backcaster = BackCaster(period="monthly", steps=24)

	def test_fit_transform(self):
		transformed_data = self.backcaster.fit_transform(self.data)
		self.assertEqual(transformed_data.shape[0], 124)


class TestNumericalTimeSeriesExtender(unittest.TestCase):
	def setUp(self):
		self.data = pl.Series(np.random.randn(100))
		self.extender = NumericalTimeSeriesExtender(period="monthly", steps=24)

	def test_forward(self):
		extended_data = self.extender.forward(self.data)
		self.assertEqual(extended_data.shape[0], 124)

	def test_backward(self):
		extended_data = self.extender.backward(self.data)
		self.assertEqual(extended_data.shape[0], 124)


if __name__ == "__main__":
	unittest.main()
