import unittest
import numpy as np
from .scaler_n_dim import ScalerNDim


class TestScalerNDim(unittest.TestCase):
	def test_scaler_with_provided_divisor(self):
		X = np.array([[2, 4], [6, 8]])
		scaler = ScalerNDim(divisor=2)
		scaler.fit(X) # type: ignore
		transformed = scaler.transform(X) # type: ignore
		expected = np.array([[1, 2], [3, 4]])
		np.testing.assert_array_almost_equal(transformed, expected)

	def test_scaler_with_calculated_divisor(self):
		X = np.array([[2, 4], [6, 8]])
		scaler = ScalerNDim()
		scaler.fit(X) # type: ignore
		transformed = scaler.transform(X) # type: ignore
		expected = X / 8  # Max value in X is 8
		np.testing.assert_array_almost_equal(transformed, expected)

	def test_scaler_prevent_division_by_zero(self):
		X = np.array([[0, 0], [0, 0]])
		scaler = ScalerNDim()
		scaler.fit(X) # type: ignore
		transformed = scaler.transform(X) # type: ignore
		expected = X / 0.01  # prevent_div_by_zero_val is 0.01
		np.testing.assert_array_almost_equal(transformed, expected)

	def test_scaler_with_zero_divisor_raises_error(self):
		with self.assertRaises(ValueError) as context:
			ScalerNDim(divisor=0)
		self.assertEqual(str(context.exception), "divisor cannot be 0")

	def test_scaler_transform_without_fit(self):
		X = np.array([[2, 4], [6, 8]])
		scaler = ScalerNDim(divisor=2)
		transformed = scaler.transform(X) # type: ignore
		expected = np.array([[1, 2], [3, 4]])
		np.testing.assert_array_almost_equal(transformed, expected)

	def test_scaler_with_negative_values(self):
		X = np.array([[-2, -4], [6, 8]])
		scaler = ScalerNDim()
		scaler.fit(X) # type: ignore
		transformed = scaler.transform(X) # type: ignore
		expected = X / 8  # Max absolute value in X is 8
		np.testing.assert_array_almost_equal(transformed, expected)


if __name__ == "__main__":
	unittest.main()
