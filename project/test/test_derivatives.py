import unittest
import numpy as np
from scipy.special import erf

import project.model.optimization_loglikelihood.derivatives as d


class TestDerivatives(unittest.TestCase):

    def test_expected_i(self):
        result = d.expected_i(coordinate=3, theta_i=6, pixel_size_i=4, sigma=np.sqrt(2))
        expected = 0.5 * (erf(-0.5) - erf(-2.5))
        self.assertAlmostEqual(result, expected, places=8)

    def test_expected_count(self):
        result = d.expected_count(coordinates_x=2, coordinates_y=3, theta=np.array([3, 1, 6]), pixel_size=4, sigma=np.sqrt(2))
        expected = 6 * 0.5 * (erf(0.5) - erf(-1.5)) * 0.5 * (erf(2) - erf(0))
        self.assertAlmostEqual(result, expected, places=8)

    def test_d_expected_i_d_theta_i(self):
        result = d.d_expected_i_d_theta_i(coordinate=7, theta_i=4, pixel_size_i=2, sigma=np.sqrt(2))
        expected = (np.exp(-1) - np.exp(-4)) / (2 * np.sqrt(np.pi))
        self.assertAlmostEqual(result, expected, places=8)

    def test_d_expected_i_d_theta_i_zero(self):
        result = d.d_expected_i_d_theta_i(coordinate=4, theta_i=4, pixel_size_i=0, sigma=5)
        self.assertEqual(result, 0)

    def test_d2_expected_i_d_theta_i2(self):
        result = d.d2_expected_i_d_theta_i2(coordinate=3, theta_i=5, pixel_size_i=2, sigma=np.sqrt(2))
        expected = ((-3) * np.exp(-(1.5**2)) - (-1) * np.exp(-(0.5**2))) / (np.sqrt(np.pi) * 4)
        self.assertAlmostEqual(result, expected, places=8)


if __name__ == '__main__':
    unittest.main()
