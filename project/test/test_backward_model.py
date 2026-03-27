import unittest
import numpy as np
import project.model.optimization_loglikelihood.backward_model as bm
from project.model.sample import Sensor


class TestFitter(unittest.TestCase):
    def test_initializer(self):
        s = Sensor(3)
        f = bm.Fitter(s, 0.43)
        self.assertEqual(f.sensor, s)
        self.assertEqual(f.assumed_psf_sigma, 0.43)

    def test_expected_count_zero_pixel_size(self):
        s = Sensor(3)
        s.pixel_size = 0
        f = bm.Fitter(s, 0.43)
        result = f.expected_count_per_pixel(np.array([1, 2, 3]))
        self.assertEqual(np.shape(result), (3,3))
        self.assertTrue(np.sum(result != 0) == 0)

    def test_expected_count_nonzero(self):
        s = Sensor(3)
        f = bm.Fitter(s, 600)
        result = f.expected_count_per_pixel(np.array([1, 2, 3]))
        self.assertTrue(np.sum(result == 0) == 0)


if __name__ == '__main__':
    unittest.main()
