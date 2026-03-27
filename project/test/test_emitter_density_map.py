import unittest
import numpy as np
import project.model.emitter_density_map as ne
import numpy.testing as np_test


class TestZerosCorrection(unittest.TestCase):
    def test_func(self):
        intensity = np.array([[2, 3, 1, 4], [4, 1, 0, 3]])
        e_d = np.ones((2, 4))
        np_test.assert_array_equal(ne.zeros_correction(e_d, intensity, 0.5), np.array([[1, 1, 0, 1], [1, 0, 0, 1]]))


class TestOutliersInterpolation(unittest.TestCase):
    a = np.array([
        [5, 1, 3, 4, 2],
        [3, 5, 2, 1, 4],
        [3, 4, 2, 1, 3],
        [1, 3, 4, 2, 2],
    ])

    def test_one_outlier_8nn(self):
        r = np.copy(self.a)
        r[1, 2] = 7
        expected = np.copy(self.a)
        expected[1, 2] = 21 / 8
        np_test.assert_array_almost_equal(ne.outliers_interpolation(r, threshold=6), expected)

    def test_threshold(self):
        r = np.copy(self.a)
        r[1, 2] = 7
        np_test.assert_array_almost_equal(ne.outliers_interpolation(r, threshold=7), r)

    def test_two_outliers_8nn_neighbours(self):
        r = np.copy(self.a)
        r[1, 2] = 7
        r[1, 3] = 8
        expected = np.copy(self.a)
        expected[1, 2] = 20 / 7
        expected[1, 3] = 19 / 7
        np_test.assert_array_almost_equal(ne.outliers_interpolation(r, threshold=6), expected)

    def test_border_outlier(self):
        r = np.copy(self.a)
        r[0, 0] = 7
        expected = np.copy(self.a)
        expected[0, 0] = 9 / 3
        np_test.assert_array_almost_equal(ne.outliers_interpolation(r, threshold=6), expected)

    def test_multiple(self):
        r = np.copy(self.a)
        r[0, 0] = 7
        r[0, 1] = 10
        r[1, 1] = 8.5
        expected = np.copy(self.a)
        expected[0, 0] = 3
        expected[0, 1] = 8 / 3
        expected[1, 1] = 17 / 6
        np_test.assert_array_almost_equal(ne.outliers_interpolation(r, threshold=6), expected)


if __name__ == '__main__':
    unittest.main()
