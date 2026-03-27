import unittest
import numpy as np
import numpy.testing as np_test
from project.model import coherence_from_data as c
from project.model.helper_functions import average_nn_first_degree, average_nn_second_degree


a = np.array([0, 2, 0, 0, 0, 1, 0, 1, 0, 0])
times_a = np.array([1.5, 1.5, 5.5, 7.5])
a_result = np.array([2, 0, 1, 0, 2])

b = np.array([0, 1, 0, 0, 3, 0, 0, 0, 1, 2])
times_b = np.array([1.3, 4.1, 4.15, 4.73, 8.9, 9.3, 9.5])
b_result = np.array([8, 2, 0, 3, 3, 6])

c = np.array([0, 2, 0, 0, 1, 4, 0, 3, 0, 0])
times_c = np.array([1.4, 1.4, 4.7, 5, 5.1, 5.2, 5.6, 7.4, 7.88, 7.89])
c_result = np.array([20, 4, 12, 5, 8, 0, 6])


class TestAutoCoherence(unittest.TestCase):

    def test_a(self):
        N = len(a)
        pairs, lags = c.auto_coherence(times_a, N, 1, len(a_result), normalize=False)
        expected = a_result * N / np.arange(N, N - len(a_result), -1)
        np_test.assert_array_equal(pairs, expected)
        np_test.assert_array_equal(lags, np.array([0, 1, 2, 3, 4]))

    def test_b(self):
        N = len(b)
        pairs, _ = c.auto_coherence(times_b, N, 1, len(b_result), normalize=True)
        expected = b_result * N / np.arange(N, N - len(b_result), -1)
        expected = expected * N / (len(times_b) * len(times_b))
        np_test.assert_array_almost_equal(pairs, expected)

    def test_c_repeating_indices(self):
        pairs, _ = c.auto_coherence(times_c, 10, 1, len(c_result), normalize=False)
        expected = c_result * 10 / np.arange(10, 10 - len(c_result), -1)
        np_test.assert_array_almost_equal(pairs, expected)

    def test_c_repeating_indices_offset(self):
        pairs, _ = c.auto_coherence(times_c, 10, 1, len(c_result) - 2, offset=2, normalize=False)
        expected = c_result * 10 / np.arange(10, 10 - len(c_result), -1)
        np_test.assert_array_almost_equal(pairs, expected[2:])


class TestCoherence(unittest.TestCase):

    def test_empty_signal(self):
        pairs, _ = c.coherence(times_a, np.array([]), 10, 1, 5, normalize=True)
        expected = np.array([0, 0, 0, 0, 0])
        np_test.assert_array_equal(pairs, expected)

    def test_a_b(self):
        pairs, _ = c.coherence(times_a, times_b, 10, 1, 5, normalize=False)
        expected = np.array([2, 3, 0, 3, 1]) * 10 / np.arange(10, 5, -1)
        np_test.assert_array_equal(pairs, expected)

    def test_b_a(self):
        pairs, _ = c.coherence(times_b, times_a, 10, 1, 5, normalize=False)
        expected = np.array([2, 1, 2, 7, 2]) * 10 / np.arange(10, 5, -1)
        np_test.assert_array_equal(pairs, expected)

    def test_b_c(self):
        pairs, _ = c.coherence(times_b, times_c, 10, 1, 5, normalize=False, offset=0)
        expected = np.array([5, 3, 6, 10, 9]) * 10 / np.arange(10, 5, -1)
        np_test.assert_array_equal(pairs, expected)

    def test_b_c_offset(self):
        pairs, lags = c.coherence(times_b, times_c, 10, 1, 3, normalize=False, offset=2)
        expected = np.array([5, 3, 6, 10, 9]) * 10 / np.arange(10, 5, -1)
        np_test.assert_array_equal(pairs, expected[2:])
        np_test.assert_array_equal(lags, np.array([2, 3, 4]))

    def test_a_b_normalize(self):
        pairs, _ = c.coherence(times_a, times_b, 10, 1, 5, normalize=True)
        expected = np.array([2, 3, 0, 3, 1]) * 10 / np.arange(10, 5, -1)
        normalized = expected * 10 / (len(times_a) * len(times_b))
        np_test.assert_array_equal(pairs, normalized)


class TestNearestNeighbour(unittest.TestCase):
    x_dim = 2
    y_dim = 3
    c_matrix = np.array([
        [3, 4, 5, 1, 3, 8],
        [8, 3, 3, 2, 0, 9],
        [0, 7, 5, 4, 5, 8],
        [9, 0, 8, 1, 0, 4],
        [7, 4, 2, 7, 1, 1],
        [8, 1, 5, 3, 7, 9]
    ])

    def test_degree0(self):
        result = c.nearest_neighbour_coherence(self.x_dim, self.y_dim, self.c_matrix, degree=0)
        expected = np.array([
            [3, 3, 5],
            [1, 1, 9]
        ])
        np_test.assert_array_equal(result, expected)

    def test_degree1(self):
        result = c.nearest_neighbour_coherence(self.x_dim, self.y_dim, self.c_matrix, degree=1, threshold=5)
        expected = average_nn_first_degree(self.x_dim, self.y_dim, self.c_matrix, threshold=5)
        np_test.assert_array_equal(result, expected)

    def test_degree2(self):
        expected = average_nn_second_degree(self.x_dim, self.y_dim, self.c_matrix, threshold=5)
        result = c.nearest_neighbour_coherence(self.x_dim, self.y_dim, self.c_matrix, degree=2, threshold=5)
        np_test.assert_array_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
