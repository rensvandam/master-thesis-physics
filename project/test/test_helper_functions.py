import unittest
import numpy as np
import numpy.testing as np_test
from project.model import helper_functions as h


class MergeK(unittest.TestCase):
    def test_empty(self):
        r = h.merge_k([])
        np_test.assert_array_equal(r, np.array([]))

    def test_no_elements(self):
        r = h.merge_k([np.array([]), np.array([])])
        np_test.assert_array_equal(r, np.array([]))

    def test_one_array(self):
        input = np.array([2, 6, 9])
        r = h.merge_k([input])
        np_test.assert_array_equal(r, input)

    def test_multiple_arrays(self):
        array1 = np.array([5, 7, 9])
        array2 = np.array([1, 1, 2, 6])
        array3 = np.array([2, 6, 10])
        array_list = [array1, array2, array3]
        r = h.merge_k(array_list)
        expected = np.array([1, 1, 2, 2, 5, 6, 6, 7, 9, 10])
        np_test.assert_array_equal(r, expected)


class MergeK2D(unittest.TestCase):
    def test_empty(self):
        r = h.merge_k_2D([])
        np_test.assert_array_equal(r, np.array([]))

    def test_no_elements(self):
        r = h.merge_k_2D([np.array([]), np.array([])])
        np_test.assert_array_equal(r, np.array([]))

    def test_one_array(self):
        input = np.array([[2, 4, 3], [0, 8, 6], [1, 2, 10]])
        r = h.merge_k_2D([input])
        np_test.assert_array_equal(r, input)

    def test_multiple_arrays(self):
        array1 = np.array([[2, 4, 3], [0, 8, 6], [1, 2, 10]])
        array2 = np.array([[7, 10, 1], [5, 1, 2], [9, 6, 7]])
        array3 = np.array([[1, 3, 9]])
        array_list = [array1, array2, array3]
        r = h.merge_k_2D(array_list)
        expected = np.array([
            [7, 10, 1],
            [5, 1, 2],
            [2, 4, 3],
            [0, 8, 6],
            [9, 6, 7],
            [1, 3, 9],
            [1, 2, 10],
        ])
        np_test.assert_array_equal(r, expected)


class InsertionSort2D(unittest.TestCase):
    def test_empty(self):
        np_test.assert_array_equal(np.array([]), h.insertion_sort_2D(np.array([]), 1))

    def test_one(self):
        np_test.assert_array_equal(np.array([[1, 3]]), h.insertion_sort_2D(np.array([[1, 3]]), 1))

    def test_array(self):
        a = np.array([
            [3, 1],
            [5.5, 2],
            [1, 15],
            [6.2, 6.1],
            [2, 8],
            [3, 14],
            [5.1, 10]
        ])
        expected = np.array([
            [3, 1],
            [5.5, 2],
            [6.2, 6.1],
            [2, 8],
            [5.1, 10],
            [3, 14],
            [1, 15],
        ])
        np_test.assert_array_equal(expected, h.insertion_sort_2D(a, sort_by_index=1))


class CountPairs(unittest.TestCase):
    def test_count_pairs_empty_arrays(self):
        r = h.count_pairs(np.array([]), np.array([]))
        self.assertEqual(r, 0)

    def test_count_pairs_array1_empty(self):
        r = h.count_pairs(np.array([]), np.array([1, 2, 3]))
        self.assertEqual(r, 0)

    def test_count_pairs_array2_empty(self):
        r = h.count_pairs(np.array([1, 2, 3]), np.array([]))
        self.assertEqual(r, 0)

    def test_count_pairs_different_length(self):
        r = h.count_pairs(np.array([1, 2, 3]), np.array([3, 4]))
        self.assertEqual(r, 1)

    def test_count_pairs_equal_length(self):
        r = h.count_pairs(np.array([1, 2, 3]), np.array([2, 3, 4]))
        self.assertEqual(r, 2)

    def test_count_pairs_repeating_digits(self):
        r = h.count_pairs(np.array([1, 2, 2, 4]), np.array([2, 2, 2, 3, 5]))
        self.assertEqual(r, 6)

    def test_count_pairs(self):
        r = h.count_pairs(np.array([0, 4, 6, 8, 8]), np.array([0, 1, 1, 5, 6, 8, 10]))
        self.assertEqual(r, 4)


def array_of_indices(array):
    result = []
    for i, elem in enumerate(array):
        for j in range(elem):
            result.append(i)
    return np.array(result)


class SparseConvolution(unittest.TestCase):
    a = array_of_indices(np.array([0, 2, 0, 0, 0, 1, 0, 1, 0, 0]))
    a_result = np.array([6, 0, 1, 0, 2])

    b = array_of_indices(np.array([0, 1, 0, 0, 3, 0, 0, 0, 1, 2]))
    b_result = np.array([15, 2, 0, 3, 3, 6])

    c = array_of_indices(np.array([0, 2, 0, 0, 1, 4, 0, 3, 0, 0]))
    c_result = np.array([30, 4, 12, 5, 8, 0, 6])

    def test_sparse(self):
        pairs = h.sparse_convolution(self.a, self.a, nr_steps=len(self.a_result))
        np_test.assert_array_equal(pairs, self.a_result)

    def test_repeating_times(self):
        result = h.sparse_convolution(self.b, self.b, nr_steps=len(self.b_result))
        np_test.assert_array_equal(result, self.b_result)

    def test_repeating_times_2(self):
        result = h.sparse_convolution(self.c, self.c, nr_steps=len(self.c_result))
        np_test.assert_array_equal(result, self.c_result)

    def test_repeating_times_2_some_steps(self):
        result = h.sparse_convolution(self.c, self.c, nr_steps=len(self.c_result) - 3, offset=3)
        np_test.assert_array_equal(result, self.c_result[3:])

    def test_repeating_times_2_only_step_0(self):
        result = h.sparse_convolution(self.c, self.c, 1, 0)
        np_test.assert_array_equal(result, self.c_result[0])


class SelectKernel(unittest.TestCase):
    def test_normal(self):
        expected = [1, 2, 3, 6, 7, 8, 11, 12, 13]
        self.assertEqual(expected, h.select_neighbours(3, 7, 4, 5))

    def test_upper_out(self):
        expected = [1, 2, 3, 6, 7, 8]
        self.assertEqual(expected, h.select_neighbours(3, 2, 4, 5))

    def test_left_out(self):
        expected = [0, 1, 5, 6, 10, 11]
        self.assertEqual(expected, h.select_neighbours(3, 5, 4, 5))

    def test_right_out(self):
        expected = [8, 9, 13, 14, 18, 19]
        self.assertEqual(expected, h.select_neighbours(3, 14, 4, 5))

    def test_lower_out(self):
        expected = [17, 18, 19]
        self.assertEqual(expected, h.select_neighbours(3, 23, 4, 5))

    def test_multiple_out(self):
        expected = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18]
        self.assertEqual(expected, h.select_neighbours(5, 11, 4, 5))

    def test_single_kernel_inside(self):
        self.assertEqual([12], h.select_neighbours(1, 12, 4, 5))

    def test_single_kernel_outside(self):
        self.assertEqual([], h.select_neighbours(1, 33, 4, 5))


class Average(unittest.TestCase):
    def testAverage(self):
        a = np.array([
            [6, 12, 6, 8, 0],
            [7, 0, 1, 5, 9]
        ])
        c = np.array([
            [6, 3, 1, 0, 2],
            [7, 0, 4, 5, 5]
        ])
        expected = np.array([
            [1, 4, 6, 0, 0],
            [1, 0, 0.25, 1, 9 / 5]
        ])
        np_test.assert_array_equal(h.average(a, c), expected)


class AverageNN(unittest.TestCase):
    x = 2
    y = 3
    a = np.array([
        [1, 2, 3, 6, 0, 3],
        [6, 3, 4, 2, 2, 8],
        [8, 2, 10, 2, 5, 5],
        [5, 28, 35, 8, 6, 2],
        [0, 9, 7, 5, 3, 7],
        [4, 6, 8, 2, 54, 5]
    ])

    def test_zero_degree(self):
        expected = np.array([
            [1, 3, 10],
            [8, 3, 5]
        ])
        result = h.average_nn(self.x, self.y, self.a, threshold=8.0, degree=0)
        np_test.assert_array_equal(result, expected)

    def test_other_degree(self):
        expected = np.array([
            [0, 0, 0],
            [0, 0, 0]
        ])
        result = h.average_nn(self.x, self.y, self.a, threshold=8.0, degree=4)
        np_test.assert_array_equal(result, expected)

    def test_first_degree(self):
        expected = np.array([
            [4, 4, 3.5],
            [5.5, 6, 0]
        ])
        result = h.average_nn(self.x, self.y, self.a, threshold=8.0, degree=1)
        np_test.assert_array_equal(result, expected)

    def test_second_degree(self):
        expected = np.array([
            [0, 2, 5],
            [0, 3.5, 6]
        ])
        result = h.average_nn(self.x, self.y, self.a, threshold=8.0, degree=2)
        np_test.assert_array_equal(result, expected)

    def test_third_degree(self):
        expected = np.array([
            [8 / 3, 14 / 4, 4],
            [5.5, 19 / 4, 6]
        ])
        result = h.average_nn(self.x, self.y, self.a, threshold=8.0, degree=3)
        np_test.assert_array_equal(result, expected)


class SumNN(unittest.TestCase):
    a = np.array([
        [5, 6, 8, 0],
        [5, 2, 10, 0],
        [6.5, 4, 3, 5],
        [1, 1, 3, 9]
    ])

    def test_first_degree(self):
        expected_sum = np.array([[6, 5], [11.5, 4]])
        expected_count = np.array([[1, 2], [2, 2]])
        s, c = h.sum_first_degree_nn(2, 2, self.a, threshold=7)
        np_test.assert_array_equal(s, expected_sum)
        np_test.assert_array_equal(c, expected_count)

    def test_second_degree(self):
        expected_sum = np.array([[0, 0], [4, 1]])
        expected_count = np.array([[1, 0], [1, 1]])
        s, c = h.sum_second_degree_nn(2, 2, self.a, threshold=7)
        np_test.assert_array_equal(s, expected_sum)
        np_test.assert_array_equal(c, expected_count)

    def test_sum_nn_1(self):
        expected_sum = np.array([[6, 5], [11.5, 4]])
        expected_count = np.array([[1, 2], [2, 2]])
        s, c = h.sum_nn(2, 2, self.a, threshold=7, degree=1)
        np_test.assert_array_equal(s, expected_sum)
        np_test.assert_array_equal(c, expected_count)

    def test_sum_nn_2(self):
        expected_sum = np.array([[0, 0], [4, 1]])
        expected_count = np.array([[1, 0], [1, 1]])
        s, c = h.sum_nn(2, 2, self.a, threshold=7, degree=2)
        np_test.assert_array_equal(s, expected_sum)
        np_test.assert_array_equal(c, expected_count)

    def test_sum_nn_3(self):
        expected_sum = np.array([[6, 5], [15.5, 5]])
        expected_count = np.array([[2, 2], [3, 3]])
        s, c = h.sum_nn(2, 2, self.a, threshold=7, degree=3)
        np_test.assert_array_equal(s, expected_sum)
        np_test.assert_array_equal(c, expected_count)

    def test_sum_nn_4(self):
        expected_sum = np.array([[0, 0], [0, 0]])
        expected_count = np.array([[0, 0], [0, 0]])
        s, c = h.sum_nn(2, 2, self.a, threshold=7, degree=4)
        np_test.assert_array_equal(s, expected_sum)
        np_test.assert_array_equal(c, expected_count)


class TestMSE(unittest.TestCase):
    def test_different_sizes(self):
        self.assertRaises(ValueError, h.mean_squared_error, np.ones((3, 3)), np.ones((3, 3)), np.ones((2, 3)))

    def test_no_weights(self):
        a_1 = np.array([[1, 2, 3], [5, 6, 7]]).astype('float')
        a_2 = np.array([[2, 5, 3], [5, -1, 4]]).astype('float')
        expected = (1 + 9 + 49 + 9) / 6
        self.assertEqual(h.mean_squared_error(a_1, a_2), expected)

    def test_no_difference(self):
        a_1 = np.array([[4.3, 1, -3], [8.1, 2, 4]])
        a_2 = np.array([[4.3, 1, -3], [8.1, 2, 4]])
        self.assertEqual(h.mean_squared_error(a_1, a_2), 0.0)

    def test_weights(self):
        weights = np.array([[0.5, 1, 1], [1, 2.0, 4]])
        a_1 = np.array([[1.5, 2, 3], [5, 6, 7]])
        a_2 = np.array([[2.5, 5, 3], [5, 5.5, 4]])
        expected = (1 / 8 + 9 / 4 + 0.25 * 0.5 + 9) / (9.5 / 4)
        self.assertEqual(h.mean_squared_error(a_1, a_2, weights), expected)


if __name__ == '__main__':
    unittest.main()
