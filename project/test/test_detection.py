import unittest
import numpy as np
import numpy.testing as np_test

import project.model.detection as d


class TestSpad23(unittest.TestCase):
    def test_init(self):
        s = d.Spad23(1.1, 1, 2, 3, 0.4, 0.5, 6, 7, 8)
        self.assertEqual(s.magnification, 1.1)
        self.assertEqual(s.nr_pixel_rows, 1)
        self.assertEqual(s.pixel_radius, 2)
        self.assertEqual(s.spacing, 3)
        self.assertEqual(s.crosstalk, 0.4)
        self.assertEqual(s.afterpulsing, 0.5)
        self.assertEqual(s.jitter, 6)
        self.assertEqual(s.dead_time, 7)
        self.assertEqual(s.dark_count_rate, 8)
        self.assertIsNotNone(s.pixel_coordinates)
        self.assertIsNotNone(s.x_limits)
        self.assertIsNotNone(s.y_limits)
        np_test.assert_array_equal(s.data_by_time, np.array([]))
        self.assertIsNotNone(s.data_per_pixel)

    def test_even_pixel_rows_error(self):
        with self.assertRaises(ValueError):
            d.Spad23(nr_pixel_rows=2)

    def test_nr_pixels(self):
        self.assertEqual(d.Spad23(nr_pixel_rows=1).nr_pixels, 1)
        self.assertEqual(d.Spad23(nr_pixel_rows=3).nr_pixels, 8)
        self.assertEqual(d.Spad23(nr_pixel_rows=5).nr_pixels, 23)

    def test_calculate_pixel_coordinates(self):
        s = d.Spad23(spacing=10)
        np_test.assert_array_almost_equal(s.pixel_coordinates[0, :], np.array([-20, -10 * np.sqrt(3)]))
        np_test.assert_array_almost_equal(s.pixel_coordinates[22, :], np.array([20, 10 * np.sqrt(3)]))
        np_test.assert_array_almost_equal(s.pixel_coordinates[11, :], np.array([0, 0]))
        np_test.assert_array_almost_equal(s.pixel_coordinates[14, :], np.array([-15, 5 * np.sqrt(3)]))

    def test_init_sensor_limits(self):
        s = d.Spad23(pixel_radius=5, spacing=20)
        np_test.assert_array_almost_equal(s.x_limits, np.array([-45, 45]))
        np_test.assert_array_almost_equal(s.y_limits, np.array([-20 * np.sqrt(3) - 5, 20 * np.sqrt(3) + 5]))

    def test_measure_photons_empty(self):
        s = d.Spad23()
        s.measure([], duration=1, seed=1)
        np_test.assert_array_equal(s.data_by_time, np.array([]))
        np_test.assert_array_equal(s.photon_count, np.zeros(23))

    def test_clear(self):
        s = d.Spad23()
        s.data_per_pixel[1] = [1, 2, 3, 4]
        s.data_per_pixel[19] = [1, 1, 1]
        s.photon_count[1] = 4
        s.photon_count[19] = 3
        s.data_by_time = np.array([[1, 2, 3], [1, 2, 3], [4, 4, 4], [2, 3, 6]])
        s.clear()

        np_test.assert_array_equal(s.data_by_time, np.array([]))
        self.assertEqual(s.data_per_pixel, {})
        np_test.assert_array_equal(s.photon_count, np.zeros(23))

    def test_discard_off_limit_photons(self):
        s = d.Spad23()
        s.x_limits = np.array([-4.9, 4.9])
        s.y_limits = np.array([-4.8, 4.8])
        photons = np.array([
            [-5, -10, 1],  # outside x, outside y corner 1
            [-8.5, 1, 2],  # outside x, inside  y
            [-0.5, -9.5, 3],  # inside x, outside y
            [0, 0, 4],  # on the sensor
            [6, 2, 5],  # outside x, inside y
            [2, 9.3, 6],  # inside x, outside y
            [5.1, 5, 7],  # outside x, outside y, corner 2
            [-6, 7, 8],  # corner 3
            [9, -5.5, 9],  # corner 4
            [4.9, -4.8, 10]  # on the boundary
        ])
        np_test.assert_array_equal(s._discard_off_limit_photons(photons), np.array([False, False, False, True, False,
                                                                                    False, False, False, False, True]))

    def test_project_onto_pixels(self):
        photons = np.array([
            [-100, 183.9, 1],   # pixel 19
            [-25, 0, 1.5],      # left edge pixel 11
            [25, 0, 2],         # right edge pixel 11
            [-200, 86.6, 3],    # no pixel
            [40, 70, 4],        # pixel 16
            [0, -25, 5],        # lower edge pixel 11 ! Is not detected due to rounding errors.
            [0, 25, 6],         # upper edge pixel 11
            [-30, -140, 7],     # no pixel
            [223, -170, 8],     # pixel 4
            [-210, -189, 9],    # pixel 0
            [147, -90, 10],     # pixel 8
        ])

        expected = np.array([
            [19, 1],    # pixel 19
            [11, 1.5],  # left edge pixel 11
            [11, 2],    # right edge pixel 11
            [14, 3],    # no pixel
            [16, 4],    # pixel 16
            [11, 5],    # not detected due to rouding error
            [11, 6],    # upper edge pixel 11
            [2, 7],     # not on active pixel area
            [4, 8],     # pixel 4
            [0, 9],     # pixel 0
            [8, 10],    # pixel 8
        ])
        s = d.Spad23(pixel_radius=25, spacing=100)
        result, is_detected = s._project_onto_pixels(photons)
        np_test.assert_array_equal(result, expected)
        np_test.assert_array_equal(is_detected, np.array([True, True, True, False, True, False, True, False, True, True,
                                                          True]))

    def test_project_onto_pixels_false_positive(self):
        photons = np.array([
            [-210, 86.6, 1],    # upper left
            [220, 86.6, 2],     # upper right
            [-215, -90, 3],     # lower left
            [205, -90, 4]       # lower right
        ])
        expected = np.array([
            [13, 1],    # upper left
            [18, 2],     # upper right
            [4, 3],     # lower left
            [9, 4]       # lower right
        ])
        s = d.Spad23(pixel_radius=50, spacing=100)
        result, is_detected = s._project_onto_pixels(photons)
        np_test.assert_array_equal(result, expected)
        np_test.assert_array_equal(is_detected, np.array([False, False, False, False]))

    def test_find_neighbors(self):
        s = d.Spad23()
        actual = s._Spad23__find_neighbors()
        self.assertEqual([1, 5], sorted(actual[0]))
        self.assertEqual([17, 21], sorted(actual[22]))
        self.assertEqual([2, 3, 6, 8, 11, 12], sorted(actual[7]))
        self.assertEqual([9, 10, 15, 18, 19], sorted(actual[14]))
        self.assertEqual([8, 12, 17], sorted(actual[13]))


class TestSpad512(unittest.TestCase):
    def test_init(self):
        s = d.Spad512(1, 8, 3, 4, 5, 6, 7, 8, 9)
        self.assertEqual(1, s.magnification)
        self.assertEqual(8, s.nr_pixel_rows)
        self.assertEqual(3, s.pixel_radius)
        self.assertEqual(4, s.spacing)
        self.assertEqual(5, s.crosstalk)
        self.assertEqual(6, s.afterpulsing)
        self.assertEqual(7, s.jitter)
        self.assertEqual(8, s.dead_time)
        self.assertEqual(9, s.dark_count_rate)

        self.assertEqual(64, s.nr_pixels)
        self.assertIsNotNone(s.pixel_coordinates)
        self.assertIsNotNone(s.x_limits)
        self.assertIsNotNone(s.y_limits)
        self.assertIsNotNone(s.data_by_time)
        self.assertIsNotNone(s.data_per_pixel)
        self.assertIsNotNone(s.photon_count)

    def test_calculate_pixel_coordinates(self):
        s = d.Spad512(nr_pixel_rows=3, spacing=12.5)
        expected = np.array([
            [0, 0],
            [12.5, 0],
            [25, 0],
            [0, 12.5],
            [12.5, 12.5],
            [25, 12.5],
            [0, 25],
            [12.5, 25],
            [25, 25]
        ])
        np_test.assert_array_equal(expected, s.pixel_coordinates)

    def test_init_sensor_limits(self):
        s = d.Spad512(nr_pixel_rows=5, spacing=20, pixel_radius=12)
        self.assertEqual((-12, 92), s.x_limits)
        self.assertEqual((-12, 92), s.y_limits)

    def test_project_onto_pixels(self):
        s = d.Spad512(nr_pixel_rows=5, spacing=20, pixel_radius=5)
        photons = np.array([
            [-3, -2, 1],  # pixel 0
            [34, 68, 2],  # not on a pixel
            [41, 23, 3],  # pixel 7
            [82, 78, 4],  # pixel 24
            [-5, 30, 5],  # not on a pixel
        ])
        expected = np.array([
            [0, 1],
            [17, 2],
            [7, 3],
            [24, 4],
            [10, 5]
        ])
        result, is_detected = s._project_onto_pixels(photons)
        np_test.assert_array_equal(expected, result)
        np_test.assert_array_equal(np.array([True, False, True, True, False]), is_detected)

