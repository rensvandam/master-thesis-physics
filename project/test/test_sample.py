import unittest
import numpy as np
import numpy.testing as np_test

import project.model.sample as m


class TestEmitter(unittest.TestCase):
    def test_constructor(self):
        e = m.Emitter(1.5, -4.3, 415, 450, 5, 50000)
        self.assertEqual(e.x, 1.5)
        self.assertEqual(e.y, -4.3)
        self.assertEqual(e.absorption_wavelength, 415)
        self.assertEqual(e.emission_wavelength, 450)
        self.assertEqual(e.sigma, 112.5)
        self.assertEqual(np.shape(e.photons), (0,))
        self.assertEqual(e.lifetime, 5)
        self.assertEqual(e.extinction_coefficient, 50000)

    def test_expected_exp_time(self):
        laser = 10 * 10 ** 3  # W / cm^2
        abs_wavelength = 650
        em_wavelength = 665
        extinct_coef = 240000
        lifetime = 5
        planck = 6.626 * 10 ** (-34)
        c = 2.9979 * 10 ** 8

        cross_section = 3.82 * 10 ** (-21) * extinct_coef  # cm^2
        # J s^-1 cm^-2 * nm / (J s * nm s^-1) = s^-1 cm^-2
        photon_flux = laser * abs_wavelength / (planck * c * 10 ** 9)
        photon_flux_per_ns = photon_flux * 10 ** (-9)

        e = m.Emitter(0, 0, abs_wavelength, em_wavelength, lifetime, extinct_coef)
        np_test.assert_almost_equal(m.expected_excitation_time(extinct_coef, abs_wavelength, laser),
                                    1 / (cross_section * photon_flux_per_ns), decimal=1)

    def test_excitation_time(self):
        N = 10000
        exp_exc = 5.5
        random = np.random.default_rng(seed=1)
        result = m.excitation_time(exp_exc, N, random)

        average = np.mean(result)
        np_test.assert_almost_equal(average, exp_exc, decimal=1)

        std_dev = np.std(result)
        np_test.assert_almost_equal(std_dev, exp_exc, decimal=1)

    def test_emission_time(self):
        random = np.random.default_rng(seed=1)
        N = 100000
        result = m.emission_time(5, N, random)
        np_test.assert_almost_equal(np.mean(result), 5, decimal=1)
        np_test.assert_almost_equal(np.std(result), 5, decimal=1)

    def test_generate_timestamps(self):
        # Generate timestamps for 10 kW/cm2 during a 1000 ns interval
        e = m.Emitter(0, 0, 650, 665, 1, 240000)
        result = e.__generate_timestamps(10 * 10 ** 3, 1000, np.random.default_rng(seed=5), detection_efficiency=1,
                                         statistics="sub_poisson")
        np_test.assert_array_equal(result, sorted(result))

    def test_generate_photons(self):
        e = m.Emitter(2, 3.5, 550, 1, 5, 300000)
        self.assertEqual(np.shape(e.photons), (0,))

        e.generate_photons(laser_power=100*10**3, time_interval=100000, seed=1)
        self.assertEqual(np.shape(e.photons)[1], 3)

        average = np.mean(e.photons, axis=0)
        self.assertAlmostEqual(average[e.X], 2, places=1)
        self.assertAlmostEqual(average[e.Y], 3.5, places=1)

        std_dev = np.std(e.photons, axis=0)
        self.assertAlmostEqual(std_dev[e.X], e.sigma, places=1)
        self.assertAlmostEqual(std_dev[e.Y], e.sigma, places=1)


if __name__ == '__main__':
    unittest.main()
