import numpy as np
from matplotlib import pyplot as plt

from project.model.sample import Alexa647, Sensor
from project.model.optimization_loglikelihood.backward_model import Fitter


def run_simulation(nr_simulations=1):
    print(f"Running {nr_simulations} simulations...")
    nr_pixels = 7
    pixel_size = 665 / 4
    theta_estimates = np.zeros((nr_simulations, 3))
    nr_photons = np.zeros(nr_simulations)

    s = Sensor(nr_pixels, pixel_size=pixel_size)
    f = Fitter(sensor=s, assumed_psf_sigma=665 / 4)

    for i in range(nr_simulations):
        e = Alexa647(x=400, y=580)
        e.generate_photons(laser_power=10 * 10 ** 3, time_interval=1000, seed=i)

        photons = s.measure(emitters=[e])
        nr_photons[i] = len(photons)
        # s.show_measured_photons()
        # show_log_likelihood(np.array([s.pixel_size * nr_pixels / 2, s.pixel_size * nr_pixels / 2, len(photons)], f)

        theta_estimates[i, 0], theta_estimates[i, 1], theta_estimates[i, 2] = f.levenberg_marquardt(
            initial_guess=np.array([500, 500, 20]),
            max_iterations=100,
            tolerance=0.00001
        )

        s.clear_sensor()

    expected = np.array([e.x, e.y, np.mean(nr_photons)])
    crlb = f.cramer_rao_lower_bound(np.mean(theta_estimates, axis=0))
    show_simulation_results(nr_simulations, theta_estimates, crlb, expected)


def show_simulation_results(nr_simulations, theta_estimates, crlb, expected):
    mean = np.mean(theta_estimates, axis=0)
    std_dev = np.std(theta_estimates, axis=0)
    print(f"\nRESULTS - {nr_simulations} simulations"
          f"\n--------------------------"
          f"\n   | Expected |  Found  |  Std dev  |    CRLB   |"
          f"\n x |  {expected[0]} | {mean[0]:#.3f} |  {std_dev[0]:#.4f}  | {crlb[0]:#.4f} |"
          f"\n y |  {expected[1]} | {mean[1]:#.3f} |  {std_dev[1]:#.4f}  | {crlb[1]:#.4f} |"
          f"\n i |   {expected[2]} |  {mean[2]:#.3f} |   {std_dev[2]:#.4f}  |   {crlb[2]:#.4f} |"
          )

    labels = np.array(['x coordinate', 'y coordinate', 'number of photons'])
    for i in range(3):
        plt.hist(theta_estimates[:, i])
        plt.axvline(expected[i], color='red', linestyle='--', label='Expected')
        plt.title(f"Estimated {labels[i]}")
        plt.legend()
        plt.show()
