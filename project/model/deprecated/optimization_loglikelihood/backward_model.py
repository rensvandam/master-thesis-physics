import numpy as np
import matplotlib.pyplot as plt
from project.model.optimization_loglikelihood.derivatives import hessian, gradient, fisher_information_matrix, expected_count


class Fitter:
    def __init__(self, sensor, assumed_psf_sigma=137.5):
        """
        Initializes a new Fitter object, coupled to the data that was measured by a Sensor.

        Parameters
        ----------
        sensor : Sensor
            Provides the measured data and the specifics of the sensor.
        assumed_psf_sigma : float
            The assumed standard deviation of PSFs in the measurements.
        """
        self.sensor = sensor
        self.assumed_psf_sigma = assumed_psf_sigma

    def expected_count_per_pixel(self, theta):
        """
        Calculates the expected value measured by pixels that are centered at (x, y) and have size (dx, dy).

        Parameters
        ----------
        theta : nd.array()
            (3,) array containing theta_x, theta_y and theta_i, estimates for the 2D position and the intensity of the
            emitter.

        Returns
        -------
        np.array()
            The expected number of photon counts in pixel k.
        """
        pixel_size = self.sensor.pixel_size
        sigma = self.assumed_psf_sigma
        x = self.sensor.x_centers
        y = self.sensor.y_centers
        return expected_count(x, y, theta, pixel_size, sigma)

    def log_likelihood(self, theta):
        """
        Calculates the natural logarithm of the likelihood function.

        Parameters
        ----------
        theta : nd.array()
            (3,) array containing theta_x, theta_y and theta_i, estimates for the 2D position and the intensity of the
            emitter.

        Returns
        -------
        float
            The log-likelihood of the data given this model.
        """
        expected_counts = self.expected_count_per_pixel(theta)
        counts = self.sensor.histogram

        # Prevent log(0) error by adding a 10^-300
        per_pixel = counts * np.log(expected_counts + 10 ** (-300)) \
                    - expected_counts \
                    - counts * np.log(counts + 10 ** (-300)) \
                    - counts
        return np.sum(per_pixel)

    def show_log_likelihood(self, expected):
        n = 1000
        x_axis = np.linspace(100, 1000, n)
        x_axis_intensity = np.linspace(5, 300, n)
        log_likelihood = np.zeros((n, 3))

        for i in range(n):
            log_likelihood[i, 0] = self.log_likelihood(np.array([x_axis[i], expected[1], expected[2]]))
            log_likelihood[i, 1] = self.log_likelihood(np.array([expected[0], x_axis[i], expected[2]]))
            log_likelihood[i, 2] = self.log_likelihood(np.array([expected[0], expected[1], x_axis_intensity[i]]))

        labels = np.array(['x', 'y'])
        for i in range(2):
            plt.plot(x_axis, log_likelihood[:, i])
            plt.title(fr"Loglikelihood $\theta_{labels[i]}$")
            plt.xlabel(f"{labels[i]} coordinate (nm)")
            plt.show()

        plt.plot(x_axis_intensity, log_likelihood[:, 2])
        plt.title(r"Loglikelihood $\theta_i$")
        plt.xlabel("Number of photons")
        plt.show()

    def levenberg_marquardt(self, initial_guess=np.array([1, 1, 1]), max_iterations=10, tolerance=0.5):
        """
        Performs the Levenberg-Marquardt algorithm in order to optimize the parameters theta_x, theta_y and theta_z. The
        optimization stops either when the maximum number of iterations was done or when the error is smaller than the
        tolerance.

        Parameters
        ----------
        initial_guess : np.array()
            An initial guess for the parameters that need to be optimized. Default is [1, 1, 1]
        max_iterations : int
            The maximum number of iterations that should be done.
        tolerance : float
            The maximum accepted tolerance of the error in the function that is approximated.
        """
        theta = initial_guess
        f_current = self.log_likelihood(theta)
        l = 0.001
        l_factor = 10
        error = tolerance + 1
        i = 0
        while error > tolerance and i < max_iterations:
            i = i + 1
            # print(f'Iteration: {i},\ttheta = {theta}')

            hessian_matrix = hessian(self.sensor.histogram, self.sensor.x_centers, self.sensor.y_centers, theta,
                                     self.sensor.pixel_size, self.assumed_psf_sigma)
            diag_hessian = np.diag(np.diag(hessian_matrix))
            gradient_vector = gradient(self.sensor.histogram, self.sensor.x_centers, self.sensor.y_centers, theta,
                                       self.sensor.pixel_size, self.assumed_psf_sigma)
            d_theta = (-1) * np.linalg.inv(hessian_matrix + l * diag_hessian) @ gradient_vector

            proposed_theta = theta + d_theta
            f_new = self.log_likelihood(proposed_theta)
            if f_new <= f_current:
                l = l * l_factor
            else:
                l = l / l_factor
                theta = proposed_theta
                error = abs(f_new - f_current)
                f_current = f_new

        print(f"Result:\t\t\ttheta = {theta}, error: {error}, iterations: {i}")
        return theta

    def cramer_rao_lower_bound(self, theta):
        """
        Determines the limiting lower bound of the variance y for the estimated theta by determining the Cramer Rao
        lower bound.

        Parameters
        ----------
        theta : np.array()
            (3,) array containing the estimated parameters for theta_x, theta_y and theta_i.

        Returns
        -------
        np.array()
            The minimum obtainable standard error for parameter theta_x, theta_y and theta_i.
        """
        fisher = fisher_information_matrix(self.sensor.x_centers, self.sensor.y_centers, theta, self.sensor.pixel_size,
                                           self.assumed_psf_sigma)
        inv_fisher = np.linalg.inv(fisher)
        return np.sqrt(np.diag(inv_fisher))
