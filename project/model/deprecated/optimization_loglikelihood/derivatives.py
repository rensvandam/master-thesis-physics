from scipy.special import erf
import numpy as np

X = 0  # The index of the X coordinate
Y = 1  # The index of the Y coordinate
I = 2  # The index of the intensity variable


def expected_i(coordinate, theta_i, pixel_size_i, sigma):
    """
    Calculates the expected photon count for a pixel across one dimension i, where i is x or y.

    Parameters
    ----------
    coordinate : np.array()
        A (nr_pixels, nr_pixels) array containing the 1d center coordinates of all pixels.
    theta_i : float
        An estimate for the true value of the emitter position.
    pixel_size_i : float
        The size of the pixel in this direction.
    sigma : float
        The assumed size of a PSF on the sensor.

    Returns
    -------
    np.array()
        The expected photon count on this pixel calculated over 1 dimension.
    """
    return 0.5 * (erf((coordinate - theta_i + 0.5 * pixel_size_i) / (sigma * np.sqrt(2)))
                  - erf((coordinate - theta_i - 0.5 * pixel_size_i) / (sigma * np.sqrt(2))))


def d_expected_i_d_theta_i(coordinate, theta_i, pixel_size_i, sigma):
    """
    Calculates the first derivative of 'the expected photon count in dimension i' to 'the estimated position theta_i'.

    Parameters
    ----------
    coordinate : np.array()
        A (nr_pixels, nr_pixels) array containing the 1d center coordinates of all pixels.
    theta_i : float
        An estimate for the true value of the emitter position.
    pixel_size_i : float
        The size of the pixel in this direction.
    sigma : float
        The assumed size of a PSF on the sensor.

    Returns
    -------
    np.array()
        The first derivative to theta_i of the expected photon count over one dimension.
    """
    exp_negative = np.exp(-((coordinate - theta_i - 0.5 * pixel_size_i) ** 2) / (2 * sigma ** 2))
    exp_positive = np.exp(-((coordinate - theta_i + 0.5 * pixel_size_i) ** 2) / (2 * sigma ** 2))
    return (exp_negative - exp_positive) / (np.sqrt(2 * np.pi) * sigma)


def d2_expected_i_d_theta_i2(coordinate, theta_i, pixel_size_i, sigma):
    """
    Calculates the second derivative of 'the expected photon count in dimension i' to 'the estimated position theta_i'.

    Parameters
    ----------
    coordinate : np.array()
        A (nr_pixels, nr_pixels) array containing the 1d center coordinates of all pixels.
    theta_i : float
        An estimate for the true value of the emitter position.
    pixel_size_i : float
        The size of the pixel in this direction.
    sigma : float
        The assumed size of a PSF on the sensor.

    Returns
    -------
    np.array()
        The second derivative to theta_i of the expected photon count over one dimension.
    """
    exp_negative = np.exp(-((coordinate - theta_i - 0.5 * pixel_size_i) ** 2) / (2 * sigma ** 2))
    exp_positive = np.exp(-((coordinate - theta_i + 0.5 * pixel_size_i) ** 2) / (2 * sigma ** 2))
    return ((coordinate - theta_i - 0.5 * pixel_size_i) * exp_negative
            - (coordinate - theta_i + 0.5 * pixel_size_i) * exp_positive) / (np.sqrt(2 * np.pi) * (sigma ** 3))


def expected_count(coordinates_x, coordinates_y, theta, pixel_size, sigma):
    """
    Calculates the gradient of f(x, theta) = ln(L(x|theta)).

    Parameters
    ----------
    coordinates_x : np.array()
        (nr_pixels, nr_pixels) array containing the x coordinates of the pixel centers.
    coordinates_y : np.array()
        (nr_pixels, nr_pixels) array containing the y coordinates of the pixel centers.
    theta : np.array()
        (3,) array containing the estimated values: theta_x, theta_y and theta_i.
    pixel_size : float
        The size of a pixel.
    sigma : float
        The assumed size of a PSF on the sensor.

    Returns
    -------
    np.array()
        A (nr_pixels, nr_pixels) matrix containing the expected count per pixel.
    """
    e_x = expected_i(coordinates_x, theta[X], pixel_size, sigma)
    e_y = expected_i(coordinates_y, theta[Y], pixel_size, sigma)
    return theta[I] * e_x * e_y


def gradient(data, coordinates_x, coordinates_y, theta, pixel_size, sigma):
    """
    Calculates the gradient of f(x, theta) = ln(L(x|theta)).

    Parameters
    ----------
    data : np.array()
        The recorded number of photons per pixel during a measurement.
    coordinates_x : np.array()
        (nr_pixels, nr_pixels) array containing the x coordinates of the pixel centers.
    coordinates_y : np.array()
        (nr_pixels, nr_pixels) array containing the y coordinates of the pixel centers.
    theta : np.array()
        (3,) array containing the estimated values: theta_x, theta_y and theta_i.
    pixel_size : float
        The size of a pixel.
    sigma : float
        The assumed size of a PSF on the sensor.

    Returns
    -------
    np.array()
        A (3,) matrix containing the Jacobian for these values of theta
    """
    mu = expected_count(coordinates_x, coordinates_y, theta, pixel_size, sigma)
    prefactor = (data / mu) - 1.0

    expected_x = expected_i(coordinates_x, theta[X], pixel_size, sigma)
    expected_y = expected_i(coordinates_y, theta[Y], pixel_size, sigma)

    j_1 = np.sum(prefactor * theta[I] * expected_y * d_expected_i_d_theta_i(coordinates_x, theta[X], pixel_size, sigma))
    j_2 = np.sum(prefactor * theta[I] * expected_x * d_expected_i_d_theta_i(coordinates_y, theta[Y], pixel_size, sigma))
    j_3 = np.sum(prefactor * expected_x * expected_y)
    return np.array([j_1, j_2, j_3])


def hessian(data, coordinates_x, coordinates_y, theta, pixel_size, sigma):
    """
    Calculates the Hessian matrix for f(x, theta) = ln(L(x|theta)).

    Parameters
    ----------
    data : np.array()
        The recorded number of photons per pixel during a measurement.
    coordinates_x : np.array()
        (nr_pixels, nr_pixels) array containing the x coordinates of the pixel centers.
    coordinates_y : np.array()
        (nr_pixels, nr_pixels) array containing the y coordinates of the pixel centers.
    theta : np.array()
        (3,) array containing the estimated values: theta_x, theta_y and theta_i.
    pixel_size : float
        The size of a pixel.
    sigma : float
        The assumed size of a PSF on the sensor.

    Returns
    -------
    np.array()
        A 2x2 matrix containing the Hessian for these values of theta.
    """
    mu = expected_count(coordinates_x, coordinates_y, theta, pixel_size, sigma) + 10**(-12)
    prefactor = data / mu

    expected_x = expected_i(coordinates_x, theta[X], pixel_size, sigma)
    expected_y = expected_i(coordinates_y, theta[Y], pixel_size, sigma)

    d_expected_x_d_theta_x = d_expected_i_d_theta_i(coordinates_x, theta[X], pixel_size, sigma)
    d_expected_y_d_theta_y = d_expected_i_d_theta_i(coordinates_y, theta[Y], pixel_size, sigma)

    d_mu_d_theta_x = theta[I] * expected_y * d_expected_x_d_theta_x
    d_mu_d_theta_y = theta[I] * expected_x * d_expected_y_d_theta_y
    d_mu_d_theta_i = expected_x * expected_y

    d2_mu_d_theta_xx = theta[I] * expected_y * d2_expected_i_d_theta_i2(coordinates_x, theta[X], pixel_size, sigma)
    d2_mu_d_theta_xy = theta[I] * d_expected_x_d_theta_x * d_expected_y_d_theta_y
    d2_mu_d_theta_xi = expected_y * d_expected_x_d_theta_x
    d2_mu_d_theta_yy = theta[I] * expected_x * d2_expected_i_d_theta_i2(coordinates_y, theta[Y], pixel_size, sigma)
    d2_mu_d_theta_yi = expected_x * d_expected_y_d_theta_y

    h_11 = np.sum((prefactor - 1) * d2_mu_d_theta_xx - prefactor * d_mu_d_theta_x * d_mu_d_theta_x)
    h_12 = np.sum((prefactor - 1) * d2_mu_d_theta_xy - prefactor * d_mu_d_theta_x * d_mu_d_theta_y)
    h_13 = np.sum((prefactor - 1) * d2_mu_d_theta_xi - prefactor * d_mu_d_theta_x * d_mu_d_theta_i)
    h_21 = h_12
    h_22 = np.sum((prefactor - 1) * d2_mu_d_theta_yy - prefactor * d_mu_d_theta_y * d_mu_d_theta_y)
    h_23 = np.sum((prefactor - 1) * d2_mu_d_theta_yi - prefactor * d_mu_d_theta_y * d_mu_d_theta_i)
    h_31 = h_13
    h_32 = h_23
    h_33 = np.sum((-1) * prefactor * d_mu_d_theta_i * d_mu_d_theta_i)
    return np.array([[h_11, h_12, h_13],
                     [h_21, h_22, h_23],
                     [h_31, h_32, h_33]])


def fisher_information_matrix(coordinates_x, coordinates_y, theta, pixel_size, sigma):
    """
    Calculates the Fisher information matrix for this model, which can be used to determine the Cramer-Rao lower bound.

    Parameters
    ----------
    coordinates_x : np.array()
        (nr_pixels, nr_pixels) array containing the x coordinates of the pixel centers.
    coordinates_y : np.array()
        (nr_pixels, nr_pixels) array containing the y coordinates of the pixel centers.
    theta : np.array()
        (3,) array containing the estimated values: theta_x, theta_y and theta_i.
    pixel_size : float
        The size of a pixel.
    sigma : float
        The assumed size of a PSF on the sensor.

    Returns
    -------
    np.array()
        A 2x2 matrix containing the Fisher information matrix for this model.
    """
    mu = expected_count(coordinates_x, coordinates_y, theta, pixel_size, sigma) + 10**(-12)

    expected_x = expected_i(coordinates_x, theta[X], pixel_size, sigma)
    expected_y = expected_i(coordinates_y, theta[Y], pixel_size, sigma)

    d_expected_x_d_theta_x = d_expected_i_d_theta_i(coordinates_x, theta[X], pixel_size, sigma)
    d_expected_y_d_theta_y = d_expected_i_d_theta_i(coordinates_y, theta[Y], pixel_size, sigma)

    d_mu_d_theta_x = theta[I] * expected_y * d_expected_x_d_theta_x
    d_mu_d_theta_y = theta[I] * expected_x * d_expected_y_d_theta_y
    d_mu_d_theta_i = expected_x * expected_y

    i_11 = np.sum((1 / mu) * d_mu_d_theta_x * d_mu_d_theta_x)
    i_12 = np.sum((1 / mu) * d_mu_d_theta_x * d_mu_d_theta_y)
    i_13 = np.sum((1 / mu) * d_mu_d_theta_x * d_mu_d_theta_i)
    i_21 = i_12
    i_22 = np.sum((1 / mu) * d_mu_d_theta_y * d_mu_d_theta_y)
    i_23 = np.sum((1 / mu) * d_mu_d_theta_y * d_mu_d_theta_i)
    i_31 = i_13
    i_32 = i_23
    i_33 = np.sum((1 / mu) * d_mu_d_theta_i * d_mu_d_theta_i)
    return np.array([[i_11, i_12, i_13],
                     [i_21, i_22, i_23],
                     [i_31, i_32, i_33]])
