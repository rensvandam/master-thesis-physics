import numpy as np


def excitation_time(expected_excitation, N, random_generator):
    """
    Generates random time intervals to simulate the amount of time it takes for an emitter to reach the excited
    state. The excitation time depends on the extinction coefficient of the emitter and the power of the laser
    that illuminates the emitter. It is assumed that the wavelength of the laser equals the absorption wavelength.

    Note that the NumPy random exponential function is defined as f(x) = 1/b * exp(-x/b), where scale=b.

    Parameters
    ----------
    expected_excitation : float
        The expected time that it takes for the emitter to become excited (ns).
    N : int
        The amount of intervals that should be produced.
    random_generator : Generator
        A random number generator that produces the random distribution in this function.

    Returns
    -------
    np.array()
        (N) array of floats that shows the time in ns that it takes for the emitter to go to the excited state.
    """
    return random_generator.exponential(scale=expected_excitation, size=N)  # time in ns


def expected_excitation_time(extinction_coefficient, absorption_wavelength, laser_power):
    """
    Calculates the expected value of the excitation time of this emitter for a certain laser power.

    Parameters
    ----------
    extinction_coefficient : float
        Indicates how strongly the emitter absorbs light at the given absorption wavelength (cm^-1 M^-1).
    absorption_wavelength : float
        The wavelength at which the emitter absorbs photons (mu m).
    laser_power : int
        The power emitted by the laser (W cm^-2).

    Returns
    -------
    float
        The expected value of the excitation time (ns).
    """
    avogadro = 6.02214076 * 10 ** 23  # mol^-1
    cross_section = np.log(10) * 1000 * extinction_coefficient / avogadro  # cm^2

    c = 299792458 * 10 ** 9  # nm s^-1
    planck = 6.62607015 * 10 ** (-34)  # J s
    # TODO: it is assumed that the laser emits exactly the absorption wavelength.
    photon_flux = (10 ** (-6)) * laser_power * absorption_wavelength / (planck * c)  # ns^-1 cm^-2
    return 1 / (cross_section * photon_flux)


def emission_time(lifetime, N, random_generator):
    """
    Generates random time intervals to simulate the time that it takes for an excited emitter to fall back to the
    ground state, producing a photon. Depends on the lifetime of the emitter.

    Note that the NumPy random exponential function is defined as f(x) = 1/b * exp(-x/b), where scale=b.

    Parameters
    ----------
    lifetime : float
        The expected time that it takes for the emitter to emit a photon after excitation (ns).
    N : int
        The amount of emission time intervals that should be produced.
    random_generator : Generator
        A random number generator that produces the random distribution in this function.

    Returns
    -------
    np.array()
        (N) array of floats containing the time in ns it takes for the emitter to emit a photon, measured from the
        moment that it was excited.
    """
    return random_generator.exponential(scale=lifetime, size=N)


class Emitter:
    """
    A fluorophore at position (x, y) emitting photons at a certain wavelength.

    Attributes
    ----------
    sigma : float
        The spread of the PSF of the photons.
    photons : np.array()
        The emitted photons.
    """
    X = 0
    Y = 1
    T = 2
    NUM_APERTURE = 1

    def __init__(self, x, y, absorption_wavelength, emission_wavelength, lifetime, extinction_coefficient):
        """
        Initializes a fluorophore that is able to emit photons.

        Parameters
        ----------
        x : float
            The x coordinate of the emitter (mu m).
        y : float
            The y coordinate of the emitter (mu m).
        absorption_wavelength : float
            The wavelength at which the emitter absorbs the most photons (mu m).
        emission_wavelength : float
            The wavelength of the emitted photon (mu m).
        lifetime : float
            The lifetime of this emitter (ns).
        extinction_coefficient : int
            Indicates how strongly the emitter absorbs light at the given absorption wavelength (cm^-1 M^-1).
        """
        self.x = x
        self.y = y
        self.absorption_wavelength = absorption_wavelength
        self.emission_wavelength = emission_wavelength
        self.lifetime = lifetime
        self.extinction_coefficient = extinction_coefficient
        self.sigma = emission_wavelength / (4 * self.NUM_APERTURE)
        self.photons = np.array([])

    def generate_photons(self, laser_power, time_interval, seed=1, detection_efficiency=1, statistics="sub_poisson",
                         widefield=True):
        """
        By illuminating this emitter with a laser, the fluorophore produces photons during a time interval. This method
        generates the 2D position and the time coordinate of each photon. The x and y position are generated with a
        Gaussian distribution. The number of photons that is produced in this time interval is determined by the number
        of timestamps that fit into the interval, which follows a sub-Poissonian distribution.

        Parameters
        ----------
        laser_power : int
            The power emitted by the laser (W).
        time_interval : int
            Length of the interval in which the emitter emits photons (ns).
        seed : int, optional
            The seed for the random generator. Default is 1.
        detection_efficiency : float
            Indicates the fraction of photons that is actually detected by the sensor.
        statistics : str
            Options: "super-poisson", "poisson" or "sub_poisson". Indicates with which photon characteristics the
            photons should be generated. Default is "sub_poisson".
        widefield : bool
            Indicates if the sample is inspected in a widefield measurement (True) or with a scanning microscope
            (False). This influences the (x, y) coordinates of the photons.

        Returns
        -------
        photons : np.array()
            An (N, 3) array of n photons with x, y and t coordinates, sorted by t.
        """
        random = np.random.default_rng(seed)

        timestamps = self.__generate_timestamps(laser_power, time_interval, random, detection_efficiency, statistics)
        N = len(timestamps)
        photons = np.zeros((N, 3))

        if widefield:
            mu_x = self.x
            mu_y = self.y
        else:
            mu_x = self.x #this was 0
            mu_y = self.y #this was 0

        photons[:, self.X] = random.normal(loc=mu_x, scale=self.sigma, size=N)
        photons[:, self.Y] = random.normal(loc=mu_y, scale=self.sigma, size=N)
        photons[:, self.T] = timestamps
        self.photons = photons
        return photons

    def __generate_timestamps(self, laser_power, time_interval, random_generator, detection_efficiency, statistics):
        """
        Creates timestamps for the photons that are emitted distributed according to the requested statistics.

        Note that the NumPy random exponential function is defined as f(x) = 1/b * exp(-x/b), where scale=b.

        Parameters
        ----------
        laser_power : int
            The power emitted by the laser (W cm^-2).
        time_interval : int
            Length of the interval in which the emitter emits photons (ns).
        random_generator : Generator
            A random number generator that produces the random distribution in this function.
        detection_efficiency : float
            Indicates the fraction of photons that is actually detected by the sensor.
        statistics : str
            Options: "poisson" or "sub_poisson". Indicates with which photon characteristics the
            timestamps should be generated.

        Returns
        -------
        np.array()
            A (nr_photons,) array with timestamps, in ascending order.
        """
        expected_excitation = expected_excitation_time(self.extinction_coefficient, self.absorption_wavelength,
                                                       laser_power)
        N = 10 ** 7
        timestamps = np.array([])
        total_time = 0
        # Skip the first 10 cycles to ensure that there is no correlation between the emitters.
        start = 10 * (expected_excitation + self.lifetime)
        stop = start + time_interval

        while total_time < stop:
            new_intervals = self.__intervals(expected_excitation, N, random_generator, statistics)  # time in ns
            new_timestamps = total_time + np.cumsum(new_intervals)
            total_time = new_timestamps[-1]

            # Apply detection efficiency
            probabilities = random_generator.random(len(new_timestamps))
            selected_timestamps = new_timestamps[probabilities < detection_efficiency]
            timestamps = np.concatenate((timestamps, selected_timestamps))
            # print(f"{100 * total_time / stop}%")

        # Subtract Start to rescale timestamps in [0, interval).
        return timestamps[(timestamps >= start) & (timestamps < stop)] - start

    def __intervals(self, expected_excitation, N, random_generator, statistics):
        """
        Generates intervals between photon emissions based on a certain type of photon statistics.

        Parameters
        ----------
        expected_excitation : float
            The expected excitation time in ns.
        N : int
            The amount of intervals that should be produced.
        random_generator : Generator
            A random number generator that produces the random distribution in this function.
        statistics : str
            Options: "poisson" or "sub_poisson". Indicates with which photon characteristics the
            timestamps should be generated.

        Returns
        -------
        np.array()
            (N,) array of floats that shows the time in ns between emitter photons.
        """
        if statistics == "sub_poisson":
            return excitation_time(expected_excitation, N, random_generator)\
                   + emission_time(self.lifetime, N, random_generator)
        elif statistics == "poisson":
            return emission_time(self.lifetime, N, random_generator)
        else:
            raise ValueError(f"'{statistics}' is not a valid value for statistics")


class Alexa647(Emitter):
    """
    An Alexa 647 fluorophore placed at position (x, y). This fluorophore has specific properties that are the same for
    every instance of this molecule.
    """

    def __init__(self, x, y):
        super().__init__(x, y, absorption_wavelength=0.650, emission_wavelength=0.665, lifetime=1,
                         extinction_coefficient=240000)