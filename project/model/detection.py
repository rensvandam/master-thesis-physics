from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize
from project.model.helper_functions import merge_k_2D, insertion_sort_2D


def show_photons(photons, is_detected=None, ax=None):
    """
    Creates a scatter plot of the photons. Optional: show which are detected and which not.

    Parameters
    ----------
    photons : np.ndarray
        A (number of photons, 3) array containing the (x, y, t) coordinates of each photon, sorted by time.
    is_detected : np.ndarray
        A (number of photons, ) array that indicates which photon was detected (1) and which not (0). Default is None.
    """
    if ax:
        plt.sca(ax)
    if is_detected is None:
        color = 'black'
    else:
        color_map = np.array(['red', 'black'])
        color = color_map[is_detected.astype(int)]
    plt.scatter(photons[:, 0], photons[:, 1], color=color, s=5, alpha=0.2)


def merge_photons(emitters):
    """
    Merges all photons that are emitted by these fluorophores into one array, which is sorted in ascending order by
    time.

    Parameters
    ----------
    emitters : list of Emitter
        A list of fluorophores that possibly emitted photons.

    Returns
    -------
    np.ndarray
        A (number of photons, 3) array containing the (x, y, t) coordinate of every photon that was emitted by the
        input Emitters.
    """
    return merge_k_2D([emitter.photons for emitter in emitters], sort_by_index=2)


class Sensor(ABC):
    """
    A sensor with a number of single-photon detecting pixels.

    Attributes
    ----------
    magnification : float
        The magnification of the measurement setup.
    nr_pixels : int
        The number of pixels on the sensor.
    pixel_radius : float
        The radius of the active detection area for each pixel (mu m).
    spacing : float
        The distance between the midpoints of two pixels (mu m).
    crosstalk : float
        The crosstalk probability (in the interval [0, 1]).
    afterpulsing : float
        The afterpulsing probability (in the interval [0, 1]).
    jitter : float
        The timing jitter on the photon timestamps at FWHM (ns).
    dead_time : float
        The dead time of all pixels (assumed to be uniform) (ns).
    dark_count_rate : float
        The dark count rate per pixel. (cps)
    pixel_coordinates : np.ndarray
        A (number of pixels, 2) array that the midpoint coordinates (x, y) of each pixel.
    x_limits : tuple
        The min, max coordinates of the sensor in the x direction.
    y_limits : tuple
        The min, max coordinates of the sensor in the y direction.
    data_by_time : np.ndarray
        A (number of photons, 2) array showing the (pixel number, t) coordinates of each measured photon. Is sorted by
        time.
    data_per_pixel : dict
        Maps pixel number to a 1D array of timestamps of measured photons for that pixel (int -> np.ndarray).
    photon_count : np.ndarray
        A (number of pixels, ) array that stores the number of photon detections per pixel.
    """

    def __init__(self, magnification, nr_pixels, pixel_radius, spacing, crosstalk, afterpulsing, jitter, dead_time,
                 dark_count_rate):
        self.magnification = magnification
        self.nr_pixels = nr_pixels
        self.pixel_radius = pixel_radius
        self.spacing = spacing
        self.crosstalk = crosstalk
        self.afterpulsing = afterpulsing
        self.jitter = jitter
        self.dead_time = dead_time
        self.dark_count_rate = dark_count_rate

        self.pixel_coordinates = self._calculate_pixel_coordinates()

        # Find sensor boundaries by taking the bottom left and top right pixel coordinates and add half of the pixel
        # spacing.
        self.x_limits = (self.pixel_coordinates[0, 0] - self.pixel_radius,
                         self.pixel_coordinates[-1, 0] + self.pixel_radius)
        self.y_limits = (self.pixel_coordinates[0, 1] - self.pixel_radius,
                         self.pixel_coordinates[-1, 1] + self.pixel_radius)

        self.data_by_time = np.array([])
        self.data_per_pixel = {k: [] for k in range(self.nr_pixels)}
        self.photon_count = np.zeros(nr_pixels)

    def magnify(self, photons, debug=None):
        """
        Magnifies the input photons by the magnification factor.
        
        Parameters
        ----------
        photons : np.ndarray
            A (number of photons, 3) array containing the (x, y, t) coordinates of each photon, sorted by time.
        debug : bool
            If True, information about the magnification process is printed. Default is None.

        Returns
        -------
        np.ndarray
            A (number of photons, 3) array containing the (x, y, t) coordinates of each photon, sorted by time.
        """
        if debug:
            print(f"DEBUG for magnification:")
            print(f"Magnification factor: {self.magnification}")
            print(f"10 random photons before magnification: {photons[np.random.choice(len(photons), 10)]}")

        magnified = photons.copy()
        x_center = (self.x_limits[0] + self.x_limits[1]) / 2
        magnified[:, 0] = self.magnification * (photons[:, 0] - x_center) + x_center

        y_center = (self.y_limits[0] + self.y_limits[1]) / 2
        magnified[:, 1] = self.magnification * (photons[:, 1] - y_center) + y_center
    
        if debug:
            print(f"10 random photons after magnification: {magnified[np.random.choice(len(magnified), 10)]}")
        return magnified

    def measure(self, photons, duration, seed, 
                enable_dark_counts = True, 
                enable_timestamp_jitter = True, 
                enable_deadtime = True, 
                enable_afterpulsing = True, 
                enable_crosstalk = True, 
                debug=None):
        """
        Simulates a measurement given the incident photons during a certain measurement interval.

        Parameters
        ----------
        photons : np.ndarray
            A (number of photons, 3) array containing the (x, y, t) coordinate of all photons. Is sorted in ascending
            order by time.
        duration : float
            The time during which photons are measured. (ns)
        seed : int
            The seed for the random number generator (used for noise sources).
        enable_deadtime : bool, optional
            Whether to apply dead time effect. Default is True.
        enable_afterpulsing : bool, optional
            Whether to apply afterpulsing effect. Default is True.
        enable_crosstalk : bool, optional
            Whether to apply crosstalk effect. Default is True.
        debug : bool
            If True, information about the measuring process is printed. Default is None.

        Returns
        -------
        np.ndarray
            A (number of detected photons, 2) array showing the (pixel number, t) coordinates of each measured photon,
            sorted by time.
        np.ndarray
            A (number of photons, ) array that indicates which photon hits a pixel (True) and which not (False).
        """
        if len(photons) == 0:
            return self.data_by_time

        # Check sensor geometry.
        is_on_sensor = self._discard_off_limit_photons(photons, debug=debug)
        self.data_by_time, is_on_pixel = self._project_onto_pixels(photons)

        # Discard photons that are not on pixels.
        is_on_active_pixel = is_on_sensor * is_on_pixel
        self.data_by_time = self.data_by_time[is_on_active_pixel]

        if len(self.data_by_time) == 0:
            return self.data_by_time, is_on_active_pixel

        if debug:
            photons_arriving_on_pixels = self.data_by_time.copy()
            print(f"Number of photons on pixels: {len(self.data_by_time)}")

        rng = np.random.default_rng(seed)

        # Add dark counts.
        # TODO: pay attention! As the dark counts are added as 'dark photons', some of them will go undetected in the
        #  detection pipeline. As a result, the effective dark count rate of the simulation may be lower than the actual
        
        if enable_dark_counts:
            #  dark count rate of the sensor.
            self._add_dark_counts(duration, rng)
        if enable_timestamp_jitter:
            # Add jitter to the timestamps.
            self._apply_jitter(rng)

        # Apply dead time, afterpulsing and crosstalk.
        self._detection_pipeline(rng, enable_deadtime=enable_deadtime, enable_afterpulsing=enable_afterpulsing, enable_crosstalk=enable_crosstalk)

        # Sort the measurement again (jitter and afterpulsing may have changed the order).
        self.data_by_time = insertion_sort_2D(self.data_by_time, sort_by_index=1)

        if debug:
            print(f"Number of photons after detection pipeline: {len(self.data_by_time)}")

        # Store the number of photon detections per pixel and convert the timestamp lists per pixel to Numpy arrays.
        for pixel_number in range(self.nr_pixels):
            self.photon_count[pixel_number] = len(self.data_per_pixel[pixel_number])
            self.data_per_pixel[pixel_number] = np.array(self.data_per_pixel[pixel_number])

        if debug:
            return self.data_by_time, is_on_active_pixel, photons_arriving_on_pixels
        return self.data_by_time, is_on_active_pixel

    def clear(self):
        """
        Removes all measured data.
        """
        self.data_by_time = np.array([])
        self.data_per_pixel = {k: [] for k in range(self.nr_pixels)}
        self.photon_count = np.zeros(self.nr_pixels)

    def show(self, data_to_show=None, title=None, save_as=None, vmin=None, vmax=None, color_map='viridis', ax=None):
        """
        Creates a plot that shows the input data on each pixel.

        Parameters
        ----------
        data_to_show : np.ndarray
            A (number of pixels, ) array that contains the data that is shown on each pixel. If None, the photon count
            per pixel is shown (default).
        title : str
            The tile of the plot. Default is None.
        save_as
            The filepath for saving the plot. Default is None.
        vmin : float
            The minimum value of the color bar. Default is None.
        vmax : float
            The maximum value of the color bar. Default is None.
        color_map : str
            The colormap that is used in the plot. Default is viridis.
        """
        if data_to_show is None:
            data_to_show = self.photon_count
        if vmin is None:
            vmin = np.amin(data_to_show)
        if vmax is None:
            vmax = np.amax(data_to_show)
        if vmin == vmax:
            vmin -= 1
            vmax += 1

        cmap = plt.get_cmap(color_map)
        norm = Normalize(vmin, vmax, clip=True)
        colors = norm(data_to_show)

        if ax is None:
            fig, ax = plt.subplots()
        offsets = list(zip(self.pixel_coordinates[:, 0], self.pixel_coordinates[:, 1]))
        ax.add_collection(EllipseCollection(widths=self.pixel_radius * 2, heights=self.pixel_radius * 2, angles=0,
                                            units='xy', facecolors=cmap(colors), offsets=offsets,
                                            transOffset=ax.transData))
        ax.set_xlim((self.x_limits[0] - self.spacing / 2, self.x_limits[1] + self.spacing / 2))
        ax.set_ylim((self.y_limits[0] - self.spacing / 2, self.y_limits[1] + self.spacing / 2))



        ax.set_aspect('equal', adjustable='box')
        fig = ax.get_figure()  # Get the parent figure of the axis
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        ax.set_xlabel(r"x [$\mu m]$", fontsize=20)
        ax.set_ylabel(r"y [$\mu m]$", fontsize=20)
        ax.set_title(title)
        if save_as:
            plt.savefig(save_as)
        # plt.show()

    def _discard_off_limit_photons(self, photons, debug=None):
        """
        Discards any photons with coordinates that are not on the sensor surface.

        Parameters
        ----------
        photons : np.ndarray
            A (number of photons, 3) array containing the (x, y, t) coordinates of each photon, sorted by time.

        Returns
        -------
        np.ndarray
            A (number of photons, ) boolean array. True if the photon is on the sensor surface, false otherwise.
        """
        if debug:
            print(f"Before discarding off limit photons: {len(photons)} photons.")
            print(f"Sensor limits: x = {self.x_limits}, y = {self.y_limits}")
        on_sensor = ((photons[:, 0] >= self.x_limits[0]) & (photons[:, 0] <= self.x_limits[1]) &
                          (photons[:, 1] >= self.y_limits[0]) & (photons[:, 1] <= self.y_limits[1]))
        if debug:
            photons_on_sensor = photons.copy()[on_sensor]
            print(f"After discarding off limit photons: {len(photons_on_sensor)} photons.")

        return on_sensor
    

    def _apply_jitter(self, rng):
        """
        Adds timing jitter to the timestamps of the detected photons.

        Parameters
        ----------
        rng : np.random.Generator
            Seeded random number generator.
        """
        sigma = self.jitter / (2 * np.sqrt(2 * np.log(2)))
        noise = rng.normal(loc=0, scale=sigma, size=len(self.data_by_time))
        self.data_by_time[:, 1] += noise

    def _add_dark_counts(self, duration, rng):
        """
        Adds dark photons to the measurement. The number of dark counts is drawn from a Poisson distribution, where the
        expected value equals the dark count rate per pixel multiplied by the number of pixels and the measurement time.

        Parameters
        ----------
        duration : float
            The time interval of the measurement. (ns)
        rng : np.random.Generator
            Seeded random number generator.

        Returns
        -------
        np.ndarray
            A (number of photons, 2) array containing the pixel number and the timestamp for each photon.
        """
        nr_dark_counts = rng.poisson(self.dark_count_rate * self.nr_pixels * duration * 10**(-9))
        dark_photons = np.empty((nr_dark_counts, 2))
        dark_photons[:, 0] = rng.choice(range(self.nr_pixels), nr_dark_counts)
        dark_photons[:, 1] = sorted(rng.random(nr_dark_counts) * duration)
        self.data_by_time = merge_k_2D([self.data_by_time, dark_photons], sort_by_index=1)
        return self.data_by_time


    def _detection_pipeline(self, rng, enable_deadtime=True, enable_afterpulsing=True, enable_crosstalk=True):
        """
        Applies afterpulsing, crosstalk and dead time to obtain the final measurement.
        Parameters
        ----------
        rng : np.random.Generator
            Seeded random number generator.
        enable_deadtime : bool, optional
            Whether to apply dead time effect. Default is True.
        enable_afterpulsing : bool, optional
            Whether to apply afterpulsing effect. Default is True.
        enable_crosstalk : bool, optional
            Whether to apply crosstalk effect. Default is True.
        Returns
        -------
        np.ndarray
            A (number of detected photons, 2) array showing the (pixel number, t) coordinates of each measured photon,
            sorted by time.
        """
        number_of_photons = len(self.data_by_time)
        result = []
        # Store the time from which moment on the pixel is available to detect a photon.
        status = np.ones(self.nr_pixels) * -1
        # Generate arrays with random numbers to determine if there is afterpulsing and/or crosstalk.
        afterpulsing = rng.random(number_of_photons) # Generate random numbers (0,1) for each photon
        crosstalk = rng.random(number_of_photons)

        # Go over each photon
        photon_index = 0
        while photon_index < number_of_photons:
            current = self.data_by_time[photon_index, :]
            # Get the pixel number of where the photon hit and the timestamp of the photon.
            pixel_number = int(current[0])
            timestamp = current[1]
            if not enable_deadtime or status[pixel_number] < timestamp:
                # This triggers if deadtime is off or if the pixel is available to detect a photon.
                # So: the pixel detects the photon.

                ### DEADTIME ###
                if enable_deadtime:
                    result.append([pixel_number, timestamp])
                    #np.append(self.data_per_pixel[pixel_number], timestamp)
                    self.data_per_pixel[pixel_number].append(timestamp)
                    #print(f"{timestamp} appended to {self.data_per_pixel[pixel_number]}")
                    # # The pixel will be available again in t + dead time seconds.
                    status[pixel_number] = timestamp + self.dead_time

                    #### Test with variation in dead time
                    # # Add random variation to dead time (±10%)
                    # import random
                    # variation_factor = random.uniform(0.9, 1.1)  # Random factor between 0.9 and 1.1
                    # varied_dead_time = self.dead_time * variation_factor
                    
                    # # The pixel will be available again in t + varied dead time seconds
                    # status[pixel_number] = timestamp + varied_dead_time
                else:
                    result.append([pixel_number, timestamp])
                    self.data_per_pixel[pixel_number].append(timestamp)

                ### AFTERPULSING ###
                if enable_afterpulsing and afterpulsing[photon_index] < self.afterpulsing:
                    # Create an afterpulsing event.
                    result.append([pixel_number, status[pixel_number] if enable_deadtime else timestamp])
                    self.data_per_pixel[pixel_number].append(status[pixel_number] if enable_deadtime else timestamp)
                    if enable_deadtime:
                        status[pixel_number] += self.dead_time

                ### CROSSTALK ###
                if enable_crosstalk and crosstalk[photon_index] < self.crosstalk:
                    # Create a crosstalk event
                    neighbor_pixel = self._select_neighbor_pixel(pixel_number, rng)
                    # TODO: if the selected neighbor pixel is not available, then choose another or just let it be?
                    # TODO: edge pixels have less neighbors. Because of that, the same neighbors will be selected more
                    #  often than for center pixels.
                    # TODO: should the timestamp contain jitter?
                    if not enable_deadtime or status[neighbor_pixel] < timestamp:
                        result.append([neighbor_pixel, timestamp])
                        self.data_per_pixel[neighbor_pixel].append(timestamp)
                        if enable_deadtime:
                            status[neighbor_pixel] = timestamp + self.dead_time
            photon_index += 1
        self.data_by_time = np.array(result)
        return self.data_by_time

    @abstractmethod
    def _project_onto_pixels(self, photons, debug):
        """
        Projects the photons onto the sensor pixels and registers which photon hits which pixel.

        Parameters
        ----------
        photons : np.ndarray
            A (number of photons, 3) array containing the (x, y, t) coordinates of each photon, sorted by time.

        Returns
        -------
        np.ndarray
            A (number of detected photons, 2) array showing the (pixel number, t) coordinates of each photon that hit
            the active pixel area, sorted by time.
        np.ndarray
            A (number of photons, ) array that indicates which photon was detected (True) and which not (False).
        """
        pass

    @abstractmethod
    def _select_neighbor_pixel(self, pixel_number, rng):
        """
        Takes a pixel number and returns the pixel number of a neighboring pixel.

        Parameters
        ----------
        pixel_number : int
            The number of the current pixel.
        rng : np.random.Generator
            Seeded random number generator.

        Returns
        -------
        int
            The pixel number of a randomly selected direct neighbor of the current pixel.
        """
        pass

    @abstractmethod
    def _calculate_pixel_coordinates(self):
        """
        Calculates the midpoint coordinates (x, y) of each pixel on the sensor.

        Returns
        -------
        np.ndarray
            A (number of pixels, 2) array that maps the pixel number to the midpoint coordinates (x, y) of the pixel.
        """
        pass


class Spad23(Sensor):
    """
    Attributes
    ----------
    nr_pixel_rows : int
        The number of rows with pixels in the array. Should be odd, default is 5 (as in the Spad23).
    __hexagonal_base_vectors : np.ndarray
        The base vectors of the coordinate system that is used to describe the hexagonal pixel placement.
    __neighbors : dict
        Maps the pixel number to the direct neighbors of that pixel (int -> list[int]).
    """

    def __init__(self, magnification=1, nr_pixel_rows=5, pixel_radius=10.3, spacing=23, crosstalk=0.0014,
                 afterpulsing=0.001, jitter=0.120, dead_time=50, dark_count_rate=100):
        # Define the base vectors of the hexagonal grid.
        self.__hexagonal_base_vectors = np.array([[1, 0.5], [0, 0.5 * np.sqrt(3)]]) * spacing

        # Calculate the number of pixels.
        self.nr_pixel_rows = nr_pixel_rows
        if nr_pixel_rows % 2 != 1:
            raise ValueError(f"The number of pixel rows should be odd. {nr_pixel_rows} != odd")
        nr_pixels = int(nr_pixel_rows ** 2 - nr_pixel_rows // 2)

        # Initialize other parameters using the parent class (Sensor).
        super().__init__(magnification, nr_pixels, pixel_radius, spacing, crosstalk, afterpulsing, jitter, dead_time,
                         dark_count_rate)

        # Find the neighbors of each pixel and store them.
        self.__neighbors = self.__find_neighbors()

    def _project_onto_pixels(self, photons):
        # Convert the (x, y) coordinates to (a, b) coordinates. The hexagonal base now defines the coordinate system.
        coordinates_xy = photons[:, :2]
        transformation_matrix = np.transpose(np.linalg.inv(self.__hexagonal_base_vectors))
        coordinates_ab = np.matmul(coordinates_xy, transformation_matrix)

        # Add 0.5 to facilitate finding the closest pixel.
        coordinates_ab += 0.5
        quotient, remainder = np.divmod(coordinates_ab, 1)

        # Calculate the pixel number by using that:
        #   1) the sensor has 5 rows of pixels
        #   2) the pixel that is on location (-1, -2)_ab should have pixel number 0. The center pixel is number 11,
        #      which is on (0, 0)_xy and (0, 0)_ab. This choice was made so that the coordinate system aligns with the
        #      center pixel, while at the same time, the pixel numbers are strictly positive.
        pixel_number = (quotient[:, 1] + 2) * 5 + quotient[:, 0] + 1

        # Check if the photons hit the active surface of the pixel that they are closest to.
        remainder -= 0.5
        remainder_xy = np.matmul(remainder, self.__hexagonal_base_vectors.T)

        # Create a mask to select only those photons that hit the active surface of a pixel.
        is_on_pixel = (remainder_xy[:, 0] * remainder_xy[:, 0] + remainder_xy[:, 1] * remainder_xy[:, 1]
                       <= self.pixel_radius * self.pixel_radius)
        is_false_positive = (
                (((quotient[:, 0] == -3) | (quotient[:, 0] == 2)) & (quotient[:, 1] == 1))
                | (((quotient[:, 0] == -2) | (quotient[:, 0] == 3)) & (quotient[:, 1] == -1))
                | (quotient[:, 1] == 3)
                | (quotient[:, 1] == -3)
        )
        is_detected = is_on_pixel * (~is_false_positive)

        # Store the pixel numbers
        self.data_by_time = np.empty((np.shape(photons)[0], 2))
        self.data_by_time[:, 0] = pixel_number
        self.data_by_time[:, 1] = photons[:, 2]
        return self.data_by_time, is_detected

    def _select_neighbor_pixel(self, pixel_number, rng):
        return rng.choice(self.__neighbors[pixel_number])

    def _calculate_pixel_coordinates(self):
        pixel_coordinates = np.zeros((self.nr_pixels, 2))

        # Loop through all pixel indices on the hexagonal grid.
        for ind2 in range(self.nr_pixel_rows):
            cols = (np.arange(0, self.nr_pixel_rows - ind2 % 2) - ind2 // 2).astype(int)
            for ind1 in cols:
                pixel_number = ind2 * self.nr_pixel_rows + ind1

                # Calculate the pixel coordinates such that the origin is in the middle of the center pixel.
                pixel_coordinates[pixel_number, :] = self.__hexagonal_base_vectors @ np.array([ind1 - 1, ind2 - 2])
        return pixel_coordinates

    def __find_neighbors(self):
        """
        Finds the direct neighbors for each pixel. Each pixel can have up to 6 neighbors on the hexagonal grid.

        Returns
        -------
        dict
            Maps the pixel number to the direct neighbors of that pixel (int -> list[int]).
        """
        neighbor_dict = {}
        for pixel_number in range(self.nr_pixels):
            neighbors = []
            if pixel_number not in (0, 5, 9, 14, 18):
                neighbors.append(pixel_number - 1)  # left
            if pixel_number <= 17 and pixel_number not in (0, 9):
                neighbors.append(pixel_number + 4)  # upper left
            if pixel_number <= 17 and pixel_number not in (4, 13):
                neighbors.append(pixel_number + 5)  # upper right
            if pixel_number not in (4, 8, 13, 17, 22):
                neighbors.append(pixel_number + 1)  # right
            if pixel_number >= 5 and pixel_number not in (13, 22):
                neighbors.append(pixel_number - 4)  # bottom right
            if pixel_number >= 5 and pixel_number not in (9, 18):
                neighbors.append(pixel_number - 5)  # bottom left
            neighbor_dict[pixel_number] = neighbors
        return neighbor_dict


class Spad512(Sensor):
    def __init__(self, magnification=1, nr_pixel_rows=512, nr_pixel_columns=512, pixel_radius=5, spacing=16.38, crosstalk=0,
                 afterpulsing=0, jitter=0, dead_time=0, dark_count_rate=25):
        self.nr_pixel_rows = nr_pixel_rows
        self.nr_pixel_columns = nr_pixel_columns
        nr_pixels = nr_pixel_rows * nr_pixel_columns
        super().__init__(magnification, nr_pixels, pixel_radius, spacing, crosstalk, afterpulsing, jitter, dead_time,
                         dark_count_rate)

    def _project_onto_pixels(self, photons):
        coordinates = photons[:, :2].copy() / self.spacing
        coordinates += 0.5
        quotient, remainder = np.divmod(coordinates, 1)

        # Calculate the number of the pixel that each photon is closed to.
        pixel_number = quotient[:, 1] * self.nr_pixel_rows + quotient[:, 0]

        # Check if the photons are within pixel area.
        remainder = (remainder - 0.5) * self.spacing
        is_detected = (remainder[:, 0] * remainder[:, 0] + remainder[:, 1] * remainder[:, 1]
                       <= self.pixel_radius * self.pixel_radius)

        # Store the results
        self.data_by_time = np.empty((np.shape(photons)[0], 2))
        self.data_by_time[:, 0] = pixel_number
        self.data_by_time[:, 1] = photons[:, 2]
        return self.data_by_time, is_detected

    def _select_neighbor_pixel(self, pixel_number, rng):
        # TODO: test this method.
        neighbors = []
        if pixel_number >= self.nr_pixel_rows:
            neighbors.append(pixel_number - self.nr_pixel_rows)  # down
        if pixel_number // self.nr_pixel_rows != 0:
            neighbors.append(pixel_number - 1)  # left
        if pixel_number < (self.nr_pixels - self.nr_pixel_rows):
            neighbors.append(pixel_number + self.nr_pixel_rows)  # up
        if pixel_number // self.nr_pixel_rows != self.nr_pixel_rows:
            neighbors.append(pixel_number + 1)  # right
        return rng.choice(neighbors)

    def _calculate_pixel_coordinates(self):
        return (np.array([[i % self.nr_pixel_rows, i // self.nr_pixel_columns] for i in range(self.nr_pixels)])
                * self.spacing)

