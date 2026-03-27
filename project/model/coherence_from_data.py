import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import convolve2d
from project.model.helper_functions import sparse_convolution, average_nn, select_neighbours, merge_k, get_neighbors
from project.model.sample import expected_excitation_time
from project.model.sample import Alexa647


def discrete_deviation_coherence(bin_size, k):
    """
    Calculates the deviation factor added to coherence due to discretization.

    Parameters
    ----------
    bin_size : float
        The length of one bin. (ns)
    k : float
        The sum of excitation and emission rate of a fluorophore. (ns^-1)

    Returns
    -------
    float
        The extra factor that influences coherence through g^(2)[0]=1-(1/n)*factor.
    """
    return (2 / (k * bin_size)) * (1 - (1 - np.exp(-k * bin_size)) / (k * bin_size))


def expected_number_of_emitters(measured_coherence, bin_size, excitation_rate, emission_rate):
    """
    Calculates the number of emitters corresponding to a measured coherence value.

    Parameters
    ----------
    measured_coherence : float
        The value of the second-order quantum coherence at lag 0.
    bin_size : float
        The length of one bin. (ns)
    excitation_rate : float
        The excitation rate of one fluorophore. (ns^-1)
    emission_rate : float
        The emission rate of one fluorophore. (ns^-1)

    Returns
    -------
    float
        The number of emitters associated with this coherence.
    """
    k = excitation_rate + emission_rate
    return discrete_deviation_coherence(bin_size, k) / (1 - measured_coherence)

def coherence(signal1, signal2, interval, bin_size, nr_steps, offset=0, normalize=True, auto_correlation=False):
    """
    Calculates the second-order quantum coherence of the measured photons in a certain interval on different pixels.

    Parameters
    ----------
    signal1, signal2 : np.array()
        A signal consisting of hits in a certain interval.
    interval : float
        The time interval in which the photons were measured.
    bin_size : float
        The size of the sub-intervals that split up the data in order to create a discrete time signal.
    nr_steps : int
        The number of steps across which the coherence is calculated.
    offset : int
        Indicates from which number of steps on the coherence should be calculated. Default is 0, so that the coherence
        is calculated for steps [0, nr_steps).
    normalize : bool
        If True, the coherence is normalized. If False, the values that are returned are the photon counts per lag,
        corrected for bias. Default is True.
    auto_correlation : bool
        If True, a signal is correlated with itself. The found number of pairs then needs to be adjusted so that photons
        do not form a pair with themselves.

    Returns
    -------
    correlation : np.array()
        (nr_steps,) array containing the photon pairs per shift.
    lag : np.array()
        (nr_steps,) array indicating the amount of delay in the signal shift.
    """
    # Calculate the bin indices of the photon hitting times.
    photon_indices1 = np.floor(signal1 / bin_size).astype(np.int64)
    photon_indices2 = np.floor(signal2 / bin_size).astype(np.int64)
    photon_pairs = sparse_convolution(photon_indices1, photon_indices2, nr_steps, offset)

    if auto_correlation and offset == 0:
        photon_pairs[0] -= len(signal1)  # subtract the photons that form a pair with themselves

    # Correct for bias. This is necessary if you use an offset or use multipe (>1) steps.
    m = int(interval / bin_size) # Number of bins in the interval.
    bias = np.arange(m - offset, m - offset - nr_steps, -1)
    correlation = photon_pairs * (m / bias)

    # Normalize in order to obtain the correlation from the photon pair count.
    if normalize and len(signal1) != 0 and len(signal2) != 0:
        correlation = correlation * m / (len(signal1) * len(signal2))
        # m / (len(signal1) * len(signal2)) is the 'probability' of two photons over the whole interval landing in the same bin.

    bins = np.arange(offset, nr_steps + offset) * bin_size
    return correlation, bins


def auto_coherence(signal, interval, bin_size, nr_steps, offset=0, normalize=True):
    """
    Calculates the second-order quantum coherence of a signal with itself.

    Parameters
    ----------
    signal : np.array()
        A signal consisting of photon hits on the detector in a certain interval.
    interval : float
        The time interval in which the photons were measured.
    bin_size : float
        The size of the sub-intervals that split up the data in order to create a discrete time signal.
    nr_steps : int
        The number of steps across which the coherence is calculated.
    offset : int
        Indicates from which number of steps on the coherence should be calculated. Default is 0, so that the coherence
        is calculated for steps [0, nr_steps).
    normalize : bool
        If True, the coherence is normalized such that g2(inf) ~= 1. If False, the values that are returned are the
        photon counts per lag. Default is True.

    Returns
    -------
    correlation : np.array()
        (nr_steps,) array containing the photon pairs per shift.
    lag : np.array()
        (nr_steps,) array indicating the amount of delay in the signal shift.
    """
    return coherence(signal, signal, interval, bin_size, nr_steps, offset, normalize, auto_correlation=True)

def coherence_per_neighbourhood(sensor, interval, bin_size, kernel_size=1, nr_steps=200):
    """
    Calculates the second-order quantum coherence of the signal on a pixel with the neighbouring pixels.
    Method: add up all arrival times over a neighbourhood and calculate the autocoherence of that signal.

    Parameters
    ----------
    sensor : Sensor
        The sensor on which the measurements were performed.
    interval : float
        The time interval in which the photons were measured.
    bin_size : float
        The size of the sub-intervals that split up the data in order to create a discrete time signal.
    kernel_size : int
        The degree of neighbours that should be included in the average. Default is 1.
    nr_steps : int
        The number of steps across which the coherence is calculated.
        
    Returns
    -------
    coherence : np.array()
        (nr_steps,) array: The coherence of a neighborhood around every pixel
    lag : np.array()
        (nr_steps,) array: The amount of delay in the signal shift.
    """
    nr_pixels = sensor.nr_pixels
    result = np.zeros((nr_pixels, nr_steps))

    # For hexagonal SPAD23 sensor
    # Access the private attribute using name mangling
    if kernel_size == 1 and hasattr(sensor, '_Spad23__neighbors'):
        neighbors_dict = getattr(sensor, '_Spad23__neighbors')
    else: 
        print("Error: for hexagonal SPAD23 sensors, kernel_size is only available for 1 ring of neighbours.")

    pixel_neighborhood_coherence_array = []
    for i in range(nr_pixels):
        # Get the neighbours of the pixel
        neighbor_indices = neighbors_dict[i]
        #print("Neighbours of pixel", i, ":", neighbor_indices)
        list_of_arrival_times = [sensor.data_per_pixel[index] for index in neighbor_indices if index in sensor.data_per_pixel]
        if list_of_arrival_times:
            signal = merge_k(list_of_arrival_times)
        else:
            signal = np.array([])

        pixel_neighborhood_coherence, bins = auto_coherence(signal, interval, bin_size, nr_steps, normalize=True)
        pixel_neighborhood_coherence_array.append(pixel_neighborhood_coherence)
    
    return pixel_neighborhood_coherence_array, bins

def auto_coherence_per_neighbourhood(sensor, interval, bin_size, kernel_size=1):
    """
    Calculates the second-order coherence at lag 0 of all photons detected by pixels in every neighborhood.

    Parameters
    ----------
    sensor : Sensor
        The sensor of which the coherence is calculated.
    interval : float
        The time interval in which the photons were measured.
    bin_size : float
        The size of the sub-intervals that split up the data in order to create a discrete time signal.
    kernel_size : int
        For hexagonal grids, this defines how many "rings" of neighbors to include.
        E.g., 1 means just direct neighbors, 2 includes neighbors of neighbors, etc.
    Returns
    -------
    np.array()
        The auto-coherence of a neighborhood around every pixel.
    """
    nr_pixels = sensor.nr_pixels
    result = np.zeros(nr_pixels)
    
    # For hexagonal SPAD23 sensor
    if hasattr(sensor, '_Spad23__neighbors'):
        # Access the private attribute using name mangling
        neighbors_dict = getattr(sensor, '_Spad23__neighbors')
        
        for i in range(nr_pixels):
            # Start with the pixel itself
            neighbor_indices = [i]
            
            # Add direct neighbors
            if i in neighbors_dict:
                neighbor_indices.extend(neighbors_dict[i])
                
            # If kernel_size > 1, add more rings of neighbors
            if kernel_size > 1:
                expanded_neighbors = neighbor_indices.copy()
                for _ in range(kernel_size - 1):
                    temp_neighbors = []
                    for n in expanded_neighbors:
                        if n in neighbors_dict:
                            temp_neighbors.extend([nn for nn in neighbors_dict[n] 
                                                if nn not in neighbor_indices])
                    neighbor_indices.extend(temp_neighbors)
                    expanded_neighbors = temp_neighbors
            
            # Get arrival times and calculate coherence
            list_of_arrival_times = [sensor.data_per_pixel[index] for index in neighbor_indices 
                                    if index in sensor.data_per_pixel]
            if list_of_arrival_times:
                signal = merge_k(list_of_arrival_times)
            else:
                signal = np.array([])
            
            result[i] = auto_coherence(signal, interval, bin_size, nr_steps=1, offset=0, normalize=True)[0]
    
    # For rectangular grid sensors, use the original method
    else:
        for i in range(nr_pixels):
            neighbor_indices = select_neighbours(kernel_size, i, sensor.nr_pixel_rows, sensor.nr_pixel_columns)
            list_of_arrival_times = [sensor.data_per_pixel[index] for index in neighbor_indices 
                                    if index in sensor.data_per_pixel]
            if list_of_arrival_times:
                signal = merge_k(list_of_arrival_times)
            else:
                signal = np.array([])
            
            result[i] = auto_coherence(signal, interval, bin_size, nr_steps=1, offset=0, normalize=True)[0]
    
    # The original reshape won't work for SPAD23 due to its hexagonal layout
    # For SPAD23, return a flat array instead of trying to reshape
    if hasattr(sensor, 'nr_pixel_columns'):
        return np.reshape(result, (sensor.nr_pixel_rows, sensor.nr_pixel_columns))
    else:
        return result
    
def pixel_to_pixel_coherence(sensor, interval, bin_size, lag, normalize=True):
    """
    Calculates the 2nd order coherence at at the specified lag between every pixel of the sensor.

    Parameters
    ----------
    sensor : Sensor
        The sensor of which the coherence is calculated.
    interval : int
        The time interval in which a measurement was recorded.
    bin_size : float
        The size of the bins in which the time interval is split up.
    lag : int
        The lag at which the coherence is calculated.
    normalize : bool
        Indicates whether the coherence should be normalized or not. If True, the coherence equals g^2(0). If False, the
        coherence equals the amount of photon pairs at lag 0, G^2(0)
        

    Returns
    -------
    c_array : np.array()
        An array of all pixels x all pixels where c_array[x, y] equals the coherence between pixel number x and pixel
        number y at the specified lag.
    """
    nr_pixels = sensor.nr_pixels
    c_array = np.zeros((nr_pixels, nr_pixels))

    if len(sensor.data_per_pixel) == 0:
        return c_array

    for i in range(nr_pixels):
        #print(f"Pixel number: {i + 1}/{nr_pixels}")
        if i not in sensor.data_per_pixel:
            continue
        data1 = sensor.data_per_pixel[i]
        for j in range(nr_pixels):
            if j not in sensor.data_per_pixel:
                continue
            data2 = sensor.data_per_pixel[j]
            c, _ = coherence(data1, data2, interval, bin_size, 100, lag, normalize, auto_correlation=bool(i == j))
            c_array[i, j] = c[0]
    return c_array


def denoised_auto_coherence(sensor, interval, bin_size, degree):
    """
    Calculates the auto-coherence of each pixel. By averaging the number of photon pairs over the neighbouring pixels,
    noise is suppressed.

    Parameters
    ----------
    sensor : Sensor
        The sensor of which the coherence is calculated.
    interval : float
        The time interval in which the photons were measured.
    bin_size : float
        The size of the sub-intervals that split up the data in order to create a discrete time signal.
    degree : int
        Indicates the degree of nearest neighbours that is included in the average.

    Returns
    -------
    result : np.array()
        The noise-suppressed auto-coherence of all pixels on the sensor.
    """
    x_pixels = sensor.dimensions[sensor.X]
    y_pixels = sensor.dimensions[sensor.Y]
    nr_pixels = x_pixels * y_pixels
    pairs = np.zeros(nr_pixels)
    photon_count = np.zeros(nr_pixels)

    # Generate photon pairs and photon count matrix.
    for i in range(nr_pixels):
        if i not in sensor.data_per_pixel:  # If this pixel did not detect any photons, go to the next pixel.
            continue
        signal = sensor.data_per_pixel[i]
        pairs[i] = auto_coherence(signal, interval, bin_size, 1, normalize=False)[0]
        photon_count[i] = len(signal)
    pairs = np.reshape(pairs, (x_pixels, y_pixels))
    photon_count = np.reshape(photon_count, (x_pixels, y_pixels))
    photon_count = photon_count ** 2

    kernels = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
    ])
    kernel = np.sum(kernels[:(degree + 1)], axis=0)

    summed_pairs = convolve2d(pairs, kernel, mode='same', boundary='fill', fillvalue=0)
    summed_photon_count = convolve2d(photon_count, kernel, mode='same', boundary='fill', fillvalue=0)

    # Normalize
    nr_bins = int(interval / bin_size)
    result = nr_bins * summed_pairs / summed_photon_count
    result[summed_photon_count == 0] = 0
    return result


def nearest_neighbour_coherence(x_dim, y_dim, coherence_matrix, degree=0, threshold=float('inf')):
    """
    Calculates the average coherence of the n-th order nearest neighbours of every pixel.

    Example: a 3x3 pixel array [[0, 1, 2], [3, 4, 5], [6, 7, 8]] averages for pixel 4:
    - degree 0: correlation (4, 4)
    - degree 1: average of correlation (4, 1), (4, 3), (4, 5), (4, 7)
    - degree 2: average of correlation (4, 0), (4, 2), (4, 6), (4, 8)
    - degree 3: average of correlation (4, 1), (4, 3), (4, 5), (4, 7), (4, 0), (4, 2), (4, 6), (4, 8)

    Parameters
    ----------
    x_dim : int
        The number of pixels in the x direction.
    y_dim : int
        The number of pixels in the y direction.
    coherence_matrix : np.array()
        The second order coherence at delay 0 calculated for the correlation between pixel i and j.
    degree : int
        The degree of neighbours that should be included in the average. Default is 0.
    threshold : float
        Values that are larger than the threshold should be ignored in the computation.
    """
    if degree > 3:
        raise ValueError(f"The maximum degree of nearest neighbours is 3. {degree} > 3")
    return average_nn(x_dim, y_dim, coherence_matrix, threshold, degree)


def show_coherence(x_data, y_data, show_fit=False, auto_scale = False, title=None, save_as=None):
    """
    Plots the coherence. Optional: fit an exponential function to the coherence and plot that as well.

    Parameters
    ----------
    x_data : np.array()
        The amount of delay of the shifted signal.
    y_data : np.array()
        The auto-correlation of the photon count with itself.
    show_fit : bool
        If True, an exponential fit is calculated and plotted on top of the coherence. Default is False.
    auto_scale : bool
        If True, the y-axis is automatically scaled. Default is False.
    title : str
        The title that is put on top of the plot. Default is None.
    save_as : str
        The filepath for saving the plot. Default is None.
    """
    plt.plot(x_data, y_data)
    if show_fit:
        fit, popt = fit_coherence_function(x_data, y_data, initial_guess=np.array([3, 2]))
        plt.plot(x_data, fit, color='r',
                 label=f"1 - (1/{np.round(popt[0], 2)})exp(-{np.round(popt[1], 2)}x)"
                 )
        plt.legend()

    plt.xlabel(r'$\tau \Delta t$ [ns]')
    plt.ylabel(r'$g^{(2)}[\ell]$')
    if auto_scale:
        plt.ylim(auto=True)
    else:
        plt.ylim(-0.01, 1.25)
    plt.tight_layout()
    if title:
        plt.title(title)
    if save_as:
        plt.savefig(fname=save_as)
    plt.show()


def show_pixel_to_pixel(sensor, coherence_matrix, title=None, save_as=None):
    """
    Plots a 2D histogram of the second order coherence at lag 0 for all pixels correlated to all pixels.

    Parameters
    ----------
    sensor : Sensor
        The sensor on which the measurements were performed.
    coherence_matrix : np.array()
        The second-order coherence for all pixels compared to all other pixels on the sensor.
    title : str
        Optional. Sets the title of the plot. Default is None.
    save_as : str
        The filepath for saving the plot. Default is None.
    """
    nr_pixels = sensor.dimensions[0] * sensor.dimensions[1]
    x, y = np.meshgrid(np.arange(nr_pixels), np.arange(nr_pixels))

    # Array does not follow Cartesian convention, therefore transpose the data for visualization.
    plt.pcolormesh(x, y, coherence_matrix.T, vmin=0, vmax=1)
    plt.xlabel("Pixel number")
    plt.ylabel("Pixel number")
    plt.colorbar()
    if title:
        plt.title(title)
    if save_as:
        plt.savefig(fname=save_as)
    plt.show()


def fit_coherence_function(lag, coherence_array, method='with_k', initial_guess=np.array([1, 1]), debug=False, example_emitter=None, laser_power=None):
    """
    Fits the function f(x) = 1 - (1 / n) * exp(-(sigma * I + 1 / tau) * x) to the second-order coherence function
    g^2(x).

    Parameters
    ----------
    lag : np.array()
        The x coordinate of the function that is fit.
    coherence_array : np.array()
        The y coordinate of the function that is fit.
    method : str
        The method that is used for fitting the function. Options are 'with_k' and 'without_k' and 'custom'.
    initial_guess : np.array()
        Initial guess for the parameters. The first parameter equals the number of emitters, the second parameter equals
        the summed rate of the hypoexponential distribution of the photons (sigma * I + 1 / tau).
    debug : bool
        If True, prints debug information about the fitting process. Default is True.
    example_emitter : Emitter
        An example emitter object that can be used to retrieve the expected excitation time and absorption wavelength.
    laser_power : float
        The laser power used for the measurements. Required if method is 'without_k'.

    Returns
    -------
    np.array()
        The y coordinates of the fitted function belonging to the input x coordinates.
    popt : tuple
        The optimal coordinates that were found for this fit.
    """
    #k_ex = expected_excitation_time(example_emitter.extinction_coefficient, example_emitter.absorption_wavelength, laser_power)
    #k_em = example_emitter.lifetime
    #expected_emitters_analytical = expected_number_of_emitters(coherence_array[0], 0.1, k_ex, k_em)
    #print(f"Expected number of emitters (analytically determined): {expected_emitters_analytical}")

    if method == 'with_k':
        def func(x, n, a):
            return 1 - (1 / n) * np.exp(-a * x)
        popt, pcov = curve_fit(f=func, xdata=lag, ydata=coherence_array, p0=initial_guess, bounds=([0, 0.5], [np.inf, np.inf]))
        print(f"Fitted k: {popt[1]}")
    elif method == 'without_k':
        initial_guess =initial_guess[0]

        if debug:
            print(f"Example emitter extinction coefficient: {example_emitter.extinction_coefficient}, ")
            print(f"Example emitter absorption wavelength: {example_emitter.absorption_wavelength}, ")
            print(f"Laser power: {laser_power}")
        # Retrieve k = k_ex + k_em
        k_ex = expected_excitation_time(example_emitter.extinction_coefficient, example_emitter.absorption_wavelength, laser_power)
        k_em = example_emitter.lifetime
        k = np.min([k_ex, k_em])
        #print(f"Calculated k: {k}")

        if debug:
            print(f"Using k = {k} for fitting without k method.")
        def func(x, n):
            return 1 - (1 / n) * np.exp(-k* x)
        popt, pcov = curve_fit(f=func, xdata=lag, ydata=coherence_array, p0=initial_guess, bounds=([0, np.inf]))
    elif method == 'custom':
        range_g2inf = coherence_array[-100:]
        g2inf = np.mean(range_g2inf)
        range_g2dt = coherence_array[5:45]
        g2dt = np.mean(range_g2dt)
        g2max = 1 - (g2inf - g2dt)

        if debug:
            print(f"Using custom fitting method. \n g2inf: {g2inf} \n g2dt: {g2dt} \n g2max: {g2max}")

        def func(x, n):
            return g2dt - (1 / n) * np.exp(-2.0091910906356505 * x)
        popt, pcov = curve_fit(f=func, xdata=lag, ydata=coherence_array, p0=initial_guess, bounds=([0], [np.inf]))

    return func(lag, *popt), popt, pcov

def generate_est_nr_emitters_map(scan_data, 
                                 detector_data,
                                 min_photon_count, 
                                 method,
                                 initial_guess,
                                 subtract_autocoherences=False,
                                 laser_power=None,
                                 verbose=False):
    """
    Generate a map of estimated number of emitters based on photon counts.
    This is a placeholder function that can be replaced with a more complex model.

    Parameters
    ----------
    positions : tuple
        The dimensions of the map (rows, columns).
    photon_count_map : np.array
        A 2D array containing the number of photons detected at each position.
    g2_data : dict
        A dictionary where keys are (x, y) positions and values are tuples of (g2_values, bins).
    min_photon_count : int
        Minimum number of photons required to perform a fit at a position.
    method : str
        The method used for fitting the coherence function. Options are 'with_k', 'without_k', or 'custom'.
    initial_guess : np.array
        Initial guess for the fitting parameters. Should be an array of two elements: [n, a].
    verbose : bool
        If True, prints detailed information about the fitting process.
    
    Returns
    -------
    n_map : np.array
        A 2D array containing the estimated number of emitters at each position.
    sigma_map : np.array
        A 2D array containing the uncertainty of the estimated number of emitters at each position.

    """
    positions = scan_data['positions']
    area_size = scan_data['area_size']
    g2_data = scan_data['g2_data']
    G2_data = scan_data['G2_data']
    photon_count_map = scan_data['photon_count_map']


    # Create empty maps for estimated number of emitters and fit uncertainty
    n_map = np.zeros(positions) + np.nan
    sigma_map = np.zeros(positions) + np.nan
    
    # Track fitting statistics
    fit_success = 0
    fit_failed = 0
    skipped_low_counts = 0

    locations_to_smooth_out = [] #the ix,iy locations that we will smooth out later because of failed fit
    uncertainty_threshold = 0.5
    
    # Fit g2 curves for each position and extract number of emitters
    for (ix, iy), g2_values in g2_data.items():
        # Skip positions with too few photon hits
        if photon_count_map[ix, iy] < min_photon_count:
            skipped_low_counts += 1
            continue
        
        if g2_values is not None:
            if subtract_autocoherences:
                G2_values = G2_data.get((ix, iy), None)
                # subtract autocoherences from coherence.
                values, bins = coherence_autocoherence_subtracted(ix, iy, G2_values, scan_data, detector_data)

                #plot
                # fig, ax = plt.subplots(figsize=(10, 5))
                # ax.plot(bins, values, label='G2', color='blue')
                # ax.set_xlabel('Lag (ns)')
                # ax.set_ylabel('G2')
                # ax.set_title(f'G2 Curve at ({ix}, {iy})')
                # ax.legend()
                # plt.show()

            else:
                values, bins = g2_values
            # TODO justify this takign first point away
            values = values[1:]
            bins = bins[1:]

            #get rid of nans (replace by 0)
            values = np.nan_to_num(values)

            # fit, popt, pcov = fit_coherence_function(bins, values,
            #                                             method=method, 
            #                                             example_emitter = Alexa647(x=0,y=0),
            #                                             laser_power = laser_power,
            #                                             initial_guess=initial_guess)
            # n = np.round(popt[0], 2)
            # sigma = np.sqrt(np.diag(pcov))[0]

            # print(f"Fitting succeeded at position ({ix}, {iy}): n={n}, sigma={sigma}.")
            # n_map[ix, iy] = n
            # sigma_map[ix, iy] = sigma
            # fit_success += 1


            
            try:
                # Skip positions with insufficient data points (more than 50% zero or empty values in g2_values)
                if len(values) < 5 or np.all(np.isnan(values)) or np.sum(values == 0) > 0.5 * len(values):
                    locations_to_smooth_out.append((ix, iy))
                    fit_failed += 1
                    
                    continue

                # Apply fitting function
                fit, popt, pcov = fit_coherence_function(bins, values, 
                                                                method=method, 
                                                                initial_guess=initial_guess,
                                                                example_emitter = Alexa647(x=0,y=0),
                                                                laser_power = laser_power,
                                                                debug=verbose
                )
                
                # Extract number of emitters and uncertainty
                n = np.round(popt[0], 2)
                sigma = np.sqrt(np.diag(pcov))[0]
                #print(f"Fitting at position ({ix}, {iy}): n={n}, sigma={sigma}.")
                # Add sanity check for the fitted values
                if n > 0 and n < 100:
                    #if sigma > n:  # Reasonable range
                    #    print(f"Fitting with high uncertainty at position ({ix}, {iy}): n={n}, sigma={sigma}.")

                    if sigma > uncertainty_threshold*n:
                        print(f"Fitting with high uncertainty at position ({ix}, {iy}): n={n}, sigma={sigma}. Setting estimate at average of surrounding area.")
                        locations_to_smooth_out.append((ix, iy))
                    # Store results in maps
                    n_map[ix, iy] = n
                    sigma_map[ix, iy] = sigma
                    fit_success += 1
                    #print(f"Fitting succeeded at position ({ix}, {iy}): n={n}, sigma={sigma}.")
                else:
                    print(f"Fitting failed at position ({ix}, {iy}): n={n}, sigma={sigma}. Trying without first datapoint.")
                                            # Attempt to fit but without the first datapoint at zero lag
                    fit, popt, pcov = fit_coherence_function(bins[1:], values[1:],
                                                                method=method, 
                                                                initial_guess=initial_guess)
                    n = np.round(popt[0], 2)
                    sigma = np.sqrt(np.diag(pcov))[0]


                    # Add sanity check for the fitted values
                    if n > 0 and n < 100 and sigma < uncertainty_threshold*n:  # Reasonable range
                        if verbose:
                            print(f"Fitting succeeded at position ({ix}, {iy}) without first datapoint: n={n}, sigma={sigma}.")
                        # Store results in maps
                        n_map[ix, iy] = n
                        sigma_map[ix, iy] = sigma
                        fit_success += 1
                    else:
                        if verbose:
                            print(f"Fitting failed again without first datapoint at position ({ix}, {iy}): n={n}, sigma={sigma}")
                        locations_to_smooth_out.append((ix, iy))
                        fit_failed += 1
                    #fit_failed += 1
                    
            except Exception as e:
                try:
                    # Attempt to fit but without the first datapoint at zero lag
                    fit, popt, pcov = fit_coherence_function(bins[1:], values[1:],
                                                                method=method, 
                                                                initial_guess=initial_guess)
                    n = np.round(popt[0], 2)
                    sigma = np.sqrt(np.diag(pcov))[0]

                    # Add sanity check for the fitted values
                    if n > 0 and n < 100 and sigma < uncertainty_threshold*n:  # Reasonable range
                        # Store results in maps
                        #print(f"Fitting succeeded at position ({ix}, {iy}): n={n}, sigma={sigma}.")
                        n_map[ix, iy] = n
                        sigma_map[ix, iy] = sigma
                        fit_success += 1
                    else:
                        if verbose:
                            print(f"Fitting failed again without first datapoint at position ({ix}, {iy}): n={n}, sigma={sigma}")
                        locations_to_smooth_out.append((ix, iy))
                        fit_failed += 1
                except:
                    locations_to_smooth_out.append((ix, iy))
                    fit_failed += 1
                # For debugging, uncomment the following line:
                # print(f"Fitting failed at position ({ix}, {iy}): {e}")

    for ix, iy in locations_to_smooth_out:
        neigh = get_neighbors(ix, iy, shape=n_map.shape, radius=1)
        valid = [(x,y) for (x,y) in neigh if (x,y) not in locations_to_smooth_out]
        if valid:
            vals = [n_map[x,y] for (x,y) in valid]
            n_map[ix,iy] = np.mean(vals)  # or weighted mean
        else:
            n_map[ix,iy] = np.nan  # leave for later repair

    #Replace empty values with zeros
    n_map[np.isnan(n_map)] = 0  
    sigma_map[np.isnan(sigma_map)] = 0


    if verbose:
        print(f"Fitting summary: {fit_success} successful fits, {fit_failed} failed fits, "
                f"{skipped_low_counts} skipped due to low photon count (<{min_photon_count})")
        
    return n_map, sigma_map

def coherence_autocoherence_subtracted(ix, iy, G2_values, scan_data, detector_data):
    """
    ix, iy: scanning coordinates
    G2_values: values, bins
    scan_data: dictionary
    detector_data: dictionary containing (ix, iy) scanning coordinates as keys, each corresponding to a list/array with 23 SPAD pixel photon streams

    """
    # Whole detector coherence:
    full_coherence, bins = G2_values

    #print(full_coherence)

    #plot full coherence

    # plt.figure(figsize=(10, 5))
    # plt.plot(bins, full_coherence, label='Full Coherence', color='blue')
    # plt.xlabel('Lag (ns)')
    # plt.ylabel('Coherence')
    # plt.title(f'Full Coherence at ({ix}, {iy})')
    # plt.legend()
    # plt.show()

    # detector data for scanning position:
    detector_data_at_position = detector_data.get((ix, iy), {})

    #print(f"Detector data: {detector_data_at_position}")
    # Pixel specific autocoherences
    pixel_autocoherence_array = []
    for i in range(detector_data_at_position.keys().__len__()):
        photons = detector_data_at_position.get(i, [])
        #print(f"Detector data for pixel {i}: {photons}")
        #only take second parts of elements in photons
        photons = np.array([det[1] for det in photons])
        if photons.size == 0:
            continue
        #print(photons)
        pixel_autocoherence, bins = auto_coherence(photons, interval=scan_data['dwell_time'], bin_size=0.1, nr_steps=800, normalize=False)
        pixel_autocoherence_array.append(pixel_autocoherence)
    
    pixel_neighbourhood_coherence = full_coherence - np.sum(pixel_autocoherence_array, axis=0)
    # Normalize the result
    pixel_neighbourhood_coherence /= pixel_neighbourhood_coherence[-100:].mean(axis=0)


    # plt.figure()
    # plt.plot(bins, pixel_neighbourhood_coherence, label='Pixel Neighbourhood Coherence', color='orange')
    # plt.xlabel('Lag (ns)')
    # plt.ylabel('Coherence')
    # plt.title(f'Pixel Neighbourhood Coherence at ({ix}, {iy})')
    # plt.legend()
    # plt.show()

    return pixel_neighbourhood_coherence, bins

def calculate_G2_difference(correlation_data, start_index=0, tau_min=50, tau_max=None, difference_threshold=5000):
    """
    Calculate the difference between G²(0) and G²(∞) for quantum correlation data.
    
    Parameters:
    correlation_data : array-like
        The photon pair correlation data
    start_index : int
        Index corresponding to zero delay (G²(0))
    tau_min : int
        Minimum delay index to include in G²(∞) calculation
    tau_max : int or None
        Maximum delay index to include in G²(∞) calculation. If None, uses length of data.
    
    Returns:
    dict: Dictionary containing G²(0), G²(∞), and their difference
    """
    # Get G²(0)
    G2_zero = correlation_data[start_index]
    
    # Set tau_max if not provided
    if tau_max is None:
        tau_max = len(correlation_data)
    
    # Calculate G²(∞) by averaging over delay values [tau_min, tau_max]
    G2_inf = np.mean(correlation_data[tau_min:tau_max])
    
    # Calculate the difference
    G2_difference = G2_inf - G2_zero

    # Filter for extreme values
    if G2_difference > difference_threshold:
        G2_difference = 0
    elif G2_difference < 0:
        G2_difference = 0
    
    return {
        "G2_zero": G2_zero,
        "G2_infinity": G2_inf,
        "difference": G2_difference
    }

def fit_deadtime_decay(lag, coherence_array, initial_guess):
    """
    Fits the function f(x) = 1 + b*exp(-c * x) to the decay afther the deadtime of the second-order coherence function g^2(x).

    Parameters
    ----------
    lag : np.array()
        The x coordinate of the function that is fit.
    coherence_array : np.array()
        The y coordinate of the function that is fit.
    initial_guess : np.array()
        Initial guess for the parameters. The first parameter equals the height of the deadtime peak, the second parameter equals the speed of decay, depending on the excitation rate.

    Returns
    -------
    np.array()
        The y coordinates of the fitted function belonging to the input x coordinates.
    popt : tuple
        The optimal coordinates that were found for this fit.
    """

    def func(x, b, c):
        return 1 + b * np.exp(-c * x)
    popt, pcov = curve_fit(f=func, xdata=lag, ydata=coherence_array, p0=initial_guess, bounds=([0, 0], [np.inf, np.inf]))

    return func(lag, *popt), popt, pcov
