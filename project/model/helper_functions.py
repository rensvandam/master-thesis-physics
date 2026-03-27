import heapq
import numpy as np
import numba as nb
import json
import hashlib

from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass, gaussian_filter
from scipy.stats import multivariate_normal

@nb.jit
def merge_k(arrays):
    """
    Merges k arrays that are sorted ascending to form one big sorted array.

    Parameters
    ----------
    arrays : list of np.array()
        List of all arrays that should be merged.

    Returns
    -------
    result : np.array()
        Array sorted in ascending order.
    """
    nr_elements = 0
    for array in arrays:
        nr_elements += len(array)

    if len(arrays) == 0 or nr_elements == 0:
        # TODO: this is actually a bug, should not return anything when there is no photon, but I implemented it this
        #  way because Numba doesn't seem to support the initialization of empty arrays.
        return np.array([0], dtype=np.float64)
    if len(arrays) == 1:
        return arrays[0].copy()

    result = np.zeros(nr_elements)
    element_counter = 0
    min_heap = [(array[0], i, 0) for i, array in enumerate(arrays) if len(array) != 0]
    heapq.heapify(min_heap)

    while min_heap:
        _, array, element = heapq.heappop(min_heap)

        result[element_counter] = arrays[array][element]
        element_counter += 1

        element += 1
        if element < len(arrays[array]):
            heapq.heappush(min_heap, (arrays[array][element], array, element))

    return result


@nb.jit
def merge_k_2D(arrays, sort_by_index=2):
    """
    Merges k 2D arrays that are sorted ascending by a certain column to form one big sorted array.

    Parameters
    ----------
    arrays : list of np.array()
        List of all arrays that should be merged.
    sort_by_index : int
        Index of the column on which the array should be sorted.

    Returns
    -------
    result : np.array()
        Array (number of data points, properties) sorted by a column in ascending order.
    """
    nr_elements = 0
    for array in arrays:
        nr_elements += len(array)

    if len(arrays) == 0 or nr_elements == 0:
        # TODO: this is actually a bug, should not return anything when there is no photon, but I implemented it this
        #  way because Numba doesn't seem to support the initialization of empty arrays.
        return np.zeros((1, 1), dtype=np.float64)
    if len(arrays) == 1:
        return arrays[0].copy()

    result = np.empty((nr_elements, np.shape(arrays[0])[1]))
    min_heap = [(array[0][sort_by_index], i, 0) for i, array in enumerate(arrays) if len(array) != 0]
    heapq.heapify(min_heap)

    for i in range(nr_elements):
        _, array, element = heapq.heappop(min_heap)
        result[i, :] = arrays[array][element]
        element += 1
        if element < len(arrays[array]):
            heapq.heappush(min_heap, (arrays[array][element][sort_by_index], array, element))
    return result


@nb.jit
def insertion_sort_2D(array, sort_by_index):
    """
    Sorts a 2D array in ascending order based on a given column.

    Parameters
    ----------
    array : np.ndarray
        A (nr_elements, nr_columns) array that is almost sorted.
    sort_by_index : int
        The column that the array should be sorted by.

    Returns
    -------
    np.ndarray
        The sorted array.
    """
    nr_elements = len(array)
    if nr_elements == 0 or array.ndim <= 1:
        return array

    for i in range(1, nr_elements):
        data = array[i, :].copy()
        key = data[sort_by_index]
        j = i - 1
        while j >= 0 and key < array[j, sort_by_index]:
            array[j + 1, :] = array[j, :]
            j -= 1
        array[j + 1, :] = data
    return array


@nb.jit
def count_pairs(array1, array2):
    """
    Counts the number of element pairs that occur in both arrays. Example: [1, 4, 4] and [1, 3, 4, 5] counts 1 pair of
    ones and 2 pairs of fours, so the function returns 3.

    Parameters
    ----------
    array1 : np.array()
        Array of N elements, sorted in ascending order.
    array2 : np.array()
        Array of M elements, sorted in ascending order.

    Returns
    -------
    count : int
        The number of pairs of elements that was counted in the input arrays.
    """
    N = len(array1)
    M = len(array2)

    count = 0
    i = 0
    j = 0
    while i < N and j < M:
        val1 = array1[i]
        val2 = array2[j]

        if val1 < val2:
            i += 1
        elif val2 < val1:
            j += 1
        else:
            # Found a pair!
            count_array1 = 1
            while (i + 1) < N and array1[i + 1] == val1:
                count_array1 += 1
                i += 1
            count_array2 = 1
            while (j + 1) < M and array2[j + 1] == val2:
                count_array2 += 1
                j += 1

            count += count_array1 * count_array2
            i += 1
            j += 1
    return count


@nb.jit
def sparse_convolution(indices1, indices2, nr_steps, offset=0):
    """
    Calculates a convolution of indices1 with indices2 for a number of steps. The input arrays contain indices of delta
    peaks in the discrete input signal. Because the signal is sparse, the fastest way to compute the convolution is by
    doing a convolution by comparing the indices of the data to each other.

    Parameters
    ----------
    indices1 : np.array()
        Data indices of a discrete signal sorted in ascending order.
    indices2 : np.array()
        Data indices of a discrete signal sorted in ascending order. To be correlated with indices1.
    nr_steps : int
        The number of steps the signal should be shifted across itself to calculate the convolution.
    offset : int
        Optional offset for the number of lags. For example, if nr_steps is 100 and start is 30, the convolution will
        calculate the convolution for lag 30 to lag 129. Default is 0.

    Returns
    -------
    pairs : np.array()
        A (nr_steps,) array containing the number of element pairs when indices2 is shifted over indices1 with a certain
        lag.
    """
    pairs = np.zeros(nr_steps)
    for m in range(nr_steps):
        pairs[m] = count_pairs(indices1, indices2 + offset + m)
    return pairs


def select_neighbours(size, center, dim_axis_0, dim_axis_1):
    """
    Selects the elements within a kernel centered at the center element.

    Parameters
    ----------
    size : int
        The size of the kernel.
    center : int
        The number of the element with which the kernel center aligns. Should be calculated by:
        index_0 * dim_axis_1 + index_1.
    dim_axis_0 : int
        The number of elements in axis 0.
    dim_axis_1 : int
        The number of elements in axis 1.

    Returns
    -------
    list<int>
        A list of numbers that indicate the neighbouring pixels of the center element (the center element is included).
    """
    half = size // 2
    col_start = max(center % dim_axis_1 - half, 0)
    col_end = min(center % dim_axis_1 + half + 1, dim_axis_1)
    row_start = max(center // dim_axis_1 - half, 0)
    row_end = min(center // dim_axis_1 + half + 1, dim_axis_0)
    return [i * dim_axis_1 + j for i in range(row_start, row_end) for j in range(col_start, col_end)]


def average(sums, counts):
    """
    Calculates the average value of the sums that consist of counts terms.

    Parameters
    ----------
    sums : np.array()
        Array containing the sums of all terms.
    counts : np.array()
        Array containing the amount of terms that every sum consists of.

    Returns
    -------
    result : np.array()
        The average value of each element in the array. If the number of terms was 0, the average is 0.
    """
    result = sums / counts
    result[counts == 0] = 0
    return result


def average_nn(x_dim, y_dim, array, threshold, degree):
    """
    Calculates the average value of the first degree neighbours of every element. Method ignores values larger than
    threshold.

    Parameters
    ----------
    x_dim : int
        The number of elements in the x direction.
    y_dim : int
        The number of elements in the y direction.
    array : np.array()
        A (x_dim * y_dim, x_dim * y_dim) array.
    threshold : float
        Values that are larger than the threshold are ignored in the computation.
    degree : int
        Degree of neighbours that is included in the computation.

    Returns
    -------
    np.array()
        A (x_dim, y_dim) array that contains the average of the nth degree neighbours of every element.
    """
    sums, counts = sum_nn(x_dim, y_dim, array, threshold, degree)
    return average(sums, counts)


def sum_nn(x_dim, y_dim, array, threshold, degree):
    """
    Sums the nth degree nn relations of every element. Values larger than or equal to the threshold are excluded.

    Parameters
    ----------
    x_dim : int
        The number of elements in the x direction.
    y_dim : int
        The number of elements in the y direction.
    array : np.array()
        A (x_dim * y_dim, x_dim * y_dim) array.
    threshold : float
        Values that are larger than the threshold are ignored in the computation.
    degree : int
        Degree of neighbours that is included in the computation.

    Returns
    -------
    sums : np.array()
        A (x_dim, y_dim) array that contains the sum of the nth degree nearest neighbour relations of every element.
    counts : np.array()
        A (x_dim, y_dim) array containing the number of elements that were summed.
    """
    if degree == 0:
        return np.diag(array).reshape((x_dim, y_dim)), np.ones((x_dim, y_dim))
    if degree == 1:
        return sum_first_degree_nn(x_dim, y_dim, array, threshold)
    if degree == 2:
        return sum_second_degree_nn(x_dim, y_dim, array, threshold)
    if degree == 3:
        sums1, counts1 = sum_first_degree_nn(x_dim, y_dim, array, threshold)
        sums2, counts2 = sum_second_degree_nn(x_dim, y_dim, array, threshold)
        return sums1 + sums2, counts1 + counts2
    return np.zeros((x_dim, y_dim)), np.zeros((x_dim, y_dim))


def sum_first_degree_nn(x_dim, y_dim, array, threshold):
    """
    Sums the first degree (up, down, left, right) relations of every element. Values larger than or equal to the
    threshold are excluded.

    Parameters
    ----------
    x_dim : int
        The number of elements in the x direction.
    y_dim : int
        The number of elements in the y direction.
    array : np.array()
        A (x_dim * y_dim, x_dim * y_dim) array.
    threshold : float
        Values that are larger than the threshold are ignored in the computation.

    Returns
    -------
    sums : np.array()
        A (x_dim, y_dim) array that contains the sum of the first degree nearest neighbour relations of every element.
    counts : np.array()
        A (x_dim, y_dim) array containing the number of elements that were summed.
    """
    nr_elements = x_dim * y_dim
    sums = np.zeros((x_dim, y_dim))
    counts = np.zeros((x_dim, y_dim))

    for x in range(x_dim):
        for y in range(y_dim):
            index = x * y_dim + y
            total = 0
            count = 0
            current_array = array[index, :]

            upper = index - y_dim
            if upper >= 0 and current_array[upper] < threshold:
                total += current_array[upper]
                count += 1
            left = index - 1
            if index % y_dim > 0 and current_array[left] < threshold:
                total += current_array[left]
                count += 1
            right = index + 1
            if right % y_dim > 0 and current_array[right] < threshold:
                total += current_array[right]
                count += 1
            lower = index + y_dim
            if lower < nr_elements and current_array[lower] < threshold:
                total += current_array[lower]
                count += 1

            sums[x, y] = total
            counts[x, y] = count
    return sums, counts


def sum_second_degree_nn(x_dim, y_dim, array, threshold):
    """
    Sums the second degree (upper left, upper right, lower left, lower right) relations of every element. Values larger
    than or equal to the threshold are excluded.

    Parameters
    ----------
    x_dim : int
        The number of elements in the x direction.
    y_dim : int
        The number of elements in the y direction.
    array : np.array()
        A (x_dim * y_dim, x_dim * y_dim) array.
    threshold : float
        Values that are larger than the threshold are ignored in the computation.

    Returns
    -------
    sums : np.array()
        A (x_dim, y_dim) array that contains the sum of the second degree nearest neighbour relations of every element.
    counts : np.array()
        A (x_dim, y_dim) array containing the number of elements that were summed.
    """
    nr_elements = x_dim * y_dim
    sums = np.zeros((x_dim, y_dim))
    counts = np.zeros((x_dim, y_dim))

    for x in range(x_dim):
        for y in range(y_dim):
            index = x * y_dim + y
            total = 0
            count = 0
            current_array = array[index, :]

            if not index % y_dim == 0:  # Elements on position y = 0 don't have left neighbours.
                upper_left = index - y_dim - 1
                if upper_left >= 0 and current_array[upper_left] < threshold:
                    total += current_array[upper_left]
                    count += 1
                lower_left = index + y_dim - 1
                if lower_left < nr_elements and current_array[lower_left] < threshold:
                    total += current_array[lower_left]
                    count += 1

            if not (index + 1) % y_dim == 0:  # Elements on the right edge don't have right neighbours.
                upper_right = index - y_dim + 1
                if upper_right >= 0 and current_array[upper_right] < threshold:
                    total += current_array[upper_right]
                    count += 1
                lower_right = index + y_dim + 1
                if lower_right < nr_elements and current_array[lower_right] < threshold:
                    total += current_array[lower_right]
                    count += 1

            sums[x, y] = total
            counts[x, y] = count
    return sums, counts


def mean_squared_error(array_1, array_2, weights=None) -> np.ndarray:
    """
    Calculates the mean squared error between two arrays.

    Parameters
    ----------
    array_1 : np.ndarray
        The first array for the comparison.
    array_2 : np.ndarray
        The second array for the comparison.
    weights : np.ndarray
        The weight that should be given to each pixel. Optional, default is None.

    Returns
    -------
    np.ndarray
        The mean squared error of the number_map compared to the ground_truth.
    """
    if not (np.shape(array_1) == np.shape(array_2)):
        raise ValueError("The shapes of the matrices do not match.")
    if weights is not None:
        return np.sum(weights * ((array_1 - array_2) ** 2)) / np.sum(weights)
    return np.sum((array_1 - array_2) ** 2) / np.size(array_1)

# def gaussian_2d(coords: tuple,
#                 amplitude: float, x0: float, y0: float, 
#                 sigma_x: float, sigma_y: float, offset: float) -> np.ndarray:
#     """
#     Generate a 2D Gaussian function
    
#     Parameters:
#     -----------
#     x, y : np.ndarray
#         Meshgrid arrays of x and y coordinates
#     amplitude : float
#         Peak height of the Gaussian
#     x0, y0 : float
#         Center coordinates of the Gaussian
#     sigma_x, sigma_y : float
#         Standard deviations in x and y directions
#     offset : float
#         Background offset
        
#     Returns:
#     --------
#     np.ndarray
#         2D Gaussian array
#     """
#     x,y = coords
#     return amplitude * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + 
#                               (y - y0)**2 / (2 * sigma_y**2))) + offset

def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, offset=0, flatten=True):
    """2D Gaussian function"""
    x, y = coords
    result = amplitude * np.exp(-((x - x0)**2 / (2 * sigma_x**2) +
                                  (y - y0)**2 / (2 * sigma_y**2))) + offset
    return result.ravel() if flatten else result


def get_psf_params_peak_preserving(image):
    """
    Fit while strongly constraining amplitude to match observed peak.
    Best for preserving amplitude accuracy.
    """
    h, w = image.shape
    
    # Find true peak location (with subpixel accuracy)
    peak_val = np.max(image)
    peak_idx = np.unravel_index(np.argmax(image), image.shape)
    cy, cx = peak_idx[0], peak_idx[1]
    
    # Refine peak location using center of mass around peak
    window_size = 3
    y1 = max(0, cy - window_size)
    y2 = min(h, cy + window_size + 1)
    x1 = max(0, cx - window_size)
    x2 = min(w, cx + window_size + 1)
    
    peak_region = image[y1:y2, x1:x2]
    if peak_region.size > 0:
        rel_cy, rel_cx = center_of_mass(peak_region)
        cy = y1 + rel_cy
        cx = x1 + rel_cx
    
    # Create coordinate grids
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    
    #
    
    # Net amplitude (peak minus background)
    net_amplitude = peak_val
    
    # TIGHT amplitude bounds around observed peak
    amp_tolerance = 0.1  # Allow only 10% variation
    amp_bounds = (
        net_amplitude * (1 - amp_tolerance),
        net_amplitude * (1 + amp_tolerance)
    )
    
    # Initial sigma estimate
    initial_sigma = min(h, w) / 8
    
    # Bounds: Very tight on amplitude, reasonable on others
    bounds = (
        [amp_bounds[0], cx-3, cy-3, 1, 1, -0.01],
        [amp_bounds[1], cx+3, cy+3, min(w,h)/3, min(w,h)/3, 0.01]
    )
    
    print(bounds)
    
    try:
        popt, _ = curve_fit(
            gaussian_2d, 
            (X, Y), 
            image.ravel(),
            p0=[net_amplitude, cx, cy, initial_sigma, initial_sigma, 0],
            bounds=bounds,
            maxfev=2000
        )
        
        amplitude, _, _, sigma_x, sigma_y, _ = popt
        sigma = (abs(sigma_x) + abs(sigma_y)) / 2
        return amplitude, sigma  # Return total amplitude
        
    except Exception as e:
        print(f"Peak-preserving fit failed: {e}")
        return peak_val, initial_sigma

def get_psf_params_robust(image):
    """
    Robust PSF fitting using masked region and conservative bounds.
    Best for boundary-affected PSFs.
    """
    h, w = image.shape
    
    # Use center of mass for better centering
    cy, cx = center_of_mass(image)
    
    # Create mask to exclude outer regions (use inner 70%)
    mask_radius = min(h, w) * 0.35
    y, x = np.ogrid[:h, :w]
    mask = (x - cx)**2 + (y - cy)**2 <= mask_radius**2
    
    # Create coordinate grids
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Conservative initial parameters
    initial_amp = np.percentile(image, 95)  # Use 95th percentile instead of max
    initial_sigma = min(h, w) / 8  # Conservative sigma
    
    # Bounds to prevent overfitting
    bounds = (
        [0, cx-w/4, cy-h/4, 1, 1, 0],  # lower bounds
        [np.max(image)*1.5, cx+w/4, cy+h/4, min(w,h)/3, min(w,h)/3, np.max(image)*0.1]  # upper bounds
    )
    
    try:
        popt, _ = curve_fit(
            gaussian_2d, 
            (X[mask], Y[mask]), 
            image[mask],
            p0=[initial_amp, cx, cy, initial_sigma, initial_sigma, 0],
            bounds=bounds,
            maxfev=2000
        )
        
        amplitude, _, _, sigma_x, sigma_y, _ = popt
        sigma = (abs(sigma_x) + abs(sigma_y)) / 2
        return amplitude, sigma
        
    except Exception as e:
        # Fallback to moment-based estimate
        print(f"Robust fit failed: {e}, using moment estimate")
        return get_psf_params(image)
    
def get_psf_params_weighted(image):
    """
    Weighted fitting that emphasizes the peak region.
    """
    h, w = image.shape
    cy, cx = center_of_mass(image)
    
    # Create weight matrix (higher weights near center)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_r = min(h, w) / 3
    weights = np.exp(-r**2 / (2 * (max_r/3)**2))  # Gaussian weighting
    
    # Normalize weights
    weights = weights / np.max(weights)
    
    # Create coordinate grids
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Initial parameters
    initial_amp = np.max(image) * 0.9
    initial_sigma = min(h, w) / 8
    
    # Bounds
    bounds = (
        [0, cx-w/4, cy-h/4, 1, 1, 0],
        [np.max(image)*1.5, cx+w/4, cy+h/4, min(w,h)/3, min(w,h)/3, np.max(image)*0.1]
    )
    
    try:
        popt, _ = curve_fit(
            gaussian_2d, 
            (X, Y), 
            image.ravel(),
            p0=[initial_amp, cx, cy, initial_sigma, initial_sigma, 0],
            sigma=1/weights.ravel(),  # Use weights as inverse sigma
            bounds=bounds,
            maxfev=2000
        )
        
        amplitude, _, _, sigma_x, sigma_y, _ = popt
        sigma = (abs(sigma_x) + abs(sigma_y)) / 2
        return amplitude, sigma
        
    except Exception as e:
        print(f"Weighted fit failed: {e}, using moment estimate")
        return get_psf_params(image)
    
    
def get_psf_params_moments(image):
    """
    Estimate PSF parameters using image moments.
    Very robust, doesn't require curve fitting.
    """
    # Remove negative values
    image_clean = np.maximum(image, 0)
    
    # Calculate moments
    total = np.sum(image_clean)
    if total == 0:
        return 0, 1
    
    # Center of mass
    cy, cx = center_of_mass(image_clean)
    
    # Second moments for sigma estimation
    h, w = image_clean.shape
    y, x = np.ogrid[:h, :w]
    
    # Variance in x and y directions
    var_x = np.sum(image_clean * (x - cx)**2) / total
    var_y = np.sum(image_clean * (y - cy)**2) / total
    
    # Convert variance to sigma
    sigma_x = np.sqrt(var_x)
    sigma_y = np.sqrt(var_y)
    sigma = (sigma_x + sigma_y) / 2
    
    # Amplitude estimate (peak value with some smoothing)
    smoothed = gaussian_filter(image_clean, sigma=0.5)
    amplitude = np.max(smoothed)
    
    return amplitude, sigma

def get_psf_params(image):
    """
    Extract PSF amplitude and sigma from a centered PSF image.
    
    Args:
        image: 2D numpy array containing the PSF
    
    Returns:
        tuple: (amplitude, sigma) where sigma is the average of sigma_x and sigma_y
    """
    # Get image dimensions and center coordinates
    h, w = image.shape
    cx, cy = w // 2, h // 2
    
    # Create coordinate grids
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    
    # Initial guess: amplitude = max value, center at image center, sigma ~ image_size/6
    p0 = [np.max(image), cx, cy, min(w, h) / 6, min(w, h) / 6]
    
    # Fit 2D Gaussian
    try:
        popt, _ = curve_fit(gaussian_2d, (X, Y), image.ravel(), p0=p0)
        amplitude, _, _, sigma_x, sigma_y = popt
        sigma = (sigma_x + sigma_y) / 2  # Average sigma
        return amplitude, abs(sigma)
    except:
        # Fallback: use simple estimates
        amplitude = np.max(image)
        # Estimate sigma from full width at half maximum
        half_max = amplitude / 2
        mask = image > half_max
        sigma = np.sqrt(np.sum(mask)) / (2 * np.sqrt(2 * np.log(2)))
        return amplitude, sigma
    
def generate_random_positions(bounds, density, seed, avoid_boundaries=False):
    """
    Generates random positions within the specified 2D bounds with a given density.
   
    Args:
        bounds: Either a dictionary with keys 'min_x', 'max_x', 'min_y', 'max_y' or
               a tuple of (min_x, max_x, min_y, max_y)
        density: Number of positions per unit area
        seed: Random seed for reproducibility
        avoid_boundaries: If True, positions will be at least 0.5 units away from boundaries
       
    Returns:
        List of (x, y) position coordinates
    """
    random = np.random.RandomState(seed)
    
    # Handle different input formats for bounds
    if isinstance(bounds, dict):
        min_x, max_x = bounds['min_x'], bounds['max_x']
        min_y, max_y = bounds['min_y'], bounds['max_y']
    else:
        min_x, max_x, min_y, max_y = bounds
   
    # Adjust bounds if avoiding boundaries
    if avoid_boundaries:
        min_x += 0.5
        max_x -= 0.5
        min_y += 0.5
        max_y -= 0.5
        
        # Check if bounds are still valid after adjustment
        if min_x >= max_x or min_y >= max_y:
            raise ValueError("Bounds too small to avoid boundaries with 0.5 margin")
   
    # Calculate the area of the (possibly adjusted) bounds
    width = max_x - min_x
    height = max_y - min_y
    area = width * height
   
    # Calculate the number of positions to generate based on density
    count = round(area * density)
   
    # Generate the random positions
    positions = []
    for _ in range(count):
        x = min_x + random.random() * width
        y = min_y + random.random() * height
        positions.append((x, y))
   
    return positions

def get_neighbors(ix, iy, shape, radius=1, include_self=False):
    """
    Return list of valid neighbor coordinates around (ix, iy).

    Parameters
    ----------
    ix, iy : int
        Pixel coordinates.
    shape : tuple
        Shape of the map (rows, cols).
    radius : int, optional
        How far out to look (1 = immediate neighbors).
    include_self : bool, optional
        Whether to include (ix, iy) itself.

    Returns
    -------
    list of (x, y) tuples
    """
    nx, ny = shape
    neighbors = []
    for dx in range(-radius, radius+1):
        for dy in range(-radius, radius+1):
            x, y = ix + dx, iy + dy
            if 0 <= x < nx and 0 <= y < ny:
                if include_self or (x, y) != (ix, iy):
                    neighbors.append((x, y))
    return neighbors

def get_readable_filename(metadata):
    # Extract key parameters for filename
    emitter_count = metadata.get('emitter_count', 'unknown')
    scan_size = metadata.get('positions', 'unknown')[0]
    laser_power = metadata.get('laser_power', 'unknown')
    
    # Readable part + hash for uniqueness
    readable = f"emitters{emitter_count}_size{scan_size}_laser{laser_power/1000:.0f}kW"
    hash_part = get_sim_hash(metadata)[:8]  # shorter hash
    
    return f"{readable}_{hash_part}"

def get_sim_hash(params_dict):
    # Sort dict for consistent hashing
    param_str = json.dumps(params_dict, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()[:12]

def transform_coordinates(x, y, area_size, npos, pixel_size, direction = 'to_pixel'):
    """
    Transform coordinates between pixel and real space.
    
    Parameters:
    - x, y: Coordinates to transform.
    - direction: 'to_pixel' or 'to_physical'.
    - area_size: Size of the area in physical units (e.g., micrometers). (width, height)
    - npos: Number of positions in the scanning experiment. (tuple, e.g., (100, 100))
    - pixel_size: Size of a pixel in physical units (e.g., micrometers).
    
    Returns:
    - Transformed coordinates
    """
    xmin = -area_size[0]/2 + pixel_size/2 # minimum coordinate
    ymin = -area_size[1]/2 + pixel_size/2 # minimum coordinate

    if direction == 'to_pixel':
        x_transformed = npos[0] * (x - xmin) / area_size[0]
        y_transformed = npos[1] * (y - ymin) / area_size[1]
    elif direction == 'to_physical':
        x_transformed = x*area_size[0]/npos[0] + xmin
        y_transformed = y*area_size[1]/npos[1] + ymin
    else:
        raise ValueError("Direction must be 'to_pixel' or 'to_physical'")
    
    return x_transformed, y_transformed


# def estimate_emitters_on_roi(prop, timestamp_data):
#     """
#     Estimate the number of emitters within a specific ROI based on the provided timestamps by computing the coherence on the ROI.

#     Args:
#         prop: The properties of the ROI (e.g., position, size).
#         timestamp_data: The timestamps of detected events within the ROI. This is an

#     Returns:
#         int: The estimated number of emitters.
#     """

#     if len(timestamp_data) == 0:
#         return 0

#     # First add up all timestamps on the pixels of the ROI
#     coords = prop.coords  # numpy array of shape (N, 2), with (row, col) for the pixels
#     all_timestamps = collect_timestamps(timestamp_data, coords)

#     # Then compute the (normalized) autocoherence of the summed timestamps
#     roi_coherence = auto_coherence(all_timestamps, interval=dwell_time_ns, bin_size=0.1, nr_steps=200, normalize=False)

#     # Then fit the autocoherence to determine an estimate for nr of emitters


# def collect_timestamps(data, scan_positions):
#     """
#     Collects all timestamps for a list of scanning positions.
    
#     Parameters
#     ----------
#     data : dict
#         Nested dictionary: {scan_position: {camera_pixel: array([[pixel_id, timestamp], ...])}}
#     scan_positions : list of tuple
#         List of scanning positions (e.g. [(14, 19), (15, 19)])
    
#     Returns
#     -------
#     np.ndarray
#         1D array of timestamps combined from all camera pixels at those scan positions.
#     """
#     timestamps = []
    
#     for pos in scan_positions:
#         if pos in data:
#             for cam_id, arr in data[pos].items():
#                 if arr.size > 0:  # skip empty arrays
#                     timestamps.append(arr[:, 1])  # take only the timestamp column
    
#     if timestamps:
#         return np.concatenate(timestamps)
#     else:
#         return np.array([])
