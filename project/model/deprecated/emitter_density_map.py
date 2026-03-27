import numpy as np
from project.model.helper_functions import select_neighbours


def get_map(estimated_nr_emitters, sensor, round_to_int=False, intensity=None, intensity_threshold=0.1,
                            outliers_threshold=-1) -> np.ndarray:
    """
    Calculates a map of the emitter density. Each pixel contains a number, indicating the amount of emitters that
    contributed to the signal at that pixel.

    Parameters
    ----------
    estimated_nr_emitters : np.ndarray
        The number of emitters that is estimated per pixel.
    round_to_int : bool
        If True, the emitter estimates are rounded to integers. If False, the result is given in floats. Default is
        True.
    intensity : np.ndarray
        If provided, the emitter density is corrected with the intensity. Where the intensity is 0 or very low, the
        emitter density it set to zero. Default is None, in which case no correction is performed and the result
        contains value 1 where the estimate is 0 or 1.
    intensity_threshold : float
        The relative threshold for the intensity. If the intensity is lower than this threshold, the value is replaced
        by a zero. Default is 0.1. (The correction is controlled by the intensity parameter)
    outliers_threshold : float
        If a value is above this threshold, it is considered to be an outlier and the value is replaced by the 8
        surrounding nearest neighbours. If -1, outliers are not corrected.

    Returns
    -------
    emitter_density : np.ndarray
        A (nr_pixels, nr_pixels) array containing the number of emitters per pixel.
    """
    if intensity is not None:
        estimated_nr_emitters = zeros_correction(estimated_nr_emitters, intensity, intensity_threshold)
    if outliers_threshold != -1:
        estimated_nr_emitters = outliers_interpolation(estimated_nr_emitters, outliers_threshold, sensor=sensor)
    if round_to_int:
        return np.round(estimated_nr_emitters, decimals=0)
    return estimated_nr_emitters


def zeros_correction(emitter_density, intensity, threshold) -> np.ndarray:
    """
    The calculated emitter density map will show value 1 if the estimated number of emitters is either 1 or 0. This
    function corrects this by putting a 0 where the intensity is low/absent.
    Be aware that this function changes the input array.

    Parameters
    ----------
    emitter_density : np.ndarray
        The estimated emitter map.
    intensity : np.ndarray
        The measured intensity.
    threshold : float
        If the relative intensity is below this threshold, the emitter density is zero.

    Returns
    -------
    emitter_density : np.ndarray
        The emitter map corrected with the intensity.
    """
    # TODO: might be better to have an absolute threshold instead of a relative one
    scaled_intensity = intensity / np.amax(intensity)
    emitter_density[scaled_intensity < threshold] = 0
    return emitter_density


# def outliers_interpolation(emitter_density, threshold) -> np.ndarray:
#     """
#     Interpolates the values of the extreme high values and negative values with their neighbours.

#     Parameters
#     ----------
#     emitter_density : np.ndarray
#         The number of emitters per pixel.
#     threshold : float
#         If the estimated number of emitters is larger than this threshold, it is considered an outlier.

#     Returns
#     -------
#     corrected : np.ndarray
#         Emitter density map corrected for outliers.
#     """
#     dim_0, dim_1 = np.shape(emitter_density)
#     outliers = ((emitter_density < 0) | (emitter_density > threshold)).nonzero()
#     corrected = np.copy(emitter_density)
#     for i in range(len(outliers[0])):
#         element_number = outliers[0][i] * dim_1 + outliers[1][i]
#         neighbour_indices = np.array(
#             select_neighbours(size=3, center=element_number, dim_axis_0=dim_0, dim_axis_1=dim_1))
#         neighbours = emitter_density[neighbour_indices // dim_1, neighbour_indices % dim_1]  # list of neighbour values
#         neighbours = neighbours[
#             (neighbours > 0) & (neighbours < threshold)]  # filter out the high values, they don't count in the average
#         total = np.sum(neighbours)
#         corrected[outliers[0][i], outliers[1][i]] = 0 if total == 0 else total / np.size(neighbours)
#     return corrected

def outliers_interpolation(emitter_density, threshold, sensor=None) -> np.ndarray:
    """
    Interpolates the values of the extreme high values and negative values with their neighbours.
    Parameters
    ----------
    emitter_density : np.ndarray
        The number of emitters per pixel.
    threshold : float
        If the estimated number of emitters is larger than this threshold, it is considered an outlier.
    sensor : Sensor, optional
        The sensor object with neighbor information. Required for hexagonal grids.
    Returns
    -------
    corrected : np.ndarray
        Emitter density map corrected for outliers.
    """
    corrected = np.copy(emitter_density)
    
    # Check if we're dealing with a flattened array for a hexagonal grid
    if len(emitter_density.shape) == 1:
        # Hexagonal grid case
        if sensor is None or not hasattr(sensor, '_Spad23__neighbors'):
            raise ValueError("Sensor object with neighbor information required for hexagonal grid")
            
        neighbors_dict = getattr(sensor, '_Spad23__neighbors')
        outliers = np.where((emitter_density < 0) | (emitter_density > threshold))[0]
        
        for i in outliers:
            # Get neighbors of the current pixel
            if i in neighbors_dict:
                neighbour_indices = neighbors_dict[i]
                # Get neighbor values
                neighbours = emitter_density[neighbour_indices]
                # Filter out the outliers
                valid_neighbors = neighbours[(neighbours > 0) & (neighbours < threshold)]
                
                if len(valid_neighbors) > 0:
                    corrected[i] = np.mean(valid_neighbors)
                else:
                    corrected[i] = 0
    
    #  For rectangular grid
    else:
        dim_0, dim_1 = np.shape(emitter_density)
        outliers = ((emitter_density < 0) | (emitter_density > threshold)).nonzero()
        
        for i in range(len(outliers[0])):
            element_number = outliers[0][i] * dim_1 + outliers[1][i]
            
            # Use sensor's neighbor information if available
            if sensor is not None and hasattr(sensor, '_Spad23__neighbors'):
                neighbors_dict = getattr(sensor, '_Spad23__neighbors')
                if element_number in neighbors_dict:
                    neighbour_indices = neighbors_dict[element_number]
                    # Convert to 2D indices
                    neighbor_rows = [idx // dim_1 for idx in neighbour_indices]
                    neighbor_cols = [idx % dim_1 for idx in neighbour_indices]
                    neighbours = emitter_density[neighbor_rows, neighbor_cols]
                else:
                    # If the element isn't in the dictionary, use original approach
                    neighbour_indices = np.array(
                        select_neighbours(size=3, center=element_number, dim_axis_0=dim_0, dim_axis_1=dim_1))
                    neighbours = emitter_density[neighbour_indices // dim_1, neighbour_indices % dim_1]
            else:
                # Original rectangular grid approach
                neighbour_indices = np.array(
                    select_neighbours(size=3, center=element_number, dim_axis_0=dim_0, dim_axis_1=dim_1))
                neighbours = emitter_density[neighbour_indices // dim_1, neighbour_indices % dim_1]
            
            # Filter out the outliers
            neighbours = neighbours[(neighbours > 0) & (neighbours < threshold)]
            
            if len(neighbours) > 0:
                corrected[outliers[0][i], outliers[1][i]] = np.mean(neighbours)
            else:
                corrected[outliers[0][i], outliers[1][i]] = 0
                
    return corrected
