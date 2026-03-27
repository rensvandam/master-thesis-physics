from tabnanny import verbose
import numpy as np
from numba import njit, prange
from scipy.optimize import minimize
from scipy.ndimage import maximum_filter, label
from scipy import ndimage
from skimage import measure, filters
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional

from project.model.helper_functions import gaussian_2d, get_psf_params, get_psf_params_robust, get_psf_params_moments, get_psf_params_weighted, get_psf_params_peak_preserving
from project.simulations.examples.run_scanning_experiment import run_scanning_experiment

def find_initial_positions_roi(I_meas: np.ndarray, Gd_meas: np.ndarray, n_emitters_map: np.ndarray,
                              min_distance: int = 5,
                              threshold_rel: float = 0.5,
                              threshold_nemitters: float = 0.5,
                              max_emitters: int = 20,
                              roi_min_size: int = 5,
                              roi_max_size: int = 100,
                              intensity_threshold: Optional[float] = None,
                              placement_strategy: str = 'distributed') -> List[Dict]:
    """
    Enhanced method that combines ROI analysis with local maxima detection to find initial positions
    for emitters.
    
    Parameters:
    -----------
    I_meas : np.ndarray
        Measured intensity map
    Gd_meas : np.ndarray
        Measured G_d correlation map (g²(∞) - g²(0))
    n_emitters_map : np.ndarray
        Map indicating the estimated number of emitters at each position
    min_distance : int
        Minimum distance between local maxima (in pixels)
    threshold_rel : float
        Relative threshold for considering local maxima
    threshold_nemitters : float
        Threshold for considering a pixel as containing emitters
    max_emitters : int
        Maximum number of emitters to return
    roi_min_size : int
        Minimum size of ROI in pixels
    roi_max_size : int
        Maximum size of ROI in pixels
    intensity_threshold : float, optional
        Intensity threshold for ROI detection. If None, Otsu's method is used.
    placement_strategy : str
        Strategy for placing multiple emitters within an ROI:
        - 'centroid': Place all emitters at the ROI centroid with small random offsets
        - 'distributed': Distribute emitters within the ROI based on intensity
        - 'maxima': Place emitters at local maxima within the ROI
    
    Returns:
    --------
    List[Dict]
        List of dictionaries containing initial guesses for each emitter
    """
    # Find ROIs first
    roi_stats, roi_labels = roi_n_emitters(
        I_meas, Gd_meas, n_emitters_map,
        intensity_threshold=intensity_threshold,
        min_size=roi_min_size, 
        max_size=roi_max_size,
        plot_results=False  # Disable plotting for now
    )
    #print(roi_stats)
    # Sort ROIs by intensity_mean (brightest first)
    roi_stats.sort(key=lambda x: x['intensity_mean'], reverse=True)
    
    # Limit to max_emitters ROIs
    roi_stats = roi_stats[:max_emitters]
    
    initial_positions = []
    
    # Process each ROI
    for roi in roi_stats:
        roi_label = roi['label']
        roi_mask = (roi_labels == roi_label)
        #estimated_emitters = max(1, int(round(roi['emitters_max'])))

        estimated_emitters = int(round(roi['emitters_max']))
        #print(estimated_emitters)
        # Skip ROIs with too few estimated emitters
        #if estimated_emitters < threshold_nemitters:
        #    print(f"Skipping ROI {roi_label} with estimated emitters {estimated_emitters}")
        #    continue
            
        # Cap the number of emitters per ROI to avoid overloading
        #estimated_emitters = min(estimated_emitters, 10)
        
        print(f"Processing ROI {roi_label} with estimated emitters {estimated_emitters}")
        if placement_strategy == 'centroid':
            # Place all emitters at the centroid with small random offsets
            y_center, x_center = roi['centroid']
            
            for j in range(estimated_emitters):
                # Add a small offset to the x and y coordinates for each emitter
                offset_x = np.random.uniform(-0.5, 0.5)
                offset_y = np.random.uniform(-0.5, 0.5)
                
                initial_positions.append({
                    'x': x_center + offset_x,
                    'y': y_center + offset_y,
                    'I_amplitude': roi['intensity_mean'],
                    'Gd_amplitude': roi['Gd_mean'],
                    'roi_label': roi_label
                })
                
        elif placement_strategy == 'maxima':
            # Find local maxima within this ROI
            masked_intensity = np.zeros_like(I_meas)
            masked_intensity[roi_mask] = I_meas[roi_mask]
            
            x_maxima, y_maxima, heights = find_local_maxima(
                masked_intensity, 
                min_distance=min(min_distance, int(np.sqrt(roi['area'])/2)), 
                threshold_rel=0.3
            )
            
            # Sort maxima by height (brightest first)
            sorted_indices = np.argsort(-heights)
            
            # Get positions for emitters
            num_maxima = len(sorted_indices)
            num_positions = min(num_maxima, estimated_emitters)
            
            # If we have enough maxima, use them
            if num_positions > 0:
                for j in range(num_positions):
                    idx = sorted_indices[j % num_maxima]  # Cycle through maxima if needed
                    x, y = x_maxima[idx], y_maxima[idx]
                    
                    # Add a small offset
                    offset_x = np.random.uniform(-0.2, 0.2)
                    offset_y = np.random.uniform(-0.2, 0.2)
                    
                    initial_positions.append({
                        'x': x + offset_x,
                        'y': y + offset_y,
                        'I_amplitude': I_meas[y, x],
                        'Gd_amplitude': Gd_meas[y, x],
                        'roi_label': roi_label
                    })
            else:
                # Fallback to centroid if no maxima found
                y_center, x_center = roi['centroid']
                
                for j in range(estimated_emitters):
                    offset_x = np.random.uniform(-0.5, 0.5)
                    offset_y = np.random.uniform(-0.5, 0.5)
                    
                    initial_positions.append({
                        'x': x_center + offset_x,
                        'y': y_center + offset_y,
                        'I_amplitude': roi['intensity_mean'],
                        'Gd_amplitude': roi['Gd_mean'],
                        'roi_label': roi_label
                    })
                
        elif placement_strategy == 'distributed':
            # Find a probability distribution based on intensity within the ROI
            roi_y_indices, roi_x_indices = np.where(roi_mask)
            roi_intensities = I_meas[roi_mask]
            
            # Use intensity as probability (add small value to avoid zeros)
            probs = roi_intensities + 1e-10
            probs = probs / np.sum(probs)
            
            # Sample positions according to intensity distribution
            sampled_indices = np.random.choice(
                len(roi_intensities), 
                size=estimated_emitters, 
                replace=True,
                p=probs
            )
            
            for idx in sampled_indices:
                x, y = roi_x_indices[idx], roi_y_indices[idx]
                
                # Add a small offset
                offset_x = np.random.uniform(-0.2, 0.2)
                offset_y = np.random.uniform(-0.2, 0.2)
                
                initial_positions.append({
                    'x': x + offset_x,
                    'y': y + offset_y,
                    'I_amplitude': I_meas[y, x],
                    'Gd_amplitude': Gd_meas[y, x],
                    'roi_label': roi_label
                })
    
    return initial_positions

def find_initial_positions_2(I_meas: np.ndarray, Gd_meas: np.ndarray, n_emitters_map: np.ndarray, 
                             min_distance: int = 5, 
                             threshold_rel: float = 0.5,
                             threshold_nemitters: float = 0.5,
                             max_emitters: int = 10) -> List[Dict]:
    """
    Finds initial positions for emitters based on the number of emitters map and intensity map.

    Parameters:
    -----------
    I_meas : np.ndarray
        Measured intensity map
    n_emitters_map : np.ndarray
        Map indicating the estimated number of emitters at each position
    min_distance : int
        Minimum distance between local maxima (in pixels). See scipy.ndimage.maximum_filter for details.
    
    Returns:
    --------
    List[Dict]
        List of dictionaries containing initial guesses for each emitter
    """
    I_max = maximum_filter(I_meas, size=min_distance, mode='constant')
    I_max_mask = (I_meas == I_max) & (I_meas > threshold_rel * np.max(I_meas))

    # quickly visualize 
    
    # Label and count the local maxima
    labels, num_features = label(I_max_mask)
    
    # Extract coordinates of local maxima
    y_coords, x_coords = np.nonzero(I_max_mask)
    heights = I_meas[y_coords, x_coords]

    # Sort by height and take the top max_emitters
    sorted_indices = np.argsort(-heights)  # Descending order

    # Limit to max_emitters
    num_to_use = min(len(sorted_indices), max_emitters)
    sorted_indices = sorted_indices[:num_to_use]

    initial_positions = []



    # At every local maximum, check the number of emitters in the n_emitters_map.
    for i in sorted_indices:
        x, y = x_coords[i], y_coords[i]
        I_val = I_meas[y, x]
        Gd_val = Gd_meas[y, x]
        n_emitters = n_emitters_map[y, x]

        # Only include if also significant in Gd map
        if Gd_val < threshold_rel * np.max(Gd_meas):
            continue
        if n_emitters > threshold_nemitters:
            for j in range(int(n_emitters)):
                # Add a small offset to the x and y coordinates for each emitter
                offset_x = np.random.uniform(-0.1, 0.1)
                offset_y = np.random.uniform(-0.1, 0.1)
                initial_positions.append({
                    'x': x + offset_x,
                    'y': y + offset_y,
                    'I_amplitude': I_val,
                    'Gd_amplitude': Gd_val
                })
    
    return initial_positions

def roi_n_emitters(I_meas: np.ndarray, Gd_meas: np.ndarray, n_emitters_map: np.ndarray, 
                   intensity_threshold=None, min_size=3, max_size=10000, plot_results=True, verbose=False):
    """
    Finds the regions of interest (ROIs) in the measurement and estimates the number of emitters in each ROI.
    
    Parameters:
    -----------
    I_meas : np.ndarray
        Measured intensity map
    Gd_meas : np.ndarray
        Measured G_d correlation map (g²(∞) - g²(0))
    n_emitters_map : np.ndarray
        Map with the estimated number of emitters per pixel
    intensity_threshold : float, optional
        Threshold for intensity. If None, Otsu's method is used
    min_size : int, optional
        Minimum size of ROI in pixels
    max_size : int, optional
        Maximum size of ROI in pixels
    plot_results : bool, optional
        Whether to plot the results
        
    Returns:
    --------
    roi_stats : list of dicts
        List of dictionaries containing statistics for each ROI:
        - 'label': Label of the ROI
        - 'centroid': (y, x) coordinates of ROI centroid
        - 'area': Area of ROI in pixels
        - 'intensity_mean': Mean intensity in ROI
        - 'emitters_mean': Mean number of emitters in ROI
        - 'emitters_sum': Sum of emitters in ROI
    labels : np.ndarray
        Labeled image where each ROI has a unique integer value
    """
    # Make sure all input arrays have the same shape
    assert I_meas.shape == Gd_meas.shape == n_emitters_map.shape, "All input arrays must have the same shape"

    #
    # --- GMM-based threshold ---
    def gmm_threshold(image, n_components=2):
        pixels = image.reshape(-1, 1)
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(pixels)

        # Use means to determine which component is foreground
        means = gmm.means_.flatten()
        sorted_indices = np.argsort(means)
        # Pick the lowest mean between-class point as threshold
        threshold = np.mean(means[sorted_indices])
        return threshold

    if verbose:
        print("Finding ROIs based on intensity and Gd maps...")
    # Use Otsu's method to find threshold if not provided
    if intensity_threshold is None:
        #intensity_threshold = filters.threshold_otsu(Gd_meas)
        intensity_threshold = gmm_threshold(Gd_meas)
    
    if verbose:
        print(f"Using intensity threshold for finding regions (in Gd_meas): {intensity_threshold:.2f}")
    #print(intensity_threshold)
    # Create binary mask from intensity image
    binary_mask = Gd_meas > intensity_threshold
    
    #show binary mask
    if plot_results:
        plt.figure(figsize=(6, 6))
        plt.imshow(binary_mask, cmap='gray')
        plt.show()

    # Label connected regions in the binary mask
    labels, num_labels = ndimage.label(binary_mask)
    
    # Measure properties of labeled regions
    props = measure.regionprops(labels, intensity_image=Gd_meas)
    
    # for i,region in enumerate(props):
    #     intensity_image = region.intensity_image
    #     mask = region.image
    #     intensities = intensity_image[mask]
    #     plt.figure()
    #     plt.hist(intensities, bins=50, color='blue', alpha=0.7)
    #     plt.title(f'Histogram of Gd for Region {i+1}')
    #     plt.xlabel('Gd')
    #     plt.ylabel('Frequency')
    #     plt.grid(True)
    #     plt.show()

    # Filter regions based on size
    roi_stats = []
    valid_labels = set()
    
    for prop in props:
        if min_size <= prop.area <= max_size:
            # Get the region mask
            region_mask = (labels == prop.label)
            
            # Calculate statistics for this ROI
            roi_stats.append({
                'label': prop.label,
                'centroid': prop.centroid,  # (y, x) coordinates
                'area': prop.area,
                'intensity_mean': np.mean(I_meas[region_mask]),
                'emitters_max': np.max(n_emitters_map[region_mask]),
                'emitters_mean': np.mean(n_emitters_map[region_mask]),
                'emitters_median': np.median(n_emitters_map[region_mask]),
                'emitters_sum': np.sum(n_emitters_map[region_mask]),
                'emitters_estimate': np.sum(Gd_meas[region_mask])/820,
                'Gd_sum': np.sum(Gd_meas[region_mask])
            })
            valid_labels.add(prop.label)
    
    # Create new labels with only valid regions
    filtered_labels = np.zeros_like(labels)
    for label in valid_labels:
        filtered_labels[labels == label] = label
    
    # Plotting if requested
    if plot_results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot intensity image with ROI boundaries
        axes[0].imshow(Gd_meas, cmap='hot')
        axes[0].set_title('Intensity with ROIs')
        
        # Plot labeled regions
        axes[1].imshow(filtered_labels, cmap='nipy_spectral')
        axes[1].set_title(f'Detected ROIs (n={len(roi_stats)})')
        
        # Plot number of emitters map
        im = axes[2].imshow(n_emitters_map, cmap='viridis')
        axes[2].set_title('Number of Emitters Map')
        plt.colorbar(im, ax=axes[2], label='Estimated emitters')
        
        # Annotate ROIs with their stats
        for roi in roi_stats:
            y, x = roi['centroid']
            axes[0].plot(x, y, 'x', color='cyan')
            axes[0].annotate(f"{roi['label']}", (x, y), color='cyan')
            
            axes[2].plot(x, y, 'x', color='red')
            axes[2].annotate(f"{roi['label']}", (x, y), color='red')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics in a table
        print("\nROI Statistics:")
        print("-" * 80)
        print(f"{'Label':^6} | {'Area':^6} | {'X':^6} | {'Y':^6} | {'Mean Int':^10} | {'Mean Emitters':^14} | {'Median Emitters':^14} | {'Max Emitters':^14} | {'Total Emitters':^14} | {'Gd Sum':^10}") 
        print("-" * 80)
        
        for roi in roi_stats:
            y, x = roi['centroid']
            print(f"{roi['label']:^0f} | {roi['area']:^0f} | {x:^6.1f} | {y:^6.1f} | {roi['intensity_mean']:^10.2f} | {roi['emitters_mean']:^14.2f} | {roi['emitters_median']:^14.2f} | {roi['emitters_max']:^14.2f} | {roi['emitters_sum']:^14.2f} | {roi['Gd_sum']:^10.2f}")
    
    return roi_stats, filtered_labels

def model_intensity_2(params: np.ndarray, shape: Tuple[int, int], n_emitters: int, 
                   psf_sigma_I: float = 1.0, psf_amp_I: float = 32000) -> np.ndarray:
    """
    Generate model intensity map based on emitter parameters
    
    Parameters:
    -----------
    params : np.ndarray
        Array of parameters [A1, x1, y1, B1, A2, x2, y2, B2, ...]
    shape : Tuple[int, int]
        Shape of the output map
    n_emitters : int
        Number of emitters
    psf_sigma_I : float
        PSF width for intensity
        
    Returns:
    --------
    np.ndarray
        Model intensity map
    """
    height, width = shape
    y, x = np.ogrid[:height, :width]
    
    model = np.zeros((height, width))
    
    for i in range(n_emitters):
        #A = params[i*4]
        xe = params[i*2]
        ye = params[i*2 + 1]
        #B = params[i*4 + 3]
        
        model += gaussian_2d((x,y), psf_amp_I, xe, ye, psf_sigma_I, psf_sigma_I, 0, flatten=False)

    #the max value is 20000 due to deadtime saturation
    #model = np.clip(model, 0, 20000)  # Clip to max value of 32000
    
    return model

def model_Gd_2(params: np.ndarray, shape: Tuple[int, int], n_emitters: int, 
            psf_sigma_G: float = 0.7, psf_amp_G: float = 32000**2/1000000) -> np.ndarray:
    """
    Generate model G_d map based on emitter parameters
    
    Parameters:
    -----------
    params : np.ndarray
        Array of parameters [A1, x1, y1, B1, A2, x2, y2, B2, ...]
    shape : Tuple[int, int]
        Shape of the output map
    n_emitters : int
        Number of emitters
    psf_sigma_G : float
        PSF width for G_d (typically tighter than intensity PSF)
        
    Returns:
    --------
    np.ndarray
        Model G_d map
    """
    height, width = shape
    y, x = np.ogrid[:height, :width]
    
    model = np.zeros((height, width))
    
    for i in range(n_emitters):
        #A = params[i*4]
        xe = params[i*2]
        ye = params[i*2 + 1]
        #B = params[i*4 + 3]
        
        # G_d is proportional to intensity squared (A²)
        model += gaussian_2d((x,y), psf_amp_G, xe, ye, psf_sigma_G, psf_sigma_G, 0, flatten=False)

        # if there is deadtime, and there is another emitter closeby, we decrease the signal strength
        # (the amplitude) of the gaussians that overlap as it reaches the deadtime limit.

    # get rid of too high values due to deadtime saturation
    #model = np.clip(model, 0, 250)  # Clip to max value of 32000^2/1000000
    
    return model

@njit(parallel=True, fastmath=True)
def model_intensity_numba(params, height, width, n_emitters, psf_sigma_I=1.0, psf_amp_I=32000):
    model = np.zeros((height, width))
    
    for i in prange(n_emitters):
        xe = params[i*2]
        ye = params[i*2 + 1]
        
        for y in range(height):
            dy2 = (y - ye) ** 2
            for x in range(width):
                dx2 = (x - xe) ** 2
                model[y, x] += psf_amp_I * np.exp(-(dx2 + dy2) / (2.0 * psf_sigma_I**2))
                
    return model


@njit(parallel=True, fastmath=True)
def model_Gd_numba(params, height, width, n_emitters, psf_sigma_G=0.7, psf_amp_G=32000**2/1000000):
    model = np.zeros((height, width))
    
    for i in prange(n_emitters):
        xe = params[i*2]
        ye = params[i*2 + 1]
        
        for y in range(height):
            dy2 = (y - ye) ** 2
            for x in range(width):
                dx2 = (x - xe) ** 2
                model[y, x] += psf_amp_G * np.exp(-(dx2 + dy2) / (2.0 * psf_sigma_G**2))
                
    return model

@njit(fastmath=True)
def rss_objective_numba(params, I_meas, Gd_meas, n_emitters,
                        alpha=0.2, beta=0.8,
                        psf_sigma_I=1.0, psf_sigma_G=0.7,
                        psf_amp_I=32000, psf_amp_G=32000**2/1000000):
    height, width = I_meas.shape
    
    I_model = model_intensity_numba(params, height, width, n_emitters, psf_sigma_I, psf_amp_I)
    Gd_model = model_Gd_numba(params, height, width, n_emitters, psf_sigma_G, psf_amp_G)
    
    RSS_I = np.sum((I_meas - I_model) ** 2)
    RSS_Gd = np.sum((Gd_meas - Gd_model) ** 2)
    
    return alpha * RSS_I + beta * RSS_Gd

def rss_objective(params: np.ndarray, I_meas: np.ndarray, Gd_meas: np.ndarray, 
                 n_emitters: int, alpha: float = 0.2, beta: float = 0.8,
                 psf_sigma_I: float = 1.0, psf_sigma_G: float = 0.7, psf_amp_I: float = 32000, psf_amp_G: float = 32000**2/1000000) -> float:
    """
    Calculate the residual sum of squares (RSS) objective function
    
    Parameters:
    -----------
    params : np.ndarray
        Array of parameters [A1, x1, y1, B1, A2, x2, y2, B2, ...]
    I_meas : np.ndarray
        Measured intensity map
    Gd_meas : np.ndarray
        Measured G_d correlation map
    n_emitters : int
        Number of emitters
    alpha : float
        Weight for intensity RSS
    beta : float
        Weight for G_d RSS
    psf_sigma_I, psf_sigma_G : float
        PSF widths for intensity and G_d
        
    Returns:
    --------
    float
        Total weighted RSS
    """
    shape = I_meas.shape
    
    # Generate model maps
    I_model = model_intensity_2(params, shape, n_emitters, psf_sigma_I, psf_amp_I)
    Gd_model = model_Gd_2(params, shape, n_emitters, psf_sigma_G, psf_amp_G)

    # Calculate residuals
    RSS_I = np.sum((I_meas - I_model)**2)
    RSS_Gd = np.sum((Gd_meas - Gd_model)**2)
    
    # Combined RSS
    RSS = alpha * RSS_I + beta * RSS_Gd

    return RSS

def optimize_positions_2(I_meas: np.ndarray, Gd_meas: np.ndarray, 
                       initial_positions: List[Dict],
                       alpha: float = 0.5, beta: float = 0.5,
                       psf_sigma_I: float = 1.0, psf_sigma_G: float = 0.7,
                       psf_amp_I: float = 32000, psf_amp_G: float = 32000**2/1000000,
                       regularization_weight: float = 0.0,
                       bounds_slack: float = 10.0, verbose=False) -> Dict:
    """
    Optimize emitter positions by minimizing the RSS
    
    Parameters:
    -----------
    I_meas : np.ndarray
        Measured intensity map
    Gd_meas : np.ndarray
        Measured G_d correlation map
    initial_positions : List[Dict]
        Initial position guesses from find_initial_positions
    alpha, beta : float
        Weights for intensity and G_d RSS
    psf_sigma_I, psf_sigma_G : float
        PSF widths for intensity and G_d
    psf_amp_I, psf_amp_G : float
        PSF amplitudes for intensity and G_d
    regularization_weight : float
        Weight for regularization term (can be used to penalize large #emitter values)
    bounds_slack : float
        Range around initial position to search (in pixels)
    verbose : bool
        Whether to print optimization details
    Returns:
    --------
    Dict
        Optimization results including optimized parameters and RSS
    """
    n_emitters = len(initial_positions)
    gamma = regularization_weight

    if n_emitters == 0:
        raise ValueError("No initial positions provided for optimization.")
    if verbose:
        print(f"Optimizing {n_emitters} emitters with alpha={alpha}, beta={beta}")
    # Flatten initial parameters for optimizer
    initial_params = []
    param_bounds = []
    
    for pos in initial_positions:
        # Parameters: A, x, y, B
        initial_params.extend([pos['x'], pos['y']])
        
        #x_bounds = (0, I_meas.shape[1] - 1)
        #y_bounds = (0, I_meas.shape[0] - 1) 

        # Bounds for each parameter
        x_bounds = (max(0, pos['x'] - bounds_slack), min(I_meas.shape[1] - 1, pos['x'] + bounds_slack))
        y_bounds = (max(0, pos['y'] - bounds_slack), min(I_meas.shape[0] - 1, pos['y'] + bounds_slack))
        
        # A (amplitude) should be positive, B (background) can be anywhere
        param_bounds.extend([x_bounds, y_bounds])
    
    # Convert to numpy array
    initial_params = np.array(initial_params)

    if verbose:
        #print(f"Initial parameters: {initial_params}")
        print(f"Parameter bounds: {param_bounds}")
    
    # Optimize
    #print("Starting optimization of emitter positions... gtol: 1e-5")
    result = minimize(
        rss_objective,
        initial_params,
        args=(I_meas, Gd_meas, n_emitters, alpha, beta, psf_sigma_I, psf_sigma_G, psf_amp_I, psf_amp_G),
        method='L-BFGS-B',
        bounds=param_bounds,
        options={'gtol': 1e-6}, #gradient tolerance of stopping. default is 1e-5.
    )
    #print("Optimization completed.")
    
    # Extract optimized parameters
    opt_params = result.x
    
    # Calculate final models
    I_model = model_intensity_2(opt_params, I_meas.shape, n_emitters, psf_sigma_I, psf_amp_I=psf_amp_I)
    Gd_model = model_Gd_2(opt_params, I_meas.shape, n_emitters, psf_sigma_G, psf_amp_G=psf_amp_G)

    # Calculate final RSS
    RSS_I = np.sum((I_meas - I_model)**2)
    RSS_Gd = np.sum((Gd_meas - Gd_model)**2)
    RSS = alpha * RSS_I + beta * RSS_Gd + gamma * np.sum(opt_params**2)
    
    # Format the output
    emitters = []
    for i in range(n_emitters):
        emitters.append({
            #'amplitude': opt_params[i*4],
            'x': opt_params[i*2],
            'y': opt_params[i*2 + 1],
            #'background': opt_params[i*4 + 3]
        })
    
    return {
        'emitters': emitters,
        'RSS': RSS,
        'RSS_I': RSS_I,
        'RSS_Gd': RSS_Gd,
        'I_model': I_model,
        'Gd_model': Gd_model,
        'opt_success': result.success,
        'opt_message': result.message
    }

def optimize_positions_3(I_meas: np.ndarray, Gd_meas: np.ndarray,
                         initial_positions: List[Dict],
                         alpha: float = 0.5, beta: float = 0.5,
                         psf_sigma_I: float = 1.0, psf_sigma_G: float = 0.7,
                         regularization_weight: float = 0.0,
                         bounds_slack: float = 10.0,
                         verbose: bool = False) -> Dict:
    """
    Optimize emitter positions, amplitudes, and backgrounds by minimizing RSS.

    Parameters
    ----------
    I_meas : np.ndarray
        Measured intensity map
    Gd_meas : np.ndarray
        Measured G_d correlation map
    initial_positions : List[Dict]
        Initial guesses: [{'x':..., 'y':..., 'I_amplitude':..., 'Gd_amplitude':..., 'roi_label':...}, ...]
    alpha, beta : float
        Weights for intensity and G_d RSS
    psf_sigma_I, psf_sigma_G : float
        PSF sigmas (in pixels)
    regularization_weight : float
        Weight for regularization term
    bounds_slack : float
        Range around initial position to search (in pixels)
    verbose : bool
        Print debug info

    Returns
    -------
    Dict with optimized parameters and results
    """
    n_emitters = len(initial_positions)
    gamma = regularization_weight

    if n_emitters == 0:
        raise ValueError("No initial positions provided for optimization.")

    # Flatten initial parameters
    initial_params = []
    param_bounds = []

    for pos in initial_positions:
        A0 = pos.get('I_amplitude', np.max(I_meas))  # default if not provided
        x0 = pos['x']
        y0 = pos['y']
        B0 = pos.get('background', 0.0)

        initial_params.extend([A0, x0, y0, B0])

        A_bounds = (0, None)  # amplitude non-negative
        x_bounds = (max(0, x0 - bounds_slack), min(I_meas.shape[1] - 1, x0 + bounds_slack))
        y_bounds = (max(0, y0 - bounds_slack), min(I_meas.shape[0] - 1, y0 + bounds_slack))
        B_bounds = (0, None)  # background non-negative (change to (-np.inf, np.inf) if you allow neg)

        param_bounds.extend([A_bounds, x_bounds, y_bounds, B_bounds])

    initial_params = np.array(initial_params)

    if verbose:
        print(f"Optimizing {n_emitters} emitters with alpha={alpha}, beta={beta}")
        print(f"Initial params: {initial_params}")
        print(f"Bounds: {param_bounds}")

    # Objective function
    def rss_objective_3(params):
        I_model = model_intensity_3(params, I_meas.shape, n_emitters, psf_sigma_I)
        Gd_model = model_Gd_3(params, I_meas.shape, n_emitters, psf_sigma_G)

        RSS_I = np.sum((I_meas - I_model) ** 2)
        RSS_Gd = np.sum((Gd_meas - Gd_model) ** 2)
        RSS = alpha * RSS_I + beta * RSS_Gd + gamma * np.sum(params**2)
        return RSS

    # Optimize
    result = minimize(
        rss_objective_3,
        initial_params,
        method='L-BFGS-B',
        bounds=param_bounds,
        options={'gtol': 1e-12, 'eps': 1e-8, 'maxiter': 5000, 'disp': verbose},
    )

    opt_params = result.x

    # Build emitter results
    emitters = []
    for i in range(n_emitters):
        emitters.append({
            'amplitude': opt_params[i*4],
            'x': opt_params[i*4 + 1],
            'y': opt_params[i*4 + 2],
            'background': opt_params[i*4 + 3],
        })

    # Recompute final RSS components
    I_model = model_intensity_3(opt_params, I_meas.shape, n_emitters, psf_sigma_I)
    Gd_model = model_Gd_3(opt_params, I_meas.shape, n_emitters, psf_sigma_G)
    RSS_I = np.sum((I_meas - I_model)**2)
    RSS_Gd = np.sum((Gd_meas - Gd_model)**2)

    return {
        'emitters': emitters,
        'RSS': alpha * RSS_I + beta * RSS_Gd,
        'RSS_I': RSS_I,
        'RSS_Gd': RSS_Gd,
        'I_model': I_model,
        'Gd_model': Gd_model,
        'opt_success': result.success,
        'opt_message': result.message
    }

def visualize_results(I_meas: np.ndarray, Gd_meas: np.ndarray, 
                     optimization_result: Dict,
                     initial_positions: Optional[List[Dict]] = None,
                     area_size: str = None, pixel_size: float = 1,
                     true_positions: List = None) -> None:
    """
    Visualize the results of the optimization
    
    Parameters:
    -----------
    I_meas : np.ndarray
        Measured intensity map
    Gd_meas : np.ndarray
        Measured G_d correlation map
    optimization_result : Dict
        Results from optimize_positions
    initial_positions : Optional[List[Dict]]
        Initial position guesses
    """
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    area_size = area_size.split('x')
    area_size = (float(area_size[0]), float(area_size[1]))
    # Intensity maps
    im0 = axs[0, 0].imshow(I_meas, cmap='hot', extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2], origin='lower')
    axs[0, 0].set_title('I Measured')
    plt.colorbar(im0, ax=axs[0, 0])
    
    im1 = axs[0, 1].imshow(optimization_result['I_model'], cmap='hot', extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2], origin='lower')
    axs[0, 1].set_title('I Model')
    plt.colorbar(im1, ax=axs[0, 1])
    
    im2 = axs[0, 2].imshow((I_meas - optimization_result['I_model']), cmap='bwr', extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2], origin='lower')
    axs[0, 2].set_title('I Residuals')
    plt.colorbar(im2, ax=axs[0, 2])
    
    # G_d maps
    im3 = axs[1, 0].imshow(Gd_meas, cmap='viridis', extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2], origin='lower')
    axs[1, 0].set_title('G_d Measured')
    plt.colorbar(im3, ax=axs[1, 0])
    
    im4 = axs[1, 1].imshow(optimization_result['Gd_model'], cmap='viridis', extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2], origin='lower')
    axs[1, 1].set_title('G_d Model')
    plt.colorbar(im4, ax=axs[1, 1])
    
    im5 = axs[1, 2].imshow((Gd_meas - optimization_result['Gd_model']), cmap='bwr', extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2], origin='lower')
    axs[1, 2].set_title('G_d Residuals')
    plt.colorbar(im5, ax=axs[1, 2])
    
    # Mark emitter positions
    for ax in [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]:
        # Plot initial positions if provided
        if initial_positions:
            init_x = [pos['x'] for pos in initial_positions]
            init_x = [-area_size[0]/2 + pos['x'] * pixel_size for pos in initial_positions]
            init_y = [pos['y'] for pos in initial_positions]
            init_y = [-area_size[1]/2 + pos['y'] * pixel_size for pos in initial_positions]
            ax.scatter(init_x, init_y, s=50, facecolors='none', edgecolors='red', 
                      linewidths=1.5, label='Initial')
            
        if true_positions:
            true_y = [pos[0] for pos in true_positions]
            true_x = [pos[1] for pos in true_positions]
            ax.scatter(true_x, true_y, s=50, facecolors='none', edgecolors='blue', 
                      linewidths=1.5, label='True')
        
        # Plot optimized positions
        opt_x = [emitter['x'] for emitter in optimization_result['emitters']]
        opt_x = [-area_size[0]/2 + emitter['x'] * pixel_size for emitter in optimization_result['emitters']]
        opt_y = [emitter['y'] for emitter in optimization_result['emitters']]
        opt_y = [-area_size[1]/2 + emitter['y'] * pixel_size for emitter in optimization_result['emitters']]
        ax.scatter(opt_x, opt_y, s=80, facecolors='none', edgecolors='lime', 
                  linewidths=1.5, label='Optimized')
        
        
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print(f"Total RSS: {optimization_result['RSS']:.4f}")
    print(f"RSS_I: {optimization_result['RSS_I']:.4f}")
    print(f"RSS_Gd: {optimization_result['RSS_Gd']:.4f}")
    print("\nOptimized emitter positions:")
    
    for i, emitter in enumerate(optimization_result['emitters']):
        print(f"Emitter {i+1}: x={emitter['x']:.2f}, y={emitter['y']:.2f}") #, amp={emitter['amplitude']:.2f}, bg={emitter['background']:.2f}")

def optimize_with_greedy_approach(I_meas: np.ndarray, Gd_meas: np.ndarray, 
                                 n_emitters_map: np.ndarray,
                                 min_distance: int = 3,
                                 threshold_rel: float = 0.05,
                                 threshold_nemitters: float = 0.1,
                                 intensity_threshold: float = 5000,
                                 max_emitters_total: int = 15,
                                 max_variations: int = 3,
                                 alpha: float = 0.5, 
                                 beta: float = 0.5,
                                 psf_sigma_I: float = 1.0, 
                                 psf_sigma_G: float = 0.7,
                                 psf_amp_I: float = 32000,
                                 psf_amp_G: float = 32000**2/1000000,
                                 bounds_slack: float = 10.0,
                                 reg_weight: float = 0.0,
                                 plot_results: bool = False,
                                 verbose: bool = False,
                                 placement_strategy: str = 'distributed') -> Dict:
    """
    Optimize emitter detection using a greedy approach. Start with base configuration then
    iteratively try variations to improve the RSS.
    
    Parameters:
    -----------
    I_meas : np.ndarray
        Measured intensity map
    Gd_meas : np.ndarray
        Measured G_d correlation map
    n_emitters_map : np.ndarray
        Map indicating the estimated number of emitters at each position
    min_distance : int
        Minimum distance between local maxima (in pixels)
    threshold_rel : float
        Relative threshold for considering local maxima
    threshold_nemitters : float
        Threshold for considering a pixel as containing emitters
    intensity_threshold : float
        Intensity threshold for ROI detection
    max_emitters_total : int
        Maximum total number of emitters to detect
    max_variations : int
        Maximum number of variations to try per iteration
    alpha, beta : float
        Weights for intensity and G_d RSS
    psf_sigma_I, psf_sigma_G : float
        PSF widths for intensity and G_d in 
    bounds_slack : float
        Range around initial position to search
    placement_strategy : str
        Strategy for placing emitters within an ROI ('distributed', 'maxima', 'centroid')
    
    Returns:
    --------
    Dict
        Best optimization results including optimized parameters and RSS
    """
    # First, find ROIs
    if verbose:
        print("Finding ROIs and estimating number of emitters...")
    roi_stats, roi_labels = roi_n_emitters(
        I_meas, Gd_meas, n_emitters_map,
        intensity_threshold=intensity_threshold,
        plot_results=plot_results,
        verbose=verbose
    )
    
    # Sort ROIs by intensity_mean (brightest first)
    roi_stats.sort(key=lambda x: x['intensity_mean'], reverse=True)
    #print(roi_stats)
    # Start with base configuration (use estimated number of emitters)
    base_config = []
    total_emitters = 0
    
    for roi in roi_stats:
        #estimated_emitters = max(1, int(round(roi['emitters_max'])))
        #max_emitters_per_roi = 10
        #estimated_emitters = min(int(round(roi['emitters_max'])), max_emitters_per_roi)
        estimated_emitters = int(round(roi['emitters_mean']))
        # Cap at max_emitters_total
        if total_emitters + estimated_emitters <= max_emitters_total:
            base_config.append(estimated_emitters)
            total_emitters += estimated_emitters
        else:
            # If adding would exceed max, add as many as we can
            remaining = max_emitters_total - total_emitters
            if remaining > 0:
                base_config.append(remaining)
                total_emitters += remaining
            else:
                base_config.append(0)
    
    # Evaluate base configuration
    def evaluate_config(config):
        """Helper function to evaluate a configuration"""
        initial_positions = []
        
        for roi_idx, num_emitters in enumerate(config):
            if num_emitters <= 0:
                continue
                
            roi = roi_stats[roi_idx]
            roi_label = roi['label']
            roi_mask = (roi_labels == roi_label)
            
            # Generate positions based on placement strategy (same logic as before)
            if placement_strategy == 'centroid':
                y_center, x_center = roi['centroid']
                for j in range(num_emitters):
                    offset_x = np.random.uniform(-0.5, 0.5)
                    offset_y = np.random.uniform(-0.5, 0.5)
                    initial_positions.append({
                        'x': x_center + offset_x,
                        'y': y_center + offset_y,
                        'I_amplitude': roi['intensity_mean'],
                        'Gd_amplitude': roi['Gd_mean'],
                        'roi_label': roi_label
                    })
                    
            elif placement_strategy == 'maxima':
                # Find local maxima within this ROI
                masked_intensity = np.zeros_like(I_meas)
                masked_intensity[roi_mask] = I_meas[roi_mask]
                
                x_maxima, y_maxima, heights = find_local_maxima(
                    masked_intensity, 
                    min_distance=min(min_distance, int(np.sqrt(roi['area'])/2)), 
                    threshold_rel=0.3
                )
                
                # Sort maxima by height (brightest first)
                sorted_indices = np.argsort(-heights)
                
                # Get positions for emitters
                num_maxima = len(sorted_indices)
                num_positions = min(num_maxima, num_emitters)
                
                # If we have enough maxima, use them
                if num_positions > 0:
                    for j in range(num_positions):
                        idx = sorted_indices[j % num_maxima]  # Cycle through maxima if needed
                        x, y = x_maxima[idx], y_maxima[idx]
                        
                        # Add a small offset
                        offset_x = np.random.uniform(-0.2, 0.2)
                        offset_y = np.random.uniform(-0.2, 0.2)
                        
                        initial_positions.append({
                            'x': x + offset_x,
                            'y': y + offset_y,
                            'I_amplitude': I_meas[y, x],
                            'Gd_amplitude': Gd_meas[y, x],
                            'roi_label': roi_label
                        })
                else:
                    # Fallback to centroid if no maxima found
                    y_center, x_center = roi['centroid']
                    
                    for j in range(num_emitters):
                        offset_x = np.random.uniform(-0.5, 0.5)
                        offset_y = np.random.uniform(-0.5, 0.5)
                        
                        initial_positions.append({
                            'x': x_center + offset_x,
                            'y': y_center + offset_y,
                            'I_amplitude': roi['intensity_mean'],
                            'Gd_amplitude': roi['Gd_mean'],
                            'roi_label': roi_label
                        })
                
            elif placement_strategy == 'distributed':
                # Find a probability distribution based on intensity within the ROI
                roi_y_indices, roi_x_indices = np.where(roi_mask)
                roi_intensities = I_meas[roi_mask]
                
                # Use intensity as probability (add small value to avoid zeros)
                probs = roi_intensities + 1e-10
                probs = probs / np.sum(probs)
                
                # Sample positions according to intensity distribution
                sampled_indices = np.random.choice(
                    len(roi_intensities), 
                    size=num_emitters, 
                    replace=True,
                    p=probs
                )
                
                for idx in sampled_indices:
                    x, y = roi_x_indices[idx], roi_y_indices[idx]
                    
                    # Add a small offset
                    offset_x = np.random.uniform(-2,2) #plixels
                    offset_y = np.random.uniform(-2, 2) #pixels
                    #print(f"Emitter at {x}, {y} with offset {offset_x}, {offset_y}")
                    initial_positions.append({
                        'x': x + offset_x,
                        'y': y + offset_y,
                        'I_amplitude': I_meas[y, x],
                        'Gd_amplitude': Gd_meas[y, x],
                        'roi_label': roi_label
                    })
        
        # Run optimization
        if initial_positions:
            result = optimize_positions_2(
                I_meas, Gd_meas, initial_positions,
                alpha=alpha, beta=beta,
                psf_sigma_I=psf_sigma_I, psf_sigma_G=psf_sigma_G,
                psf_amp_I=psf_amp_I, psf_amp_G=psf_amp_G, regularization_weight=reg_weight,
                bounds_slack=bounds_slack,
                verbose=verbose
            )
            # result = optimize_positions_3(
            #     I_meas, Gd_meas, initial_positions,
            #     alpha=alpha, beta=beta,
            #     psf_sigma_I=psf_sigma_I, psf_sigma_G=psf_sigma_G,
            #     regularization_weight=reg_weight,
            #     bounds_slack=bounds_slack,
            #     verbose=verbose
            # )
            return result
        else:
            # Return dummy result with infinite RSS if no positions
            return {'RSS': float('inf'), 'emitters': []}
    
    # Evaluate base configuration
    if verbose:
        print(f"Evaluating base configuration: {base_config}")
    current_result = evaluate_config(base_config)
    current_config = base_config
    current_rss = current_result['RSS']
    if verbose:
        print(f"Base configuration RSS: {current_rss:.4f}")
    
    # Iteratively improve by trying variations
    improved = True
    iteration = 0
    max_iterations = 15  # Prevent infinite loops
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        if verbose:
            print(f"\nIteration {iteration}:")
        
        # Try variations: add or remove one emitter from each ROI
        for roi_idx in range(min(len(current_config), len(roi_stats))):
            # Skip ROIs with zero emitters that can't be reduced further
            # if current_config[roi_idx] == 0:
            #     continue
            if verbose:
                print(f"Checking ROI {roi_idx} with current config: {current_config[roi_idx]} emitters")
            # Try variations for this ROI
            variations = []
            
            # Try one fewer emitter
            if current_config[roi_idx] > 0:
                minus_config = current_config.copy()
                minus_config[roi_idx] -= 1
                variations.append(minus_config)
            
            # Try one more emitter if not exceeding max_total
            if sum(current_config) < max_emitters_total:
                plus_config = current_config.copy()
                plus_config[roi_idx] += 1
                variations.append(plus_config)
            
            # Evaluate all variations
            for var_config in variations:
                if verbose:
                    print(f"Trying variation: {var_config}")
                var_result = evaluate_config(var_config)
                if verbose:
                    print(f"Variation RSS: {var_result['RSS']:.4f}")
                if var_result['RSS'] < current_rss:
                    current_rss = var_result['RSS']
                    current_result = var_result
                    current_config = var_config
                    improved = True
                    if verbose:
                        print(f"Found improvement! New RSS: {current_rss:.4f}")
                    # Break early if we found an improvement
                    break
            
            if improved:
                # Break the ROI loop and start a new iteration
                break
    
    # Add configuration to result
    current_result['emitter_config'] = current_config
    
    return current_result

def optimize_with_greedy_approach_v2(I_meas: np.ndarray, Gd_meas: np.ndarray, 
                                 n_emitters_map: np.ndarray,
                                 min_distance: int = 3,
                                 threshold_rel: float = 0.05,
                                 threshold_nemitters: float = 0.1,
                                 intensity_threshold: float = 5000,
                                 max_emitters_total: int = 15,
                                 max_variations: int = 3,
                                 alpha: float = 0.5, 
                                 beta: float = 0.5,
                                 psf_sigma_I: float = 1.0, 
                                 psf_sigma_G: float = 0.7,
                                 psf_amp_I: float = 32000,
                                 psf_amp_G: float = 32000**2/1000000,
                                 bounds_slack: float = 10.0,
                                 reg_weight: float = 0.0,
                                 plot_results: bool = False,
                                 verbose: bool = False,
                                 placement_strategy: str = 'distributed') -> Dict:
    """
    Optimized greedy approach for emitter detection. Key improvements:
    - Batch evaluation of variations
    - Smarter termination criteria
    - Reduced redundant calculations
    - Warm starts from previous solutions
    """
    
    # Find ROIs (unchanged)
    if verbose:
        print("Finding ROIs and estimating number of emitters...")
    roi_stats, roi_labels = roi_n_emitters(
        I_meas, Gd_meas, n_emitters_map,
        intensity_threshold=intensity_threshold,
        plot_results=plot_results,
        verbose=verbose
    )
    
    roi_stats.sort(key=lambda x: x['intensity_mean'], reverse=True)
    
    # Initialize base configuration
    base_config = []
    total_emitters = 0
    
    for roi in roi_stats:
        estimated_emitters = int(round(roi['emitters_estimate']))
        if total_emitters + estimated_emitters <= max_emitters_total:
            base_config.append(estimated_emitters)
            total_emitters += estimated_emitters
        else:
            remaining = max_emitters_total - total_emitters
            base_config.append(max(0, remaining))
            total_emitters += max(0, remaining)
    
    # Pre-generate stable initial positions for each ROI to avoid randomness
    roi_position_templates = []
    for roi_idx, roi in enumerate(roi_stats):
        roi_label = roi['label']
        roi_mask = (roi_labels == roi_label)
        
        # Generate more positions than we'll likely need
        max_positions_needed = min(10, max_emitters_total)  # Cap reasonable limit
        positions = []
        
        if placement_strategy == 'distributed':
            roi_y_indices, roi_x_indices = np.where(roi_mask)
            roi_intensities = I_meas[roi_mask]
            probs = roi_intensities + 1e-10
            probs = probs / np.sum(probs)
            
            # Generate deterministic positions based on intensity ranking
            sorted_indices = np.argsort(-roi_intensities)
            for i in range(min(max_positions_needed, len(sorted_indices))):
                idx = sorted_indices[i % len(sorted_indices)]
                x, y = roi_x_indices[idx], roi_y_indices[idx]
                # Use deterministic small offsets instead of random
                offset_x = (i % 3 - 1) * 0.3  # -0.3, 0, 0.3, repeat
                offset_y = ((i // 3) % 3 - 1) * 0.3
                positions.append({
                    'x': x + offset_x,
                    'y': y + offset_y,
                    'I_amplitude': I_meas[y, x],
                    'Gd_amplitude': Gd_meas[y, x],
                    'roi_label': roi_label
                })
        
        elif placement_strategy == 'maxima':
            masked_intensity = np.zeros_like(I_meas)
            masked_intensity[roi_mask] = I_meas[roi_mask]
            
            x_maxima, y_maxima, heights = find_local_maxima(
                masked_intensity, 
                min_distance=min(min_distance, int(np.sqrt(roi['area'])/2)), 
                threshold_rel=0.3
            )
            
            if len(x_maxima) > 0:
                sorted_indices = np.argsort(-heights)
                for i in range(min(max_positions_needed, len(sorted_indices))):
                    idx = sorted_indices[i % len(sorted_indices)]
                    x, y = x_maxima[idx], y_maxima[idx]
                    offset_x = (i % 3 - 1) * 0.1
                    offset_y = ((i // 3) % 3 - 1) * 0.1
                    positions.append({
                        'x': x + offset_x,
                        'y': y + offset_y,
                        'I_amplitude': I_meas[y, x],
                        'Gd_amplitude': Gd_meas[y, x],
                        'roi_label': roi_label
                    })
            else:
                # Fallback to centroid
                y_center, x_center = roi['centroid']
                for i in range(max_positions_needed):
                    offset_x = (i % 3 - 1) * 0.3
                    offset_y = ((i // 3) % 3 - 1) * 0.3
                    positions.append({
                        'x': x_center + offset_x,
                        'y': y_center + offset_y,
                        'I_amplitude': roi['intensity_mean'],
                        'Gd_amplitude': roi['Gd_mean'],
                        'roi_label': roi_label
                    })
        
        else:  # centroid
            y_center, x_center = roi['centroid']
            for i in range(max_positions_needed):
                offset_x = (i % 3 - 1) * 0.3
                offset_y = ((i // 3) % 3 - 1) * 0.3
                positions.append({
                    'x': x_center + offset_x,
                    'y': y_center + offset_y,
                    'I_amplitude': roi['intensity_mean'],
                    'Gd_amplitude': roi['Gd_mean'],
                    'roi_label': roi_label
                })
        
        roi_position_templates.append(positions)
    
    def evaluate_config_fast(config, previous_result=None):
        """Fast configuration evaluation with warm starts"""
        initial_positions = []
        
        # Build positions from templates
        for roi_idx, num_emitters in enumerate(config):
            if num_emitters <= 0 or roi_idx >= len(roi_position_templates):
                continue
            
            # Take the first num_emitters positions from our template
            roi_positions = roi_position_templates[roi_idx][:num_emitters]
            initial_positions.extend(roi_positions)
        
        if not initial_positions:
            return {'RSS': float('inf'), 'emitters': []}
        
        # Use previous result as warm start if available and similar
        use_warm_start = (previous_result is not None and 
                         previous_result['RSS'] != float('inf') and
                         abs(len(previous_result.get('emitters', [])) - len(initial_positions)) <= 2)
        
        if use_warm_start:
            # Start optimization from previous solution but update initial positions
            # This assumes optimize_positions_2 can accept a warm start
            result = optimize_positions_2(
                I_meas, Gd_meas, initial_positions,
                alpha=alpha, beta=beta,
                psf_sigma_I=psf_sigma_I, psf_sigma_G=psf_sigma_G,
                psf_amp_I=psf_amp_I, psf_amp_G=psf_amp_G, 
                regularization_weight=reg_weight,
                bounds_slack=bounds_slack,
                verbose=False  # Reduce verbose output in inner loops
            )
        else:
            result = optimize_positions_2(
                I_meas, Gd_meas, initial_positions,
                alpha=alpha, beta=beta,
                psf_sigma_I=psf_sigma_I, psf_sigma_G=psf_sigma_G,
                psf_amp_I=psf_amp_I, psf_amp_G=psf_amp_G, 
                regularization_weight=reg_weight,
                bounds_slack=bounds_slack,
                verbose=False
            )
        
        return result
    
    # Evaluate base configuration
    if verbose:
        print(f"Evaluating base configuration: {base_config}")
    current_result = evaluate_config_fast(base_config)
    current_config = base_config[:]
    current_rss = current_result['RSS']
    if verbose:
        print(f"Base configuration RSS: {current_rss:.4f}")
    
    # Optimized iterative improvement
    iteration = 0
    max_iterations = min(15, len(roi_stats) * 2)  # Scale with problem size but cap it
    min_improvement = current_rss * 0.001  # Stop if improvement < 0.1%
    stagnation_count = 0
    max_stagnation = 3
    
    while iteration < max_iterations and stagnation_count < max_stagnation:
        iteration += 1
        iteration_improved = False
        best_iteration_rss = current_rss
        best_iteration_config = None
        best_iteration_result = None
        
        if verbose:
            print(f"\nIteration {iteration}:")
        
        # Generate batch of promising variations
        variations_to_try = []
        
        # For each ROI, try both +1 and -1 if valid
        for roi_idx in range(min(len(current_config), len(roi_stats))):
            # Try removing one emitter
            if current_config[roi_idx] > 0:
                minus_config = current_config[:]
                minus_config[roi_idx] -= 1
                variations_to_try.append(minus_config)
            
            # Try adding one emitter
            if sum(current_config) < max_emitters_total:
                plus_config = current_config[:]
                plus_config[roi_idx] += 1
                variations_to_try.append(plus_config)
        
        # Additionally, try some multi-ROI swaps for better exploration
        if len(current_config) >= 2 and iteration <= 5:  # Only early iterations
            for i in range(min(3, len(current_config)-1)):  # Limit swap attempts
                for j in range(i+1, min(i+4, len(current_config))):  # Nearby ROIs
                    if current_config[i] > 0:
                        swap_config = current_config[:]
                        swap_config[i] -= 1
                        swap_config[j] += 1
                        if sum(swap_config) <= max_emitters_total:
                            variations_to_try.append(swap_config)
        
        # Evaluate all variations (this is where we could parallelize in future)
        if verbose:
            print(f"Evaluating {len(variations_to_try)} variations...")
        
        for var_config in variations_to_try:
            var_result = evaluate_config_fast(var_config, current_result)
            
            if var_result['RSS'] < best_iteration_rss:
                improvement = best_iteration_rss - var_result['RSS']
                if improvement > min_improvement:  # Only accept significant improvements
                    best_iteration_rss = var_result['RSS']
                    best_iteration_config = var_config
                    best_iteration_result = var_result
                    iteration_improved = True
        
        # Update current solution if we found improvement
        if iteration_improved:
            current_rss = best_iteration_rss
            current_config = best_iteration_config
            current_result = best_iteration_result
            stagnation_count = 0
            if verbose:
                print(f"Improved! New RSS: {current_rss:.4f}, Config: {current_config}")
        else:
            stagnation_count += 1
            if verbose:
                print(f"No improvement found. Stagnation count: {stagnation_count}")
    
    if verbose:
        print(f"\nOptimization completed after {iteration} iterations")
        print(f"Final RSS: {current_rss:.4f}")
        print(f"Final config: {current_config}")
    
    # Add configuration to result
    current_result['emitter_config'] = current_config
    
    return current_result

def optimize_with_greedy_approach_v3(I_meas: np.ndarray, Gd_meas: np.ndarray, 
                                 n_emitters_map: np.ndarray,
                                 min_distance: int = 3,
                                 threshold_rel: float = 0.05,
                                 threshold_nemitters: float = 0.1,
                                 intensity_threshold: float = 5000,
                                 max_emitters_total: int = 15,
                                 max_variations: int = 3,
                                 alpha: float = 0.5, 
                                 beta: float = 0.5,
                                 psf_sigma_I: float = 1.0, 
                                 psf_sigma_G: float = 0.7,
                                 psf_amp_I: float = 32000,
                                 psf_amp_G: float = 32000**2/1000000,
                                 bounds_slack: float = 10.0,
                                 reg_weight: float = 0.0,
                                 plot_results: bool = False,
                                 verbose: bool = False,
                                 placement_strategy: str = 'distributed') -> Dict:
    """
    Optimized greedy approach for emitter detection. Key improvements:
    - Batch evaluation of variations
    - Smarter termination criteria
    - Reduced redundant calculations
    - Warm starts from previous solutions
    """
    
    # Find ROIs (unchanged)
    if verbose:
        print("Finding ROIs and estimating number of emitters...")
    roi_stats, roi_labels = roi_n_emitters(
        I_meas, Gd_meas, n_emitters_map,
        intensity_threshold=intensity_threshold,
        plot_results=plot_results,
        verbose=verbose
    )
    
    roi_stats.sort(key=lambda x: x['intensity_mean'], reverse=True)
    
    # Initialize base configuration
    base_config = []
    total_emitters = 0
    
    for roi in roi_stats:
        estimated_emitters = int(round(roi['emitters_estimate']))
        if total_emitters + estimated_emitters <= max_emitters_total:
            base_config.append(estimated_emitters)
            total_emitters += estimated_emitters
        else:
            remaining = max_emitters_total - total_emitters
            base_config.append(max(0, remaining))
            total_emitters += max(0, remaining)
    
    # Pre-generate stable initial positions for each ROI to avoid randomness
    roi_position_templates = []
    for roi_idx, roi in enumerate(roi_stats):
        roi_label = roi['label']
        roi_mask = (roi_labels == roi_label)
        
        # Generate more positions than we'll likely need
        max_positions_needed = min(10, max_emitters_total)  # Cap reasonable limit
        positions = []
        
        if placement_strategy == 'distributed':
            roi_y_indices, roi_x_indices = np.where(roi_mask)
            roi_intensities = I_meas[roi_mask]
            probs = roi_intensities + 1e-10
            probs = probs / np.sum(probs)
            
            # Generate deterministic positions based on intensity ranking
            sorted_indices = np.argsort(-roi_intensities)
            for i in range(min(max_positions_needed, len(sorted_indices))):
                idx = sorted_indices[i % len(sorted_indices)]
                x, y = roi_x_indices[idx], roi_y_indices[idx]
                # Use deterministic small offsets instead of random
                offset_x = (i % 3 - 1) * 0.3  # -0.3, 0, 0.3, repeat
                offset_y = ((i // 3) % 3 - 1) * 0.3
                positions.append({
                    'x': x + offset_x,
                    'y': y + offset_y,
                    'I_amplitude': I_meas[y, x],
                    'Gd_amplitude': Gd_meas[y, x],
                    'roi_label': roi_label
                })
        
        elif placement_strategy == 'maxima':
            masked_intensity = np.zeros_like(I_meas)
            masked_intensity[roi_mask] = I_meas[roi_mask]
            
            x_maxima, y_maxima, heights = find_local_maxima(
                masked_intensity, 
                min_distance=min(min_distance, int(np.sqrt(roi['area'])/2)), 
                threshold_rel=0.3
            )
            
            if len(x_maxima) > 0:
                sorted_indices = np.argsort(-heights)
                for i in range(min(max_positions_needed, len(sorted_indices))):
                    idx = sorted_indices[i % len(sorted_indices)]
                    x, y = x_maxima[idx], y_maxima[idx]
                    offset_x = (i % 3 - 1) * 0.1
                    offset_y = ((i // 3) % 3 - 1) * 0.1
                    positions.append({
                        'x': x + offset_x,
                        'y': y + offset_y,
                        'I_amplitude': I_meas[y, x],
                        'Gd_amplitude': Gd_meas[y, x],
                        'roi_label': roi_label
                    })
            else:
                # Fallback to centroid
                y_center, x_center = roi['centroid']
                for i in range(max_positions_needed):
                    offset_x = (i % 3 - 1) * 0.3
                    offset_y = ((i // 3) % 3 - 1) * 0.3
                    positions.append({
                        'x': x_center + offset_x,
                        'y': y_center + offset_y,
                        'I_amplitude': roi['intensity_mean'],
                        'Gd_amplitude': roi['Gd_mean'],
                        'roi_label': roi_label
                    })
        
        else:  # centroid
            y_center, x_center = roi['centroid']
            for i in range(max_positions_needed):
                offset_x = (i % 3 - 1) * 0.3
                offset_y = ((i // 3) % 3 - 1) * 0.3
                positions.append({
                    'x': x_center + offset_x,
                    'y': y_center + offset_y,
                    'I_amplitude': roi['intensity_mean'],
                    'Gd_amplitude': roi['Gd_mean'],
                    'roi_label': roi_label
                })
        
        roi_position_templates.append(positions)
    
    def quick_rss_estimate(config, I_meas, Gd_meas, roi_position_templates, alpha, beta, 
                        psf_sigma_I, psf_sigma_G, psf_amp_I, psf_amp_G):
        """
        Ultra-fast RSS estimation without full optimization.
        Uses initial positions to calculate RSS without running expensive L-BFGS-B optimization.
        
        Parameters:
        -----------
        config : list
            Configuration like [2, 1, 0, 3] meaning number of emitters per ROI
        I_meas, Gd_meas : np.ndarray  
            Measured intensity and Gd maps
        roi_position_templates : list
            Pre-computed position templates for each ROI
        alpha, beta : float
            Weights for intensity and Gd RSS
        psf_sigma_I, psf_sigma_G : float
            PSF widths
        psf_amp_I, psf_amp_G : float
            PSF amplitudes
            
        Returns:
        --------
        float
            Estimated RSS value
        """
        initial_positions = []
        
        # Build positions from templates
        for roi_idx, num_emitters in enumerate(config):
            if num_emitters <= 0 or roi_idx >= len(roi_position_templates):
                continue
            roi_positions = roi_position_templates[roi_idx][:num_emitters]
            initial_positions.extend(roi_positions)
        
        if not initial_positions:
            return float('inf')
        
        # Convert positions to parameter array for model functions
        n_emitters = len(initial_positions)
        params = []
        for pos in initial_positions:
            params.extend([pos['x'], pos['y']])
        params = np.array(params)
        
        # Generate models with initial positions (no optimization)
        I_model = model_intensity_2(params, I_meas.shape, n_emitters, psf_sigma_I, psf_amp_I)
        Gd_model = model_Gd_2(params, I_meas.shape, n_emitters, psf_sigma_G, psf_amp_G)
        
        # Calculate RSS
        RSS_I = np.sum((I_meas - I_model)**2)
        RSS_Gd = np.sum((Gd_meas - Gd_model)**2)
        RSS = alpha * RSS_I + beta * RSS_Gd
        
        return RSS


    def evaluate_config_fast(config, I_meas, Gd_meas, roi_position_templates,
                            alpha, beta, psf_sigma_I, psf_sigma_G, psf_amp_I, psf_amp_G,
                            reg_weight, bounds_slack, do_full_optimization=True):
        """
        Fast configuration evaluation with optional full optimization.
        
        Parameters:
        -----------
        config : list
            Configuration like [2, 1, 0, 3] meaning number of emitters per ROI
        I_meas, Gd_meas : np.ndarray  
            Measured intensity and Gd maps
        roi_position_templates : list
            Pre-computed position templates for each ROI
        alpha, beta : float
            Weights for intensity and Gd RSS
        psf_sigma_I, psf_sigma_G : float
            PSF widths
        psf_amp_I, psf_amp_G : float
            PSF amplitudes
        reg_weight : float
            Regularization weight
        bounds_slack : float
            Bounds slack for optimization
        do_full_optimization : bool
            If True, runs full optimize_positions_2(). If False, just returns quick RSS estimate.
            
        Returns:
        --------
        dict
            Dictionary with 'RSS' and 'emitters' keys. If do_full_optimization=False, 
            'emitters' will be empty list.
        """
        if not do_full_optimization:
            rss_estimate = quick_rss_estimate(config, I_meas, Gd_meas, roi_position_templates,
                                            alpha, beta, psf_sigma_I, psf_sigma_G, 
                                            psf_amp_I, psf_amp_G)
            return {'RSS': rss_estimate, 'emitters': []}
        
        # Build initial positions from templates
        initial_positions = []
        
        for roi_idx, num_emitters in enumerate(config):
            if num_emitters <= 0 or roi_idx >= len(roi_position_templates):
                continue
            roi_positions = roi_position_templates[roi_idx][:num_emitters]
            initial_positions.extend(roi_positions)
        
        if not initial_positions:
            return {'RSS': float('inf'), 'emitters': []}
        
        # Full optimization using optimize_positions_2
        result = optimize_positions_2(
            I_meas, Gd_meas, initial_positions,
            alpha=alpha, beta=beta,
            psf_sigma_I=psf_sigma_I, psf_sigma_G=psf_sigma_G,
            psf_amp_I=psf_amp_I, psf_amp_G=psf_amp_G, 
            regularization_weight=reg_weight,
            bounds_slack=bounds_slack,
            verbose=False
        )
        
        return result
    
    # Evaluate base configuration with full optimization
    if verbose:
        print(f"Evaluating base configuration: {base_config}")
    current_result = evaluate_config_fast(base_config, I_meas, Gd_meas, roi_position_templates,
                                         alpha, beta, psf_sigma_I, psf_sigma_G, 
                                         psf_amp_I, psf_amp_G, reg_weight, bounds_slack,
                                         do_full_optimization=True)
    current_config = base_config[:]
    current_rss = current_result['RSS']
    if verbose:
        print(f"Base configuration RSS: {current_rss:.4f}")
    
    # Optimized iterative improvement
    iteration = 0
    max_iterations = min(10, len(roi_stats))  # Reduced further
    min_improvement_ratio = 0.005  # 0.5% minimum improvement
    stagnation_count = 0
    max_stagnation = 2  # Reduced for faster termination
    
    while iteration < max_iterations and stagnation_count < max_stagnation:
        iteration += 1
        iteration_improved = False
        
        if verbose:
            print(f"\nIteration {iteration}:")
        
        # Phase 1: Quick screening with RSS estimation
        candidate_variations = []
        
        # Generate variations (fewer of them)
        for roi_idx in range(min(len(current_config), len(roi_stats))):
            # Try removing one emitter
            if current_config[roi_idx] > 0:
                minus_config = current_config[:]
                minus_config[roi_idx] -= 1
                candidate_variations.append(minus_config)
            
            # Try adding one emitter
            if sum(current_config) < max_emitters_total:
                plus_config = current_config[:]
                plus_config[roi_idx] += 1
                candidate_variations.append(plus_config)
        
        if verbose:
            print(f"Quick screening {len(candidate_variations)} variations...")
        
        # Quick screening phase - estimate RSS without full optimization
        promising_variations = []
        min_improvement = current_rss * min_improvement_ratio
        
        for var_config in candidate_variations:
            quick_rss = quick_rss_estimate(var_config, I_meas, Gd_meas, roi_position_templates,
                                          alpha, beta, psf_sigma_I, psf_sigma_G, 
                                          psf_amp_I, psf_amp_G)
            if current_rss - quick_rss > min_improvement:  # Potential improvement
                promising_variations.append((var_config, quick_rss))
        
        # Sort by estimated improvement and take top candidates
        promising_variations.sort(key=lambda x: x[1])
        top_candidates = promising_variations[:min(3, len(promising_variations))]  # Only test top 3
        
        if verbose:
            print(f"Found {len(top_candidates)} promising candidates for full optimization")
        
        # Phase 2: Full optimization on promising candidates only
        best_iteration_rss = current_rss
        best_iteration_config = None
        best_iteration_result = None
        
        for var_config, estimated_rss in top_candidates:
            var_result = evaluate_config_fast(var_config, do_full_optimization=True)
            
            if var_result['RSS'] < best_iteration_rss:
                improvement = best_iteration_rss - var_result['RSS']
                if improvement > min_improvement:
                    best_iteration_rss = var_result['RSS']
                    best_iteration_config = var_config
                    best_iteration_result = var_result
                    iteration_improved = True
                    if verbose:
                        print(f"Found improvement: RSS {var_result['RSS']:.4f} (was {current_rss:.4f})")
        
        # Update current solution if we found improvement
        if iteration_improved:
            current_rss = best_iteration_rss
            current_config = best_iteration_config
            current_result = best_iteration_result
            stagnation_count = 0
            if verbose:
                print(f"Updated to config: {current_config}")
        else:
            stagnation_count += 1
            if verbose:
                print(f"No improvement found. Stagnation count: {stagnation_count}")
    
    if verbose:
        print(f"\nOptimization completed after {iteration} iterations")
        print(f"Final RSS: {current_rss:.4f}")
        print(f"Final config: {current_config}")
    
    # Add configuration to result
    current_result['emitter_config'] = current_config
    
    return current_result

from skimage.restoration import richardson_lucy
def deconvolve_emitter_locations(emitter_map, psf, num_iterations=50, plot_results=False, verbose=False):
    """
    Deconvolves an emitter map to estimate individual emitter locations using
    the Richardson-Lucy algorithm.

    Args:
        emitter_map (np.ndarray): The 2D array representing the estimated
                                  number of emitters at each scanning location
                                  (the convoluted image).
        psf (np.ndarray): The 2D array representing the Point Spread Function
                          of a single emitter. This should be normalized such
                          that its sum is 1.
        num_iterations (int, optional): The number of iterations for the
                                        Richardson-Lucy algorithm. More iterations
                                        can lead to sharper results but also
                                        amplify noise. Defaults to 50.
        plot_results (bool, optional): Whether to plot the original emitter map,
                                       the PSF, and the deconvolved result.
                                       Defaults to True.

    Returns:
        np.ndarray: The deconvolved image, representing the estimated
                    emitter locations.
    """

    if not isinstance(emitter_map, np.ndarray) or emitter_map.ndim != 2:
        raise ValueError("emitter_map must be a 2D NumPy array.")
    if not isinstance(psf, np.ndarray) or psf.ndim != 2:
        raise ValueError("psf must be a 2D NumPy array.")
    #if not np.isclose(np.sum(psf), 1.0):
    #    print("Warning: PSF does not sum to 1. Normalizing PSF.")
    #    psf = psf / np.sum(psf)
    if verbose:
        print(f"Starting Richardson-Lucy deconvolution with {num_iterations} iterations...")
    deconvolved_image = richardson_lucy(emitter_map, psf, num_iter=num_iterations, clip=False)
    if verbose:
        print("Deconvolution complete.")

    if plot_results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original Emitter Map
        ax0 = axes[0].imshow(emitter_map, cmap='viridis')
        axes[0].set_title('Original G2 Diff Map (Convoluted)')
        fig.colorbar(ax0, ax=axes[0])

        # PSF
        ax1 = axes[1].imshow(psf, cmap='gray')
        axes[1].set_title('Point Spread Function (PSF)')
        fig.colorbar(ax1, ax=axes[1])

        # Deconvolved Image
        ax2 = axes[2].imshow(deconvolved_image, cmap='hot') # 'hot' often good for point-like features
        axes[2].set_title(f'Deconvolved Emitter Locations ({num_iterations} iter)')
        fig.colorbar(ax2, ax=axes[2])

        plt.tight_layout()
        plt.show()

    return deconvolved_image

def gaussian_2d_for_deconvolution(shape,A,sigma) -> np.ndarray:
    """
    Generate a 2D Gaussian function
    
    Parameters:
    -----------
    x, y : np.ndarray
        Meshgrid arrays of x and y coordinates
    amplitude : float
        Peak height of the Gaussian
    x0, y0 : float
        Center coordinates of the Gaussian
    sigma_x, sigma_y : float
        Standard deviations in x and y directions
    offset : float
        Background offset
        
    Returns:
    --------
    np.ndarray
        2D Gaussian array
    """

    center_y, center_x = (shape[0]-1) / 2, (shape[1]-1) / 2
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    psf = A*np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
    return psf #/ np.sum(psf) # Normalize

def extract_psf(laser_power, pixel_size, dwell_time, dead_time, plot=False, psf_file=None, verbose=False):
    """
    Calculates the PSF for both intensity and G2 difference maps for a given laser power, dwell time, and dead time.
    Args:
        laser_power (float): The laser power in mW.
        pixel_size (float): The pixel size in micrometers.
        dwell_time (float): The dwell time in ms.
        dead_time (float): The dead time in ns.
        plot (bool): Whether to plot the PSF and fitted Gaussian.
        psf_file (str): Path to PSF data file for caching.
    Returns:
        (dict, dict): two dicts containing the PSF parameters (amplitude in counts, sigma in um).
    """
    # Create composite key from all three parameters
    psf_key = f"{laser_power}_{dwell_time}_{dead_time}"
    
    if psf_file:
        import json
        with open(f'{psf_file}', 'r') as f:
             psf_data = json.load(f)
        if psf_data.get(psf_key):
            I_psf = psf_data[psf_key]['I']
            G2diff_psf = psf_data[psf_key]['G']
            if verbose:
                print(f"Loaded PSF from file for key: {psf_key}")
            return I_psf, G2diff_psf
        else:
            print(f"No PSF data found in file for key: {psf_key}, calculating new PSF.")
    
    # Do a measurement with one emitter at the center
    setup, intensity_image, G2diff_image, _, metadata = run_scanning_experiment(
        positions = (20,20),
        laser_power = laser_power,
        dwell_time = dwell_time,
        dead_time = dead_time,
        #crosstalk = 0,
        #afterpulsing = 0, 
        #jitter = 0, 
        #dark_count_rate = 0,
        area_size = (pixel_size*20, pixel_size*20),
        emitters_manual = [(0,0)], 
        show_plots = False, 
        enable_noise = True,
        save_data = False)
   
    print(f"Intensity PSF fitting...")
    I_amplitude, I_sigma = get_psf_params_peak_preserving(intensity_image)
    print(f"G2 diff PSF fitting...")
    G_amplitude, G_sigma = get_psf_params_peak_preserving(G2diff_image)

    #I_amplitude, I_sigma = get_psf_params_radial(setup.scan_data['photon_count_map'])
    #G_amplitude, G_sigma = get_psf_params_radial(setup.scan_data['G2_diff_map'])



    I_sigma = I_sigma * pixel_size  # Convert pixel sigma to micrometers
    G_sigma = G_sigma * pixel_size  # Convert pixel sigma to micrometers
    I_psf = {'amplitude': I_amplitude, 'sigma': I_sigma}
    G2diff_psf = {'amplitude': G_amplitude, 'sigma': G_sigma}
    
    if psf_file:
        psf_data[psf_key] = {
            'I': I_psf,
            'G': G2diff_psf
        }
        print(f"Saving PSF data to file for key: {psf_key}, pixel_size: {pixel_size}, dwell_time: {dwell_time}, dead_time: {dead_time}, laser_power: {laser_power}")
        print(f"PSF parameters: {I_psf}, {G2diff_psf}")
        with open(f'{psf_file}', 'w') as f:
            json.dump(psf_data, f, indent=4)
    
    if plot:
        print(f"Summary Statistics:")
        print(f"Intensity PSF: Amplitude = {I_amplitude}, Sigma = {I_sigma} um")
        print(f"G2 Difference PSF: Amplitude = {G_amplitude}, Sigma = {G_sigma} um")
        # plot a line through middle of the PSF
        plt.figure(figsize=(10, 6))
        mid_y = setup.scan_data['photon_count_map'].shape[0] // 2
        plt.plot(setup.scan_data['photon_count_map'][mid_y, :], label='Horizontal line through PSF')
        # plot the fitted Gaussian
        x = np.arange(setup.scan_data['photon_count_map'].shape[1])
        fitted_gaussian = gaussian_2d((x, np.full_like(x, mid_y)), I_amplitude, x0=x.mean(), y0=mid_y, sigma_x=I_sigma/pixel_size, sigma_y=I_sigma/pixel_size, offset=0, flatten=False)
        plt.plot(x, fitted_gaussian, label='Fitted Gaussian', color='red')
        plt.legend()
        # now for G2 diff
        plt.figure(figsize=(10, 6))
        plt.plot(setup.scan_data['G2_diff_map'][mid_y, :], label='Horizontal line through G2 diff PSF')
        print(setup.scan_data['G2_diff_map'][mid_y, :])
        # plot the fitted Gaussian
        fitted_gaussian_G2 = gaussian_2d((x, np.full_like(x, mid_y)), G_amplitude, x0=x.mean(), y0=mid_y, sigma_x=G_sigma/pixel_size, sigma_y=G_sigma/pixel_size, offset=0, flatten=False)
        plt.plot(x, fitted_gaussian_G2, label='Fitted Gaussian G2 diff', color='red')
        plt.legend()
        plt.show()
    
    return I_psf, G2diff_psf


def extract_psf_improved(laser_power, pixel_size, dwell_time, dead_time, 
                        n_simulations=10, plot=False, psf_file=None, 
                        random_seed=None):
    """
    Calculates the PSF for both intensity and G2 difference maps with improved accuracy
    by averaging multiple simulations before fitting.
    
    Args:
        laser_power (float): The laser power in mW.
        pixel_size (float): The pixel size in micrometers.
        dwell_time (float): The dwell time in ms.
        dead_time (float): The dead time in ns.
        n_simulations (int): Number of simulations to average (default: 10).
        plot (bool): Whether to plot the PSF and fitted Gaussian.
        psf_file (str): Path to PSF data file for caching.
        random_seed (int): Random seed for reproducibility.
    
    Returns:
        (dict, dict): two dicts containing the PSF parameters (amplitude in counts, sigma in um).
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Create composite key from all parameters including n_simulations
    psf_key = f"{laser_power}_{dwell_time}_{dead_time}_{n_simulations}"
    
    # Try to load from cache first
    if psf_file:
        try:
            with open(psf_file, 'r') as f:
                psf_data = json.load(f)
            if psf_data.get(psf_key):
                I_psf = psf_data[psf_key]['I']
                G2diff_psf = psf_data[psf_key]['G']
                print(f"Loaded PSF from file for key: {psf_key}")
                return I_psf, G2diff_psf
            else:
                print(f"No PSF data found in file for key: {psf_key}, calculating new PSF.")
        except (FileNotFoundError, json.JSONDecodeError):
            print("PSF file not found or corrupted, calculating new PSF.")
            psf_data = {}
    else:
        psf_data = {}
    
    print(f"Running {n_simulations} simulations to extract PSF...")
    
    # Initialize arrays to store all simulation results
    intensity_maps = []
    g2_diff_maps = []
    
    # Run multiple simulations
    for i in range(n_simulations):
        if i % max(1, n_simulations // 5) == 0:
            print(f"Running simulation {i+1}/{n_simulations}")
        
        # Run single simulation
        setup, scan_data, _, _, metadata = run_scanning_experiment(
            positions=(20, 20),
            laser_power=laser_power,
            dwell_time=dwell_time,
            dead_time=dead_time,
            crosstalk=0,
            afterpulsing=0,
            jitter=0,
            dark_count_rate=0,
            area_size=(pixel_size*20, pixel_size*20),
            emitters_manual=[(0, 0)],
            show_plots=False,
            enable_noise=True,
            save_data=False
        )
        
        intensity_maps.append(setup.scan_data['photon_count_map'])
        g2_diff_maps.append(setup.scan_data['G2_diff_map'])
    
    # Average all simulations
    print("Averaging simulations...")
    avg_intensity_map = np.mean(intensity_maps, axis=0)
    avg_g2_diff_map = np.mean(g2_diff_maps, axis=0)
    
    # Calculate standard deviation for quality assessment
    std_intensity_map = np.std(intensity_maps, axis=0)
    std_g2_diff_map = np.std(g2_diff_maps, axis=0)
    
    # Fit using your existing function
    print("Fitting intensity PSF using get_psf_params_peak_preserving...")
    I_amplitude, I_sigma_pixels = get_psf_params_peak_preserving(avg_intensity_map)
    
    print("Fitting G2 difference PSF using get_psf_params_peak_preserving...")
    G_amplitude, G_sigma_pixels = get_psf_params_peak_preserving(avg_g2_diff_map)
    
    # Convert pixel sigma to micrometers
    I_sigma = I_sigma_pixels * pixel_size
    G_sigma = G_sigma_pixels * pixel_size
    
    # Create result dictionaries
    I_psf = {
        'amplitude': I_amplitude,
        'sigma': I_sigma,
        'avg_map': avg_intensity_map,
        'std_map': std_intensity_map
    }
    
    G2diff_psf = {
        'amplitude': G_amplitude,
        'sigma': G_sigma,
        'avg_map': avg_g2_diff_map,
        'std_map': std_g2_diff_map
    }
    
    # Save to cache
    if psf_file:
        # Remove large arrays from cached data
        I_psf_cache = {k: v for k, v in I_psf.items() if k not in ['avg_map', 'std_map']}
        G2diff_psf_cache = {k: v for k, v in G2diff_psf.items() if k not in ['avg_map', 'std_map']}
        
        psf_data[psf_key] = {
            'I': I_psf_cache,
            'G': G2diff_psf_cache
        }
        
        print(f"Saving PSF data to file for key: {psf_key}")
        print(f"PSF parameters: I_sigma={I_sigma:.3f} um, G_sigma={G_sigma:.3f} um")
        
        with open(psf_file, 'w') as f:
            json.dump(psf_data, f, indent=4)
    
    # Plotting
    if plot:
        print(f"\nSummary Statistics:")
        print(f"Intensity PSF: Amplitude = {I_amplitude:.1f}, Sigma = {I_sigma:.3f} um")
        print(f"G2 Difference PSF: Amplitude = {G_amplitude:.1f}, Sigma = {G_sigma:.3f} um")
        
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Intensity PSF plots
        # 2D map
        im1 = axes[0, 0].imshow(avg_intensity_map, cmap='hot', origin='lower')
        axes[0, 0].set_title('Averaged Intensity PSF')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Cross-section with simple visualization
        mid_y = avg_intensity_map.shape[0] // 2
        x_coords = np.arange(avg_intensity_map.shape[1])
        axes[0, 1].plot(x_coords, avg_intensity_map[mid_y, :], 'b-', label='Averaged Data')
        axes[0, 1].errorbar(x_coords, avg_intensity_map[mid_y, :], 
                           yerr=std_intensity_map[mid_y, :], alpha=0.3, color='blue')
        axes[0, 1].set_title('Intensity PSF Cross-section')
        axes[0, 1].legend()
        
        # Show sigma info
        axes[0, 2].text(0.1, 0.5, f'Amplitude: {I_amplitude:.1f}\nSigma: {I_sigma:.3f} μm', 
                       transform=axes[0, 2].transAxes, fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[0, 2].set_title('Intensity PSF Parameters')
        axes[0, 2].axis('off')
        
        # G2 difference PSF plots
        # 2D map
        im3 = axes[1, 0].imshow(avg_g2_diff_map, cmap='hot', origin='lower')
        axes[1, 0].set_title('Averaged G2 Difference PSF')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Cross-section with simple visualization
        axes[1, 1].plot(x_coords, avg_g2_diff_map[mid_y, :], 'b-', label='Averaged Data')
        axes[1, 1].errorbar(x_coords, avg_g2_diff_map[mid_y, :], 
                           yerr=std_g2_diff_map[mid_y, :], alpha=0.3, color='blue')
        axes[1, 1].set_title('G2 Difference PSF Cross-section')
        axes[1, 1].legend()
        
        # Show sigma info
        axes[1, 2].text(0.1, 0.5, f'Amplitude: {G_amplitude:.1f}\nSigma: {G_sigma:.3f} μm', 
                       transform=axes[1, 2].transAxes, fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        axes[1, 2].set_title('G2 Difference PSF Parameters')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return I_psf, G2diff_psf



def localize(I_meas, Gd_meas, est_emitters, metadata, plot=False, psf_file=None, reg_weight=0, verbose=False):

    if not metadata["enable_noise"]:
        print("Warning: Metadata indicates noise is disabled. This changes the PSF specifications to noiseless values.")
        metadata["dead_time"] = 0
        metadata["afterpulsing"] = 0
        metadata["jitter"] = 0
        metadata["dark_count_rate"] = 0
        metadata["crosstalk"] = 0

    psf_I, psf_G2diff = extract_psf(laser_power = metadata['laser_power'] , pixel_size = metadata['pixel_size'], dwell_time = metadata['dwell_time'], dead_time = metadata['dead_time'], psf_file=psf_file, verbose=verbose)
    psf_amp_I = psf_I['amplitude']
    psf_sigma_I_um = psf_I['sigma']
    psf_amp_G = psf_G2diff['amplitude']
    psf_sigma_G_um = psf_G2diff['sigma']

    print(f"PSF G used: Amplitude: {psf_amp_G}, Sigma: {psf_sigma_G_um}")
    if verbose:
        print(f"PSF parameters: I - amplitude: {psf_amp_I}, sigma: {psf_sigma_I_um} um; G_d - amplitude: {psf_amp_G}, sigma: {psf_sigma_G_um} um")
        print(f"Pixel size: {metadata['pixel_size']} um")

    deconv_psf = gaussian_2d_for_deconvolution((15,15),1, 3)
    Gd_meas_deconvolved = deconvolve_emitter_locations(Gd_meas, psf=deconv_psf, num_iterations=70, plot_results=plot, verbose=verbose)
    # initial_positions = find_initial_positions_roi(Gd_meas_deconvolved, Gd_meas, est_emitters,
    #                                           min_distance=1, 
    #                                           threshold_rel=0.05, 
    #                                           threshold_nemitters=0.1,
    #                                           roi_min_size=1,
    #                                           roi_max_size=10000,
    #                                           intensity_threshold=None,
    #                                           max_emitters=100)
    
    # Optimize positions
    # result = optimize_positions_2(I_meas, Gd_meas, initial_positions,
    #                            alpha=0.5, beta=0.5,  # Weights for I and G_d
    #                            psf_sigma_I=psf_sigma_I, 
    #                            psf_sigma_G=psf_sigma_G,
    #                            psf_amp_I=32000, psf_amp_G=32000**2/1000000,
    #                            verbose=True)
    max_emitters_total = int(np.sum(est_emitters) * 1.2) # 20% more than estimated emitters

    if verbose:
        print(f"Starting greedy optimization with max emitters total: {max_emitters_total}")


    def add_random_noise(array, scale=1.0):
        import random
        """
        Adds random noise to each element of a 2D list (or array-like).
        
        Parameters:
            array (list of lists): 2D list of numbers.
            scale (float): maximum magnitude of noise. 
                        Noise is sampled uniformly from [-scale, scale].
        
        Returns:
            list of lists: new 2D list with noise added.
        """
        noisy_array = []
        for row in array:
            noisy_row = [val + random.uniform(0, scale) for val in row]
            noisy_array.append(noisy_row)
        return np.array(noisy_array)
    # Add random noise to I_meas and Gd_meas
    #I_meas = add_random_noise(I_meas, scale=2000)
    #Gd_meas = add_random_noise(Gd_meas, scale=20)


    greedy_result = optimize_with_greedy_approach(
        I_meas, Gd_meas, est_emitters,
        min_distance=1,
        threshold_rel=0.05,
        threshold_nemitters=0.1,
        intensity_threshold=None,
        max_emitters_total=max_emitters_total*10, #15, #this must be equal to the sum of est_emitters + 20%
        alpha=0.5, beta=0.5,
        psf_sigma_I=psf_sigma_I_um/metadata['pixel_size'],
        psf_sigma_G=psf_sigma_G_um/metadata['pixel_size'],
        psf_amp_I=psf_amp_I,
        psf_amp_G=psf_amp_G,
        reg_weight=reg_weight,
        plot_results = plot,
        verbose = verbose,
        placement_strategy='distributed'
    )


    return greedy_result


def test_usage():
    psf_sigma_I = 0.16 # in um
    psf_sigma_G = 0.1 # in um; tighter PSF for G_d

    #Load I and Gd maps from file
    import pickle
    import datetime
    date = datetime.date.today().strftime("%Y-%m-%d")
    area_size = '(1, 1)'
    pixel_size = 0.1 # in micrometers

    datetimestamp = datetime.datetime.now().strftime("202506031436")
    date = '2025-06-03'
    pixel_size = 0.05
    area_size = '1x1'
    #area_size = '(2, 2)'
    # PSF size in pixels
    psf_sigma_I_pix = psf_sigma_I / pixel_size
    psf_sigma_G_pix = psf_sigma_G / pixel_size

    with open(f'./project/data/{date}/photon_count_map_{datetimestamp}.pkl', 'rb') as f:
        I_meas = pickle.load(f)
    with open(f'./project/data/{date}/G2_difference_map_{datetimestamp}.pkl', 'rb') as f:
        Gd_meas = pickle.load(f)
    with open(f'./project/data/{date}/nr_emitter_map_{datetimestamp}.pkl', 'rb') as f:
        est_emitters = pickle.load(f)
    # open metdata
    import json
    with open(f'./project/data/{date}/metadata_{datetimestamp}.json', 'r') as f:
        metadata = json.load(f)


    # setup, I_meas, Gd_meas, est_emitters, metadata = run_scanning_experiment(
    #     positions = (20,20),
    #     laser_power= 100E3,
    #     area_size = (1,1),
    #     emitters_manual=None, show_plots=False, save_data=True)


    psf_I, psf_G2diff = extract_psf(laser_power = metadata['laser_power'], pixel_size = metadata['pixel_size'], dwell_time = metadata['dwell_time'], dead_time = metadata['dead_time'], psf_file='./project/data/psf.json')

    psf_amp_I = psf_I['amplitude']
    psf_sigma_I = psf_I['sigma']
    psf_amp_G = psf_G2diff['amplitude']
    psf_sigma_G = psf_G2diff['sigma']
    psf_sigma_I_pix = psf_sigma_I / pixel_size
    psf_sigma_G_pix = psf_sigma_G / pixel_size
    
    # Find initial positions
    # initial_positions = find_initial_positions(I_meas, Gd_meas, #TODO this function must be changed: must take the estimated n as input. 
    #                                           min_distance=5, 
    #                                           threshold_rel=0.1,
    #                                           max_emitters=10)
    # initial_positions = find_initial_positions_2(I_meas, Gd_meas, est_emitters, 
    #                                           min_distance=3, 
    #                                           threshold_rel=0.05,
    #                                           threshold_nemitters=0.1,
    #                                           max_emitters=10)
 
    deconv_psf = gaussian_2d_for_deconvolution((15,15),1, psf_sigma_G_pix)
    Gd_meas_deconvolved = deconvolve_emitter_locations(Gd_meas, psf=deconv_psf, num_iterations=70, plot_results=True)
    # plot gdmeasdeconv
    #plt.imshow(Gd_meas_deconvolved, origin='lower')

    initial_positions = find_initial_positions_roi(Gd_meas_deconvolved, Gd_meas, est_emitters,
                                              min_distance=1, 
                                              threshold_rel=0.05,
                                              threshold_nemitters=0.1,
                                              roi_min_size=1,
                                              roi_max_size=10000,
                                              intensity_threshold=None,
                                              max_emitters=100)
    
    print(f"Found {len(initial_positions)} initial positions:")
    for i, pos in enumerate(initial_positions):
        print(f"Initial {i+1}: x={pos['x']:.2f}, y={pos['y']:.2f}")

    #Test
    #initial_positions.append({'x': 5, 'y': 6, 'I_amplitude': 10000, 'Gd_amplitude': 1000})
    
    # Optimize positions
    result = optimize_positions_2(I_meas, Gd_meas, initial_positions,
                               alpha=0.5, beta=0.5,  # Weights for I and G_d
                               psf_sigma_I=psf_sigma_I_pix, 
                               psf_sigma_G=psf_sigma_G_pix,
                               psf_amp_I=psf_amp_I, psf_amp_G=psf_amp_G,
                               regularization_weight=1000)
    

    #
    print("Optimized emitter locations:")
    print(result['emitters'])
    # Visualize results

    visualize_results(I_meas, Gd_meas, result, initial_positions, area_size, pixel_size)

    print("Now the greedy approach")
        # Run greedy optimization (faster)
    print("\nRunning greedy optimization approach...")
    greedy_result = optimize_with_greedy_approach(
        I_meas, Gd_meas, est_emitters,
        min_distance=1,
        threshold_rel=0.05,
        threshold_nemitters=0.1,
        intensity_threshold=None,
        max_emitters_total=15,
        alpha=0, beta=1,
        psf_sigma_I=psf_sigma_I_pix,
        psf_sigma_G=psf_sigma_G_pix,
        psf_amp_I=psf_amp_I,
        psf_amp_G=psf_amp_G,
        placement_strategy='distributed',
        reg_weight=1000,  # Regularization weight
        plot_results = True,
    )
    
    # Visualize results
    print("\nBest configuration found:", greedy_result['emitter_config'])
    print(f"Best RSS: {greedy_result['RSS']:.4f}")

    visualize_results(I_meas, Gd_meas, greedy_result, None, area_size, pixel_size, true_positions=metadata['emitter_positions'])
    
    return result

if __name__ == "__main__":
    #result = example_usage()
    result = test_usage()

# def demonstrate_roi_analysis():
#     """
#     Demonstrate the ROI analysis function with example data.
#     """
#     # Create synthetic test data (just for demonstration)
#     image_size = 100
    
#     # Generate sample intensity image with 3 bright spots
#     x, y = np.meshgrid(np.linspace(-1, 1, image_size), np.linspace(-1, 1, image_size))
#     centers = [(0.3, 0.2), (-0.2, -0.3), (0.7, -0.2)]
#     I_meas = np.zeros((image_size, image_size))
    
#     for cx, cy in centers:
#         I_meas += 1000 * np.exp(-((x-cx)**2 + (y-cy)**2) / 0.01)
        
#     # Add some noise
#     I_meas += np.random.normal(10, 5, I_meas.shape)
#     I_meas = np.clip(I_meas, 0, None)
    
#     # Create a synthetic G_d map that correlates with intensity
#     Gd_meas = I_meas * 0.8 + np.random.normal(0, 50, I_meas.shape)
    
#     # Create a synthetic number of emitters map 
#     n_emitters_map = np.zeros_like(I_meas)
#     for i, (cx, cy) in enumerate(centers):
#         n_value = (i+1) * 5  # Different number of emitters per ROI
#         n_emitters_map += n_value * np.exp(-((x-cx)**2 + (y-cy)**2) / 0.01)
    
#     # Run the ROI analysis
#     roi_stats, labels = roi_n_emitters(I_meas, Gd_meas, n_emitters_map, 
#                                       intensity_threshold=100, 
#                                       min_size=3, 
#                                       max_size=500)
    
#     return roi_stats, labels, I_meas, Gd_meas, n_emitters_map

# if __name__ == "__main__":
#     demonstrate_roi_analysis()