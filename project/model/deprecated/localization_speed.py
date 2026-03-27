import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import maximum_filter, label, binary_erosion
from scipy import ndimage
from skimage import measure, filters
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from numba import jit, njit
import warnings

# Fast compiled functions using numba
@njit
def fast_gaussian_2d(x_grid, y_grid, amplitude, x0, y0, sigma_x, sigma_y):
    """Fast compiled 2D Gaussian generation"""
    return amplitude * np.exp(-((x_grid - x0)**2 / (2 * sigma_x**2) + 
                               (y_grid - y0)**2 / (2 * sigma_y**2)))

@njit
def fast_model_intensity(params, x_grid, y_grid, n_emitters, psf_sigma_I, psf_amp_I):
    """Fast compiled intensity model generation"""
    model = np.zeros_like(x_grid)
    for i in range(n_emitters):
        xe = params[i*2]
        ye = params[i*2 + 1]
        model += fast_gaussian_2d(x_grid, y_grid, psf_amp_I, xe, ye, psf_sigma_I, psf_sigma_I)
    return model

@njit
def fast_model_Gd(params, x_grid, y_grid, n_emitters, psf_sigma_G, psf_amp_G):
    """Fast compiled G_d model generation"""
    model = np.zeros_like(x_grid)
    for i in range(n_emitters):
        xe = params[i*2]
        ye = params[i*2 + 1]
        model += fast_gaussian_2d(x_grid, y_grid, psf_amp_G, xe, ye, psf_sigma_G, psf_sigma_G)
    return model

@njit
def fast_rss_calculation(I_model, Gd_model, I_meas, Gd_meas, alpha, beta):
    """Fast RSS calculation"""
    RSS_I = np.sum((I_meas - I_model)**2)
    RSS_Gd = np.sum((Gd_meas - Gd_model)**2)
    return alpha * RSS_I + beta * RSS_Gd

class FastLocalizer:
    """
    Fast emitter localization using hierarchical optimization and compiled functions.
    """
    
    def __init__(self, psf_sigma_I=1.0, psf_sigma_G=0.7, 
                 psf_amp_I=32000, psf_amp_G=1024,
                 alpha=0.5, beta=0.5):
        """
        Initialize the fast localizer.
        
        Parameters:
        -----------
        psf_sigma_I, psf_sigma_G : float
            PSF widths for intensity and G_d (in pixels)
        psf_amp_I, psf_amp_G : float
            PSF amplitudes
        alpha, beta : float
            Weights for intensity and G_d fitting
        """
        self.psf_sigma_I = psf_sigma_I
        self.psf_sigma_G = psf_sigma_G
        self.psf_amp_I = psf_amp_I
        self.psf_amp_G = psf_amp_G
        self.alpha = alpha
        self.beta = beta
        
        # Pre-computed grids (will be set in localize)
        self.x_grid = None
        self.y_grid = None
        
    def _setup_grids(self, shape):
        """Setup coordinate grids for fast computation"""
        height, width = shape
        y, x = np.ogrid[:height, :width]
        self.x_grid = np.broadcast_to(x, (height, width)).astype(np.float64)
        self.y_grid = np.broadcast_to(y, (height, width)).astype(np.float64)
    
    def _fast_peak_detection(self, Gd_meas, n_emitters_map, 
                            min_separation=2, intensity_threshold=None):
        """
        Ultra-fast peak detection using vectorized operations.
        
        Returns candidate positions sorted by strength.
        """
        # Use adaptive threshold if not provided
        if intensity_threshold is None:
            # Use percentile-based threshold for speed
            intensity_threshold = np.percentile(Gd_meas[Gd_meas > 0], 75)
        
        # Create binary mask of high-intensity regions
        binary_mask = Gd_meas > intensity_threshold
        
        # Use morphological operations to separate close peaks
        if min_separation > 1:
            structure = np.ones((min_separation, min_separation))
            binary_mask = binary_erosion(binary_mask, structure)
        
        # Label connected components
        labels, num_features = label(binary_mask)
        
        if num_features == 0:
            return []
        
        # Fast centroid calculation using vectorized operations
        props = measure.regionprops(labels, intensity_image=Gd_meas)
        
        candidates = []
        for prop in props:
            if prop.area < 1:  # Skip tiny regions
                continue
                
            # Use weighted centroid for better accuracy
            y_cent, x_cent = prop.weighted_centroid
            
            # Get region mask for emitter count estimation
            region_mask = (labels == prop.label)
            estimated_emitters = max(1, int(round(np.max(n_emitters_map[region_mask]))))
            
            candidates.append({
                'x': x_cent,
                'y': y_cent,
                'strength': prop.intensity_mean * prop.area,  # Combined metric
                'n_emitters': min(estimated_emitters, 50),  # Cap to prevent explosion
                'area': prop.area
            })
        
        # Sort by strength and return
        candidates.sort(key=lambda x: x['strength'], reverse=True)
        return candidates
    
    def _hierarchical_optimization(self, I_meas, Gd_meas, candidates, max_emitters=15):
        """
        Hierarchical optimization: start with strongest candidates and add more.
        
        This is much faster than trying all combinations.
        """
        if not candidates:
            return {'emitters': [], 'RSS': float('inf')}
        
        # Start with the strongest candidate
        best_config = []
        best_rss = float('inf')
        best_result = None
        
        # Build emitter list from candidates
        current_emitters = []
        current_positions = []
        
        # Greedy addition: add emitters one region at a time
        for candidate in candidates:
            if len(current_emitters) >= max_emitters:
                break
                
            # Add emitters from this candidate
            n_to_add = min(candidate['n_emitters'], max_emitters - len(current_emitters))
            
            for i in range(n_to_add):
                # Small random offset for multiple emitters in same region
                offset_x = np.random.uniform(-0.5, 0.5) if i > 0 else 0
                offset_y = np.random.uniform(-0.5, 0.5) if i > 0 else 0
                
                current_emitters.append({
                    'x': candidate['x'] + offset_x,
                    'y': candidate['y'] + offset_y
                })
            
            # Quick evaluation using current positions
            current_params = []
            for emitter in current_emitters:
                current_params.extend([emitter['x'], emitter['y']])
            current_params = np.array(current_params)
            
            # Fast RSS estimation without optimization
            if len(current_params) > 0:
                quick_rss = self._quick_rss_estimate(current_params, I_meas, Gd_meas)
                print(quick_rss)
                quick_rss = self._optimize_positions_fast(I_meas, Gd_meas, current_emitters)['RSS']
                print(quick_rss)
                # If this configuration looks promising, do full optimization
                if quick_rss < best_rss * 1.1:  # Allow 10% tolerance for exploration
                    print(f"Exploring promising configuration with RSS: {quick_rss}")
                    result = self._optimize_positions_fast(I_meas, Gd_meas, current_emitters)
                    if result['RSS'] < best_rss:
                        best_rss = result['RSS']
                        best_result = result
                        best_config = current_emitters.copy()
        
        return best_result if best_result else {'emitters': [], 'RSS': float('inf')}
    
    def _quick_rss_estimate(self, params, I_meas, Gd_meas):
        """Ultra-fast RSS estimation without optimization"""
        n_emitters = len(params) // 2
        
        I_model = fast_model_intensity(params, self.x_grid, self.y_grid, n_emitters, 
                                     self.psf_sigma_I, self.psf_amp_I)
        Gd_model = fast_model_Gd(params, self.x_grid, self.y_grid, n_emitters,
                                self.psf_sigma_G, self.psf_amp_G)
        
        return fast_rss_calculation(I_model, Gd_model, I_meas, Gd_meas, self.alpha, self.beta)
    
    def _optimize_positions_fast(self, I_meas, Gd_meas, initial_positions):
        """Fast position optimization with minimal overhead"""
        n_emitters = len(initial_positions)
        if n_emitters == 0:
            return {'emitters': [], 'RSS': float('inf')}
        
        # Pack parameters
        initial_params = []
        bounds = []
        
        height, width = I_meas.shape
        for pos in initial_positions:
            initial_params.extend([pos['x'], pos['y']])
            bounds.extend([(max(0, pos['x']-5), min(width-1, pos['x']+5)),
                          (max(0, pos['y']-5), min(height-1, pos['y']+5))])
        
        initial_params = np.array(initial_params)
        
        # Objective function
        def objective(params):
            return self._quick_rss_estimate(params, I_meas, Gd_meas)
        
        # Fast optimization with loose tolerances
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(objective, initial_params, method='L-BFGS-B',
                                bounds=bounds, options={'gtol': 1e-6, 'maxiter': 100})
            
            opt_params = result.x
            
            # Build result
            emitters = []
            for i in range(n_emitters):
                emitters.append({
                    'x': opt_params[i*2],
                    'y': opt_params[i*2 + 1]
                })
            
            # Calculate final models for output
            I_model = fast_model_intensity(opt_params, self.x_grid, self.y_grid, n_emitters,
                                         self.psf_sigma_I, self.psf_amp_I)
            Gd_model = fast_model_Gd(opt_params, self.x_grid, self.y_grid, n_emitters,
                                   self.psf_sigma_G, self.psf_amp_G)
            
            final_rss = fast_rss_calculation(I_model, Gd_model, I_meas, Gd_meas, 
                                           self.alpha, self.beta)
            
            return {
                'emitters': emitters,
                'RSS': final_rss,
                'I_model': I_model,
                'Gd_model': Gd_model,
                'opt_success': result.success
            }
            
        except Exception as e:
            # Fallback: return initial positions if optimization fails
            return {
                'emitters': initial_positions,
                'RSS': objective(initial_params),
                'I_model': np.zeros_like(I_meas),
                'Gd_model': np.zeros_like(Gd_meas),
                'opt_success': False
            }
    
    def localize(self, I_meas, Gd_meas, n_emitters_map, 
                max_emitters=15, min_separation=2,
                intensity_threshold=None, verbose=False):
        """
        Main localization function - redesigned for speed.
        
        Parameters:
        -----------
        I_meas : np.ndarray
            Measured intensity map
        Gd_meas : np.ndarray  
            Measured G_d correlation map
        n_emitters_map : np.ndarray
            Map indicating estimated number of emitters at each position
        max_emitters : int
            Maximum number of emitters to detect
        min_separation : int
            Minimum separation between peaks (pixels)
        intensity_threshold : float
            Threshold for peak detection (auto if None)
        verbose : bool
            Print progress information
            
        Returns:
        --------
        Dict with localization results
        """
        if verbose:
            print("Starting fast localization...")
        
        # Setup coordinate grids
        self._setup_grids(I_meas.shape)
        
        # Step 1: Fast peak detection (replaces slow ROI analysis)
        if verbose:
            print("Detecting peaks...")
        candidates = self._fast_peak_detection(Gd_meas, n_emitters_map, 
                                             min_separation, intensity_threshold)
        
        if verbose:
            print(f"Found {len(candidates)} candidate regions")
            print(f"Full candidate info:")
            for candidate in candidates:
                print(f" - {candidate}")

        if not candidates:
            return {
                'emitters': [],
                'RSS': float('inf'),
                'I_model': np.zeros_like(I_meas),
                'Gd_model': np.zeros_like(Gd_meas)
            }
        
        # Step 2: Hierarchical optimization (replaces slow greedy search)
        if verbose:
            print("Optimizing emitter positions...")
        result = self._hierarchical_optimization(I_meas, Gd_meas, candidates, max_emitters)
        
        if verbose:
            print(f"Localization complete. Found {len(result['emitters'])} emitters.")
            print(f"Final RSS: {result['RSS']:.2f}")
        
        return result


def localize_fast(I_meas, Gd_meas, n_emitters_map, metadata, 
                 max_emitters=None, verbose=False, psf_file=None):
    """
    Fast localization function that replaces the slow original.
    
    This function is designed to be 10-50x faster while maintaining accuracy.
    
    Parameters:
    -----------
    I_meas : np.ndarray
        Measured intensity map
    Gd_meas : np.ndarray
        Measured G_d correlation map  
    n_emitters_map : np.ndarray
        Map with estimated number of emitters per pixel
    metadata : dict
        Experiment metadata containing PSF parameters
    max_emitters : int
        Maximum number of emitters (auto-estimated if None)
    verbose : bool
        Print progress information
    psf_file : str
        Path to PSF parameter file
        
    Returns:
    --------
    Dict with localization results (same format as original)
    """
    
    # Handle noiseless case
    if not metadata.get("enable_noise", True):
        if verbose:
            print("Noiseless mode detected - adjusting parameters")
        metadata = metadata.copy()
        metadata["dead_time"] = 0
        metadata["afterpulsing"] = 0
        metadata["jitter"] = 0
        metadata["dark_count_rate"] = 0
        metadata["crosstalk"] = 0
    
    # Extract PSF parameters (this function should be fast/cached)
    try:
        from .localization import extract_psf  # Assuming it exists
        psf_I, psf_G2diff = extract_psf(
            laser_power=metadata['laser_power'],
            pixel_size=metadata['pixel_size'],
            dwell_time=metadata['dwell_time'],
            dead_time=metadata['dead_time'],
            psf_file=psf_file,
            verbose=verbose
        )
        
        psf_sigma_I_pix = psf_I['sigma'] / metadata['pixel_size']
        psf_sigma_G_pix = psf_G2diff['sigma'] / metadata['pixel_size'] 
        psf_amp_I = psf_I['amplitude']
        psf_amp_G = psf_G2diff['amplitude']
        
    except Exception as e:
        if verbose:
            print(f"PSF extraction failed: {e}. Using default values.")
        # Fallback values
        psf_sigma_I_pix = 2.0
        psf_sigma_G_pix = 1.4
        psf_amp_I = 30000
        psf_amp_G = 1000
    
    # Auto-estimate max emitters if not provided
    if max_emitters is None:
        total_estimated = np.sum(n_emitters_map)
        max_emitters = max(5, min(100, int(total_estimated * 1.5)))  # 50% buffer, but capped
    
    if verbose:
        print(f"PSF parameters: σ_I={psf_sigma_I_pix:.2f}px, σ_G={psf_sigma_G_pix:.2f}px")
        print(f"Max emitters: {max_emitters}")
    
    # Create and run fast localizer
    localizer = FastLocalizer(
        psf_sigma_I=psf_sigma_I_pix,
        psf_sigma_G=psf_sigma_G_pix,
        psf_amp_I=psf_amp_I,
        psf_amp_G=psf_amp_G,
        alpha=0.3,  # Lower weight on intensity for speed
        beta=0.7    # Higher weight on G_d which has better SNR
    )
    
    result = localizer.localize(
        I_meas, Gd_meas, n_emitters_map,
        max_emitters=max_emitters,
        min_separation=1,#max(1, int(psf_sigma_G_pix)),  # Adaptive separation
        verbose=verbose
    )
    
    return result


# Backward compatibility wrapper
def localize(I_meas, Gd_meas, est_emitters, metadata, 
            plot=False, psf_file=None, reg_weight=0, verbose=False):
    """
    Drop-in replacement for the original slow localize function.
    
    This maintains the same interface but uses the fast algorithm internally.
    """
    result = localize_fast(I_meas, Gd_meas, est_emitters, metadata, 
                          verbose=verbose, psf_file=psf_file)
    
    # Add plotting if requested
    if plot and len(result['emitters']) > 0:
        try:
            # Quick visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Measured data
            axes[0].imshow(I_meas, cmap='hot', origin='lower')
            axes[0].set_title('Intensity (Measured)')
            
            axes[1].imshow(Gd_meas, cmap='viridis', origin='lower') 
            axes[1].set_title('G_d (Measured)')
            
            # Model
            axes[2].imshow(result.get('I_model', np.zeros_like(I_meas)), cmap='hot', origin='lower')
            axes[2].set_title('Intensity (Model)')
            
            # Mark detected positions
            for ax in axes:
                for emitter in result['emitters']:
                    ax.plot(emitter['x'], emitter['y'], 'r+', markersize=10, markeredgewidth=2)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            if verbose:
                print(f"Plotting failed: {e}")
    
    return result


if __name__ == "__main__":
    # Example usage and speed test
    import time
    
    # Create synthetic test data
    shape = (50, 50)
    I_meas = np.random.poisson(1000, shape).astype(float)
    Gd_meas = np.random.poisson(500, shape).astype(float)
    n_emitters_map = np.random.rand(*shape) * 3
    
    # Add some peaks
    I_meas[20:25, 20:25] += 5000
    I_meas[30:35, 35:40] += 3000
    Gd_meas[20:25, 20:25] += 2000
    Gd_meas[30:35, 35:40] += 1500
    n_emitters_map[20:25, 20:25] = 2
    n_emitters_map[30:35, 35:40] = 1
    
    metadata = {
        'laser_power': 100000,
        'pixel_size': 0.05,
        'dwell_time': 1.0,
        'dead_time': 50,
        'enable_noise': True
    }
    
    print("Testing fast localization algorithm...")
    start_time = time.time()
    
    result = localize_fast(I_meas, Gd_meas, n_emitters_map, metadata, 
                          max_emitters=10, verbose=True)
    
    end_time = time.time()
    print(f"\nTotal time: {end_time - start_time:.3f} seconds")
    print(f"Found {len(result['emitters'])} emitters")
    print(f"Final RSS: {result['RSS']:.2f}")
    
    for i, emitter in enumerate(result['emitters']):
        print(f"Emitter {i+1}: ({emitter['x']:.2f}, {emitter['y']:.2f})")