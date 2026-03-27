import numpy as np
from scipy import signal
from scipy import fft

def spad23_deconvolve(emitter_map, psf, n_iterations=50, save_interval=10, verbose=True):
    """
    Deconvolve emitter locations from expected emitter count map using Richardson-Lucy algorithm.
    
    Parameters:
    ----------
    emitter_map : 2D numpy array
        Map containing expected number of emitters at each position, derived from SPAD23 timing data
    psf : 2D numpy array
        Point spread function of the imaging system
    n_iterations : int, optional
        Number of iterations for the deconvolution algorithm (default: 50)
    save_interval : int, optional
        Interval at which to save intermediate results (default: 10)
    verbose : bool, optional
        Whether to print progress information (default: True)
    
    Returns:
    -------
    emitter_locations : 2D numpy array
        Deconvolved map of emitter locations
    iterations_history : list of 2D numpy arrays
        History of the emitter location estimates at specified save intervals
    """
    # Ensure the PSF is normalized
    psf = psf / np.sum(psf)
    
    # Prepare PSF for FFT-based convolution
    # Create a version of the PSF that's flipped for proper convolution
    psf_flip = np.flip(np.flip(psf, 0), 1)
    
    # Initialize the estimate with a uniform distribution
    estimate = np.ones_like(emitter_map)
    
    # For storing the history of iterations
    save_iterations = np.unique(np.concatenate(([n_iterations], 
                                              np.arange(0, n_iterations, save_interval))))
    iterations_history = []
    
    for i in range(1, n_iterations + 1):
        # Forward projection: convolve current estimate with PSF
        conv_estimate = signal.fftconvolve(estimate, psf, mode='same')
        
        # Compute the ratio between the observed data and the current estimate projection
        # This is a key step in Richardson-Lucy deconvolution
        ratio = emitter_map / (conv_estimate + 1e-10)  # Add small value to prevent division by zero
        
        # Backward projection: convolve the ratio with flipped PSF
        correction = signal.fftconvolve(ratio, psf_flip, mode='same')
        
        # Update the estimate
        estimate = estimate * correction
        
        # Save the current estimate if needed
        if i in save_iterations:
            iterations_history.append(estimate.copy())
            
            if verbose:
                print(f"Iteration {i}/{n_iterations} completed")
    
    return estimate, iterations_history


def convolve2d_fft(image, kernel):
    """
    Fast 2D convolution using FFT, similar to the cconv2_PSFk function in the MATLAB code.
    
    Parameters:
    ----------
    image : 2D numpy array
        Image to be convolved
    kernel : 2D numpy array
        Convolution kernel
        
    Returns:
    -------
    2D numpy array
        Convolution result
    """
    # Compute FFTs
    image_fft = fft.fft2(image)
    kernel_fft = fft.fft2(kernel, s=image.shape)
    
    # Multiply in frequency domain
    result_fft = image_fft * kernel_fft
    
    # Inverse FFT to get back to spatial domain
    result = fft.ifft2(result_fft).real
    
    # Shift to center the result
    result = fft.fftshift(result)
    
    return result


def visualize_deconvolution(original_map, psf, emitter_locations, iterations_history=None):
    """
    Visualize the deconvolution results.
    
    Parameters:
    ----------
    original_map : 2D numpy array
        Original map of expected emitter counts
    psf : 2D numpy array
        Point spread function used for deconvolution
    emitter_locations : 2D numpy array
        Deconvolved emitter locations
    iterations_history : list of 2D numpy arrays, optional
        History of intermediate results during deconvolution
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    
    # Create a figure
    if iterations_history:
        n_plots = min(len(iterations_history), 4) + 3
        fig, axes = plt.subplots(1, n_plots, figsize=(n_plots*4, 4))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes = axes.flatten()
    
    # Plot original map
    im0 = axes[0].imshow(original_map, cmap='viridis')
    axes[0].set_title('Original Emitter Map')
    plt.colorbar(im0, ax=axes[0])
    
    # Plot PSF
    im1 = axes[1].imshow(psf, cmap='viridis')
    axes[1].set_title('PSF')
    plt.colorbar(im1, ax=axes[1])
    
    # Plot final result
    im2 = axes[2].imshow(emitter_locations, cmap='viridis', norm=LogNorm(vmin=0.01, vmax=np.max(emitter_locations)))
    axes[2].set_title('Deconvolved Emitter Locations')
    plt.colorbar(im2, ax=axes[2])
    
    # Plot intermediate results if available
    if iterations_history:
        # Select a subset of iterations to display
        indices = np.linspace(0, len(iterations_history)-1, n_plots-3).astype(int)
        
        for i, idx in enumerate(indices):
            im = axes[i+3].imshow(iterations_history[idx], cmap='viridis', 
                                 norm=LogNorm(vmin=0.01, vmax=np.max(iterations_history[idx])))
            axes[i+3].set_title(f'Iteration {idx+1}')
            plt.colorbar(im, ax=axes[i+3])
    
    plt.tight_layout()
    plt.show()
    
    return fig

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import time

def generate_test_data(image_size=100, n_emitters=20, psf_sigma=3.0, noise_level=0.05):
    """
    Generate synthetic test data with random emitter locations and Gaussian PSF.
    
    Parameters:
    ----------
    image_size : int
        Size of the square image
    n_emitters : int
        Number of emitters to simulate
    psf_sigma : float
        Standard deviation of the Gaussian PSF
    noise_level : float
        Level of Poisson noise to add
        
    Returns:
    -------
    true_locations : 2D numpy array
        True emitter locations (binary image)
    measured_map : 2D numpy array
        Simulated measurement of expected emitter counts
    psf : 2D numpy array
        Point spread function used
    """
    # Create empty image
    true_locations = np.zeros((image_size, image_size))
    
    # Generate random emitter positions
    x_positions = np.random.randint(10, image_size-10, n_emitters)
    y_positions = np.random.randint(10, image_size-10, n_emitters)
    
    # Create ground truth binary image
    for x, y in zip(x_positions, y_positions):
        true_locations[x, y] = 1.0
    
    # Create Gaussian PSF
    psf_size = int(6 * psf_sigma)
    if psf_size % 2 == 0:
        psf_size += 1  # Ensure odd size for centering
    
    x = np.linspace(-psf_size//2, psf_size//2, psf_size)
    y = np.linspace(-psf_size//2, psf_size//2, psf_size)
    X, Y = np.meshgrid(x, y)
    psf = np.exp(-(X**2 + Y**2) / (2 * psf_sigma**2))
    psf = psf / np.sum(psf)  # Normalize
    
    # Convolve true locations with PSF to get expected measurements
    measured_map = ndimage.convolve(true_locations, psf, mode='constant')
    
    # Add Poisson noise
    if noise_level > 0:
        # Scale up for Poisson simulation
        scale_factor = 1.0 / noise_level
        scaled_map = measured_map * scale_factor
        noisy_map = np.random.poisson(scaled_map)
        measured_map = noisy_map / scale_factor
    
    return true_locations, measured_map, psf

def evaluate_results(true_locations, deconvolved_locations, threshold=0.5):
    """
    Evaluate the deconvolution results by comparing with ground truth.
    
    Parameters:
    ----------
    true_locations : 2D numpy array
        Ground truth emitter locations
    deconvolved_locations : 2D numpy array
        Deconvolved emitter locations
    threshold : float
        Threshold for binarizing the deconvolved result
        
    Returns:
    -------
    dict
        Dictionary containing evaluation metrics
    """
    # Normalize deconvolved image to max value of 1
    deconvolved_norm = deconvolved_locations / np.max(deconvolved_locations)
    
    # Threshold to get binary image
    binary_result = (deconvolved_norm > threshold).astype(float)
    
    # Find local maxima
    from scipy import ndimage
    neighborhood = ndimage.generate_binary_structure(2, 2)
    local_max = ndimage.maximum_filter(deconvolved_norm, footprint=neighborhood) == deconvolved_norm
    local_max = local_max & (deconvolved_norm > threshold)
    
    # Count true positives, false positives, false negatives
    # For simplicity, consider a detection correct if there's a true emitter 
    # within a small radius (3 pixels) of the detected position
    detection_radius = 3
    
    # Dilate true locations to create acceptance regions
    dilated_true = ndimage.binary_dilation(
        true_locations > 0, 
        structure=np.ones((2*detection_radius+1, 2*detection_radius+1))
    )
    
    # Get detected positions
    detected_y, detected_x = np.where(local_max)
    
    # Count how many detections fall within acceptance regions
    true_positives = 0
    for x, y in zip(detected_x, detected_y):
        if dilated_true[y, x]:
            true_positives += 1
    
    # Count total detections and ground truth emitters
    total_detections = np.sum(local_max)
    total_true = np.sum(true_locations > 0)
    
    # Calculate metrics
    precision = true_positives / max(total_detections, 1)
    recall = true_positives / max(total_true, 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-10)
    
    return {
        'true_positives': true_positives,
        'total_detections': total_detections,
        'total_true': total_true,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def gaussian_2d(x: np.ndarray, y: np.ndarray, 
                amplitude: float, x0: float, y0: float, 
                sigma_x: float, sigma_y: float, offset: float) -> np.ndarray:
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
    return amplitude * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + 
                              (y - y0)**2 / (2 * sigma_y**2))) + offset

def run_test():
    """Run a complete test of the deconvolution algorithm."""
    
    print("Generating synthetic test data...")
    true_locations, measured_map, psf = generate_test_data(image_size=100, 
                                                          n_emitters=30, 
                                                          psf_sigma=3.0,
                                                          noise_level=0.1)
    
    print(f"SHAPE OF PSF: {psf.shape}")
    print(f"SHAPE OF MEASURED MAP: {measured_map.shape}")
    print("Running deconvolution...")
    start_time = time.time()


    #load emitter map and psf
    import pickle
    import datetime
    date = datetime.date.today().strftime("%Y-%m-%d")
    date = '2025-05-15'
    area_size = '(1, 1)'
    psf_sigma_G = 0.16
    with open(f'./project/data/{date}/nr_emitter_map_{area_size}.pkl', 'rb') as f:
        emitter_map = pickle.load(f).T
    psf = np.zeros((20, 20))
    psf += gaussian_2d(0, 0, 32000**2/1000000, 0, 0, psf_sigma_G, psf_sigma_G, 0)
    print(f"SHAPE OF PSF: {psf.shape}")
    print(f"SHAPE OF MEASURED MAP: {emitter_map.shape}")
    emitter_locations, iterations_history = spad23_deconvolve(
        emitter_map, psf, n_iterations=50, save_interval=10, verbose=True
    )
    
    elapsed_time = time.time() - start_time
    print(f"Deconvolution completed in {elapsed_time:.2f} seconds")
    
    # Evaluate the results
    evaluation = evaluate_results(true_locations, emitter_locations)
    print("\nEvaluation Results:")
    print(f"True Positives: {evaluation['true_positives']} / {evaluation['total_true']}")
    print(f"Total Detections: {evaluation['total_detections']}")
    print(f"Precision: {evaluation['precision']:.3f}")
    print(f"Recall: {evaluation['recall']:.3f}")
    print(f"F1 Score: {evaluation['f1_score']:.3f}")
    
    # Visualize results
    print("\nVisualizing results...")
    fig1 = plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(true_locations, cmap='viridis')
    plt.title('Ground Truth')
    plt.colorbar()
    
    plt.subplot(132)
    plt.imshow(emitter_map, cmap='viridis')
    plt.title('Measured Map')
    plt.colorbar()
    
    plt.subplot(133)
    plt.imshow(emitter_locations, cmap='viridis')
    plt.title('Deconvolved Result')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    # Visualize deconvolution progress
    fig2 = visualize_deconvolution(measured_map, psf, emitter_locations, iterations_history)
    
    return true_locations, measured_map, psf, emitter_locations, iterations_history

if __name__ == "__main__":
    run_test()