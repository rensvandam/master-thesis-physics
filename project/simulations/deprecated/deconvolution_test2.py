import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.restoration import richardson_lucy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def deconvolve_emitter_locations(emitter_map, psf, num_iterations=50, plot_results=True):
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

    print(f"Starting Richardson-Lucy deconvolution with {num_iterations} iterations...")
    deconvolved_image = richardson_lucy(emitter_map, psf, num_iter=num_iterations, clip=False)
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

def gaussian_2d(shape,A,sigma) -> np.ndarray:
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

    center_y, center_x = shape[0] // 2, shape[1] // 2
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    psf = A*np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
    return psf #/ np.sum(psf) # Normalize


def circular_psf(shape, diameter):
    center_y, center_x = shape[0] // 2, shape[1] // 2
    y, x = np.mgrid[0:shape[0], 0:shape[1]]

    # Calculate the squared distance from the center for each pixel
    distance_squared = (x - center_x)**2 + (y - center_y)**2

    # Create a circular mask
    radius = diameter / 2
    psf = np.zeros(shape)
    psf[distance_squared <= radius**2] = 1

    return psf
# --- Example Usage ---
if __name__ == "__main__":
    # 1. Create a synthetic emitter map and PSF for demonstration
    # In your actual application, 'emitter_map' will be your captured image,
    # and 'psf' will be the experimentally determined or theoretically
    # modeled PSF of your SPAD system.

    # # Synthetic Emitter Locations (Ground Truth)
    # true_emitters = np.zeros((30, 30))
    # true_emitters[10, 10] = 1.0 # Single emitter
    # true_emitters[20, 20] = 1.0 # Single emitter
    # true_emitters[5, 15] = 2.0  # Two emitters close together
    # true_emitters[15, 5] = 0.5  # Half an emitter (for testing fractional values)

    # # Synthetic PSF (e.g., a 2D Gaussian approximation)
    # # The PSF should be centered and normalized.
    # def gaussian_psf(shape, sigma):
    #     center_y, center_x = shape[0] // 2, shape[1] // 2
    #     y, x = np.mgrid[0:shape[0], 0:shape[1]]
    #     psf = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
    #     return psf / np.sum(psf) # Normalize

    # psf_size = 7
    # psf_sigma = 1.5
    # synthetic_psf = gaussian_psf((psf_size, psf_size), psf_sigma)

    # # Convolve true emitters with PSF to create a "blurred" map (your input)
    # # Use 'same' mode to keep the output size same as input
    # synthetic_emitter_map = convolve2d(true_emitters, synthetic_psf, mode='same', boundary='symm')

    # # Add some noise to simulate real data (e.g., Poisson noise)
    # # For a real scenario, you'd apply Poisson noise to the photon counts before
    # # converting to 'expected number of emitters'. Here, we'll just add some
    # # general noise to make it more realistic for demonstration.
    # synthetic_emitter_map_noisy = np.random.poisson(synthetic_emitter_map * 10) / 10.0 # Scale to get desired values
    # synthetic_emitter_map_noisy[synthetic_emitter_map_noisy < 0] = 0 # Ensure non-negative

    # print("\n--- Running Deconvolution on Synthetic Data ---")
    # deconvolved_result = deconvolve_emitter_locations(
    #     synthetic_emitter_map_noisy,
    #     synthetic_psf,
    #     num_iterations=70, # Can adjust this
    #     plot_results=True
    # )

    # # You can now analyze 'deconvolved_result' to find peak locations,
    # # potentially using peak finding algorithms if you need precise sub-pixel
    # # localization.

    # # Optional: Compare with ground truth if you have it (for synthetic data)
    # # This requires a ground truth emitter map, which you don't have for your real data.
    # # psnr = peak_signal_noise_ratio(true_emitters, deconvolved_result)
    # # ssim = structural_similarity(true_emitters, deconvolved_result, data_range=deconvolved_result.max() - deconvolved_result.min())
    # # print(f"\nPSNR (true vs deconvolved): {psnr:.2f}")
    # # print(f"SSIM (true vs deconvolved): {ssim:.2f}")


    # import numpy as np
# Assume 'your_emitter_data' is your 2D numpy array of emitter map
# Assume 'your_psf_data' is your 2D numpy array of PSF

    # Load your data here (example placeholder)
    #load emitter map and psf
    import pickle
    import datetime
    date = datetime.date.today().strftime("%Y-%m-%d")
    date = '2025-05-14'
    area_size = '(4, 4)'
    psf_sigma_G = 0.16
    with open(f'./project/data/{date}/nr_emitter_map_{area_size}.pkl', 'rb') as f:
        emitter_map = pickle.load(f).T
    with open(f'./project/data/{date}/photon_count_map_{area_size}.pkl', 'rb') as f:
        I_meas = pickle.load(f).T
    with open(f'./project/data/{date}/G2_difference_map_{area_size}.pkl', 'rb') as f:
        Gd_meas = pickle.load(f).T

    #filter high vlaues of emitter map out
    emitter_map[emitter_map > 10] = 1
    psf = gaussian_2d((15,15),1, 3)
    #instead of gaussian, try circle with same diameter as gaussian
    #psf = circular_psf((10,10), 7)

    # your_emitter_data = np.load('path/to/your_emitter_map.npy')
    # your_psf_data = np.load('path/to/your_psf.npy')
    # Or if it's an image:
    # from PIL import Image
    # your_emitter_data = np.array(Image.open('path/to/your_emitter_map.png').convert('L')) # convert to grayscale
    # your_psf_data = np.array(Image.open('path/to/your_psf.png').convert('L'))

    # If your image is uint8, you might want to convert to float for calculations
    # your_emitter_data = your_emitter_data.astype(float)
    # your_psf_data = your_psf_data.astype(float)

    #plot I_meas
    fig, ax = plt.subplots()
    ax.imshow(I_meas, cmap='hot')

    deconvolved_locations = deconvolve_emitter_locations(
        Gd_meas,
        psf,
        num_iterations=70,  # Adjust as needed
        plot_results=True
    )