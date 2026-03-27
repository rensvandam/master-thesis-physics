# New module for ISM implementation
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from scipy import fft
from scipy.ndimage import fourier_shift, shift
from scipy.optimize import curve_fit
from skimage.registration import phase_cross_correlation
from skimage.filters import gaussian
from project.model.detection import Spad23  # Import Spad23 class
from project.model.coherence_from_data import coherence, calculate_G2_difference

class ISMProcessor:
    """
    Image Scanning Microscopy processor for SPAD23 detector data.
    
    This class implements pixel reassignment and image reconstruction methods
    for ISM with a SPAD23 detector array.
    
    Attributes
    ----------
    alpha : float
        Reassignment parameter. Typically 0.5 for optimal resolution improvement.
    detector_positions : np.ndarray
        Array of (x,y) positions for each detector element.
    """
    
    def __init__(self, sensor, alpha=0.5):
        """
        Initialize the ISM processor.
        
        Parameters
        ----------
        sensor : Spad23
            The Spad23 sensor object to get detector positions from.
        alpha : float
            Pixel reassignment parameter, typically 0.5 for optimal resolution.
        """
        self.alpha = alpha
        self.sensor = sensor
        
        # Get the detector positions from the Spad23 object
        self.detector_positions = sensor.pixel_coordinates
        
        # Store the number of pixels for convenience
        self.nr_pixels = sensor.nr_pixels

    def process_scan_data(self, scan_data, detector_data, plot=False):
        """
        Process scan data to create an ISM image.
        
        Parameters
        ----------
        scan_data : dict
            Scan data dictionary from ScanningSetup.perform_ism_scan. Contains:
            'area_size': the size of the scanned area (x, y) in micrometers,
            'positions': the scanning positions (x,y) in pixels,
            'step_size': (x_step, y_step),
            'dwell_time': dwell time in ms,
            'photon_count_map': a map of photon counts for each position,
            'position_data': (dict): 'position': (ix, iy),'coordinates': micrometer position, 'photon_count': photon_count_map (ix, iy)
            'g2_data': (dict): (nr_lags,) with (coherence (array), bins (array) entries
        detector_data : dict
            Dictionary with detector data indexed by scan position (ix, iy)
            Each entry contains photon data detected by each detector element (num_of_detected_photons, 2), each item (detector id,timestamp)
        upsampling_factor : int
            Factor by which to increase the resolution in the final image
            
        Returns
        -------
        dict
            Dictionary with ISM results including reassigned photon positions and
            reconstructed images
        """
        self.scan_data = scan_data
        self.detector_data = detector_data
        
        area_size = scan_data['area_size']
        ix_max, iy_max = scan_data['positions']
        # Initialize images for each detector (23 pixels)
        num_detectors = len(self.detector_positions)
        detector_images = np.zeros((ix_max, iy_max, num_detectors), dtype=float)
        # Fill detector images with photon counts
        for pos, data in detector_data.items():
            ix, iy = pos
            # In the data dictionary, count the number of detected photons at every pixel
            detector_counts = np.zeros(num_detectors)
            for detector_id in range(num_detectors):
                photons = data[detector_id]
                if photons is not None:
                    detector_counts[detector_id] += len(photons)
            
            # Assign counts to corresponding image positions
            for detector_id in range(num_detectors):
                detector_images[ix, iy, detector_id] = detector_counts[detector_id]
            
            #print(f"Starting upsampling for position {pos}")
            #detector_images = self.upsample(detector_images, scale_factor=2)

        # Determine shift vectors
        shift_vectors = self.getShiftVectors(detector_images, plot=plot)
        
        shifted_images = self.shift(detector_images, shift_vectors)

        # sum 23 images from img_ism
        ism_image = np.sum(shifted_images, axis=2)

        # regular image
        confocal_image = np.sum(detector_images, axis=2)

        # show both images
        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            # share ylabel between ax0 and ax1
            

            ax0 = axes[0].imshow(confocal_image, cmap='hot', extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2], origin='lower')
            #axes[0].set_title(f"Original Image")
            #axes[0].set_xlabel("x position (pixels)")
            #axes[0].set_ylabel("y position (pixels)")
            #turn of x and y ticks
            #axes[0].set_xticks([])
            #axes[0].set_yticks([])
            #plt.colorbar(ax0, label='Photon Count', shrink = 0.65)
            ax1 = axes[1].imshow(ism_image, cmap='hot', extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2], origin='lower')
            #axes[1].set_title(f"ISM image")
            #axes[1].set_xlabel("x position (pixels)")
            #axes[1].set_ylabel("y position (pixels)")
            #turn of x and y ticks
            #axes[1].set_xticks([])
            #axes[1].set_yticks([])
            #plt.colorbar(ax1, label='Photon Count', shrink = 0.65)
            #add length bar
            #axes[0].plot([0.3, -0.4], [0.4, -0.4], color='white', linewidth=5)
            #plt.text(8.5, 8.9, '100 nm', color='white', fontsize=18, ha='center')


            # add length bar


            plt.show()

        return {
            'ism_image': ism_image,
            'confocal_image': confocal_image,
            'shift_vectors': shift_vectors,
            'detector_images': detector_images,
        }
    

    def process_scan_data_new(self, scan_data, detector_data, plot=False):
        """
        Process scan data to create an ISM image for both the intensity and G2_difference.
        
        Parameters
        ----------
        scan_data : dict
            Scan data dictionary from ScanningSetup.perform_ism_scan. Contains:
            'area_size': the size of the scanned area (x, y) in micrometers,
            'positions': the scanning positions (x,y) in pixels,
            'step_size': (x_step, y_step),
            'dwell_time': dwell time in ms,
            'photon_count_map': a map of photon counts for each position,
            'position_data': (dict): 'position': (ix, iy),'coordinates': micrometer position, 'photon_count': photon_count_map (ix, iy)
            'g2_data': (dict): (nr_lags,) with (coherence (array), bins (array) entries
            'G2_diff_map': (ix, iy) with G2 difference values for each position
        detector_data : dict
            Dictionary with detector data indexed by scan position (ix, iy)
            Each entry contains photon data detected by each detector element (num_of_detected_photons, 2), each item (detector id,timestamp)
        upsampling_factor : int
            Factor by which to increase the resolution in the final image
            
        Returns
        -------
        dict
            Dictionary with ISM results including reassigned photon positions and
            reconstructed images
        """
        self.scan_data = scan_data
        self.detector_data = detector_data
        
        area_size = scan_data['area_size']
        ix_max, iy_max = scan_data['positions']
        # Initialize images for each detector (23 pixels)
        num_detectors = len(self.detector_positions)

        ########### INTENSITY ISM ##################

        detector_images = np.zeros((ix_max, iy_max, num_detectors), dtype=float)
        # Fill detector images with photon counts
        for pos, data in detector_data.items():
            ix, iy = pos
            # In the data dictionary, count the number of detected photons at every pixel
            detector_counts = np.zeros(num_detectors)
            for detector_id in range(num_detectors):
                photons = data[detector_id]
                if photons is not None:
                    detector_counts[detector_id] += len(photons)
            
            # Assign counts to corresponding image positions
            for detector_id in range(num_detectors):
                detector_images[ix, iy, detector_id] = detector_counts[detector_id]
            
            #print(f"Starting upsampling for position {pos}")
            #detector_images = self.upsample(detector_images, scale_factor=2)

        # Determine shift vectors
        shift_vectors = self.getShiftVectors(detector_images, plot=plot)
        
        shifted_images = self.shift(detector_images, shift_vectors)

        # sum 23 images from img_ism
        ism_image = np.sum(shifted_images, axis=2)

        # regular image
        confocal_image = np.sum(detector_images, axis=2)

        ##################### G2 DIFFERENCE ISM ####################

        # Total number of G2 pair images: num_detectors*(num_detectors-1)//2
        G2_detector_images = np.zeros((ix_max, iy_max, num_detectors*(num_detectors-1)//2), dtype=float)
        # Fill detector images with photon counts
        for pos, data in detector_data.items():
            ix, iy = pos

            pairs = list(combinations(range(num_detectors), 2))
            for detector_id_pair, (detector_id, detector_id2) in enumerate(pairs):
                #print(f"Calculating G2 for pair {detector_id} and {detector_id2} at position {pos}, length: {len(data[detector_id])} and {len(data[detector_id2])}")
                #print(data[detector_id].shape, data[detector_id2].shape)
                #print(data[detector_id], data[detector_id2])
                if len(data[detector_id]) == 0 or len(data[detector_id2]) == 0:
                    G2_detector_images[ix, iy, detector_id_pair] = 0.0
                    continue
                detector_pair_coherence, _ = coherence(
                    data[detector_id][:,1],
                    data[detector_id2][:,1],
                    interval=scan_data['dwell_time']*1E6, #dwell time in ns
                    bin_size = 0.1,
                    nr_steps = 200,
                    offset = 0,
                    normalize = False,
                    auto_correlation=False
                )
                G2_detector_images[ix, iy, detector_id_pair] = calculate_G2_difference(
                    detector_pair_coherence, tau_min = 50)['difference']
        
        #print(f"Total number of G2 pair images: {G2_detector_images.shape[2]}")

        # Determine shift vectors
        G2_shift_vectors = self.getShiftVectorsG2(G2_detector_images, plot=plot)
        
        G2_shifted_images = self.shift(G2_detector_images, G2_shift_vectors)

        # sum 253 images from img_ism
        G2_ism_image = np.sum(G2_shifted_images, axis=2)

        # regular G2 image
        G2_confocal_image = np.sum(G2_detector_images, axis=2)
            # Assign counts to corresponding image positions
            #for detector_id in range(num_detectors):
                #detector_images[ix, iy, detector_id] = detector_counts[detector_id]
        # show both images
        if plot:
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            # share ylabel between ax0 and ax1
            
            
            ax0 = axes[0,0].imshow(confocal_image, cmap='hot', extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2], origin='lower')

            axes[0,0].set_xticks([])
            axes[0,0].set_yticks([])
            # add distance bar white
            axes[0,0].plot([0.3, 0.4], [-0.4, -0.4], color='white', linewidth=5)

            #axes[0].set_title(f"Original Image")
            #axes[0].set_xlabel("x position (pixels)")
            #axes[0].set_ylabel("y position (pixels)")
            #turn of x and y ticks
            #axes[0].set_xticks([])
            #axes[0].set_yticks([])
            plt.colorbar(ax0, shrink=0.8)
            
            ax1 = axes[0,1].imshow(ism_image, cmap='viridis', extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2], origin='lower')

            axes[0,1].set_xticks([])
            axes[0,1].set_yticks([])
            # add distance bar white
            axes[0,1].plot([0.3, 0.4], [-0.4, -0.4], color='white', linewidth=5)

            plt.colorbar(ax1, shrink=0.8)

            #ax1.set_title("ISM image")
            #axes[1].set_title(f"ISM image")
            #axes[1].set_xlabel("x position (pixels)")
            #axes[1].set_ylabel("y position (pixels)")
            #turn of x and y ticks
            #axes[1].set_xticks([])
            #axes[1].set_yticks([])
            #plt.colorbar(ax1, label='Photon Count', shrink = 0.65)
            #add length bar
            #axes[0].plot([0.3, -0.4], [0.4, -0.4], color='white', linewidth=5)
            #plt.text(8.5, 8.9, '100 nm', color='white', fontsize=18, ha='center')
            #colormap blue green: 
            ax2 = axes[1,0].imshow(G2_confocal_image, cmap='viridis', extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2], origin='lower')
            axes[1,0].set_xticks([])
            axes[1,0].set_yticks([])
            # add distance bar white
            axes[1,0].plot([0.3, 0.4], [-0.4, -0.4], color='white', linewidth=5)

            plt.colorbar(ax2, shrink=0.8)


            ax3 = axes[1,1].imshow(G2_ism_image, cmap='viridis', extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2], origin='lower')
            axes[1,1].set_xticks([])
            axes[1,1].set_yticks([])
            # add distance bar white
            axes[1,1].plot([0.3, 0.4], [-0.4, -0.4], color='white', linewidth=5)

            plt.colorbar(ax3, shrink=0.8)

            plt.savefig("results-PSF-ISM.png", dpi=300, bbox_inches='tight', format='png')
            plt.show()

            ###########################
            ### Fitting the gaussians
            ###########################

            # Gaussian function for fitting
            def gaussian(x, A, x0, sigma):
                return A * np.exp(-(x - x0)**2 / (2 * sigma**2))

            # Get image shape
            img_shape = confocal_image.shape
            center_y = img_shape[0] // 2

            # Extract horizontal cross-sections through the center
            confocal_line = confocal_image[center_y, :]
            ism_line = ism_image[center_y, :]
            G2_confocal_line = G2_confocal_image[center_y, :]
            G2_ism_line = G2_ism_image[center_y, :]

            # Normalize to max = 1
            confocal_norm = confocal_line / np.max(confocal_line)
            ism_norm = ism_line / np.max(ism_line)
            G2_confocal_norm = G2_confocal_line / np.max(G2_confocal_line)
            G2_ism_norm = G2_ism_line / np.max(G2_ism_line)

            # Create x-axis in physical units
            x_vals = np.linspace(-area_size[0]/2, area_size[0]/2, img_shape[1])

            # Fit each normalized profile to a Gaussian
            fits = {}
            for label, data in zip(
                ['Confocal', 'ISM', 'G² Confocal', 'G² ISM'],
                [confocal_norm, ism_norm, G2_confocal_norm, G2_ism_norm]
            ):
                # Initial guesses: A=1, x0=0, sigma=0.1 * area_size[0]
                popt, _ = curve_fit(gaussian, x_vals, data, p0=[1, 0, 0.1 * area_size[0]])
                fits[label] = {
                    'params': popt,
                    'sigma': popt[2]
                }


            # Plotting the fitted PSFs
            plt.figure(figsize=(10, 6))

            plt.plot(x_vals, confocal_norm, label='Confocal', color='red')
            plt.plot(x_vals, ism_norm, label='ISM', color='orange')
            plt.plot(x_vals, G2_confocal_norm, label=r'$\Delta G^{(2)}$', color='blue')
            plt.plot(x_vals, G2_ism_norm, label=r'$\Delta G^{(2)}$ ISM', color='green')

            #plot the square of confocal
            #plt.plot(x_vals, confocal_norm**2, label='Confocal²', color='orange', linestyle='--')

            plt.xlabel('x position [μm]')
            plt.ylabel('Normalized Intensity')
            #plt.title('Center Horizontal PSF Cross-sections and Gaussian Fits')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("psf_fits.png", dpi=300, bbox_inches='tight', format='png')
            plt.show()

            # Print sigma values
            for label in fits:
                print(f"{label} sigma (μm): {fits[label]['sigma']:.4f}")
            ########################
        return {
            'ism_image': ism_image,
            'confocal_image': confocal_image,
            'shift_vectors': shift_vectors,
            'detector_images': detector_images,
            'G2_ism_image': G2_ism_image,
            'G2_confocal_image': G2_confocal_image,
            'G2_shift_vectors': G2_shift_vectors,
            'G2_detector_images': G2_detector_images,
        }
    
    def getShiftVectorsG2(self, data, plot=False):
            """
            Calculate the shift vectors between the images taken at different detector positions.

            Parameters
            ----------
            data (ndarray): The data array containing images from different detector positions. Shape: (nx, ny, num_detectors)
            plot (bool): If True, plot the shift vectors.

            Returns
            -------
            shift_vectors (ndarray): The calculated shift vectors for each detector position. Shape: (num_detectors, 2)
            
            """
            # max detector x and y (for plotting)
            max_det_x = np.max(self.detector_positions[:, 0])
            max_det_y = np.max(self.detector_positions[:, 1])
            # min detector x and y
            min_det_x = np.min(self.detector_positions[:, 0])
            min_det_y = np.min(self.detector_positions[:, 1])

            center_detector_index = int(len(self.detector_positions)/2)
            # Get the center image
            #center_image = data[:, :, center_detector_index]

            # Center image is the average of the center pixel pixel_to_pixel coherences.
            # This corresponds to pixel number 11's coherences with all other pixels
            # select images from center pixel coherences
            num_detectors = 23 #hardcode #TODO
            pairs = list(combinations(range(num_detectors), 2))
            pairs_midpoints = []
            center_image = np.zeros(data[:, :, 0].shape)
            for pair in pairs:
                # First, fill pairs_midpoints. These are the origin points for the shift vectors in a visualisation.
                
                # Calculate the midpoint of the pair
                midpoint_x = (self.detector_positions[pair[0], 0] + self.detector_positions[pair[1], 0]) / 2
                midpoint_y = (self.detector_positions[pair[0], 1] + self.detector_positions[pair[1], 1]) / 2
                pairs_midpoints.append((midpoint_x, midpoint_y))
                #print(f"Pair {pair} midpoint: {midpoint_x}, {midpoint_y}")

                # Creation of 'center image'
                if pair[0] == center_detector_index or pair[1] == center_detector_index:
                    #print(f"Adding image for pair {pair}")
                    # pair index:
                    pair_index = pairs.index(pair)
                    center_image += data[:, :, pair_index]
            center_image /= (num_detectors - 1)

            # Calculate phase correlation to find shift between ISM images 
            #shift_vectors = np.zeros((len(self.detector_positions), 2))
            #for j in range(len(self.detector_positions)):
            shift_vectors = np.zeros((data.shape[2], 2))
            for j in range(data.shape[2]):
                shift_vectors[j] = self.image_correlation(center_image, data[:, :, j])

            if plot:
                # Create figure with equal-sized subplots
                fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1]})
                # set font size, tick font size etc
                plt.rcParams.update({'font.size': 15})
                plt.rcParams['xtick.labelsize'] = 20
                plt.rcParams['xtick.major.size'] = 20
                plt.rcParams['ytick.labelsize'] = 20
                plt.rcParams['axes.labelsize'] = 20
                plt.rcParams['axes.titlesize'] = 15
                plt.rcParams['figure.titlesize'] = 15
                plt.rcParams['legend.fontsize'] = 15



                # Create more room between subplots
                plt.subplots_adjust(wspace=0.4)
                


                #print(self.detector_positions.shape)
                #print(range(len(shift_vectors)))
                # shift vector locations must be the detector positions plus the individual pixel locations in that detector.
                # create detector positions where each one is repeated 23 times
                detector_positions_multiplied = np.repeat(self.detector_positions, 11, axis=0)
                # First plot - shift vectors
                for i in range(len(shift_vectors)):
                    #print(i)
                    # axes[0].quiver(self.detector_positions[i, 0], self.detector_positions[i, 1], 
                    #             -shift_vectors[i, 0], -shift_vectors[i, 1], 
                    #             angles='xy', scale_units='xy', scale=0.2, color='r')
                    # axes[0].quiver(detector_positions_multiplied[i, 0], detector_positions_multiplied[i, 1], 
                    #             -shift_vectors[i, 0], -shift_vectors[i, 1], 
                    #             angles='xy', scale_units='xy', scale=0.2, color='r')
                    axes[0].quiver(pairs_midpoints[i][0], pairs_midpoints[i][1],
                                -shift_vectors[i, 0], -shift_vectors[i, 1], 
                                angles='xy', scale_units='xy', scale=0.2, color='r')

                # Plot detector positions
                axes[0].scatter(self.detector_positions[:, 0], self.detector_positions[:, 1], c='b', marker='o')

                # Plot detector pair midpoints (size half of detector positions, bit transparent)
                axes[0].scatter(*zip(*pairs_midpoints), c='magenta', marker='o', s=10, alpha=1)
                
                # Write shift vector magnitudes
                # for i in range(len(shift_vectors)):
                #     axes[0].text(self.detector_positions[i, 0], self.detector_positions[i, 1], 
                #                 f"({shift_vectors[i, 0]:.2f}, {shift_vectors[i, 1]:.2f})", 
                #                 fontsize=8, color='r')
                
                # Set axis limits
                axes[0].set_xlim(min_det_x - 10, max_det_x + 10)
                axes[0].set_ylim(min_det_y - 10, max_det_y + 10)
                #axes[0].set_title("Shift vectors in the SPAD23 detector frame, \n with the shifts in number of pixels", loc='center')
                
                # Make the first plot square
                axes[0].set_aspect('equal')
                axes[0].set_xlabel("x position [µm]", fontsize=20)
                axes[0].set_ylabel("y position [µm]", fontsize=20)
                
                # Second plot - example image with shift vector
                # Combine images for reference
                image_combined = np.zeros(data[:, :, 0].shape)
                example_position = 3
                for j in [center_detector_index, example_position]:
                    image_combined += data[:, :, j]
                    #print(shift_vectors[j])
                
                # Display example image
                ax1 = axes[1].imshow(data[:,:,example_position], cmap='viridis', 
                                    extent=[0, 20, 0, 20], origin='lower')
                
                # Add quiver for shift vector
                axes[1].quiver(10, 10, shift_vectors[example_position][1], shift_vectors[example_position][0], 
                            angles='xy', scale_units='xy', scale=1, color='r')
                
                # Set title with information
                # axes[1].set_title(
                #     f"Example scanning image with (scaled) shift vector.\n"
                #     f"Shift vector length: {np.round(np.sqrt(shift_vectors[example_position][0]**2 + shift_vectors[example_position][1]**2) * self.scan_data['step_size'][0], 2)} μm\n"
                #     f"Detector distance from center: {np.round(np.sqrt(self.detector_positions[example_position][0]**2 + self.detector_positions[example_position][1]**2) / self.sensor.magnification, 2)} μm",
                #     loc='center'
                # )
                
                # Labels
                axes[1].set_xlabel("x position [pix]", fontsize=20)
                axes[1].set_ylabel("y position [pix]", fontsize=20)
                
                # Make second plot square as well
                axes[1].set_aspect('equal')
                
                # Add colorbar with appropriate height
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(axes[1])
                cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(ax1, cax=cax)
                
                # Display the figure
                plt.tight_layout()  # Adjust layout to prevent overlap

                plt.savefig("shift_vectors_G2.png", dpi=300, bbox_inches='tight', format='png')

                plt.show()
            
            return shift_vectors

    def upsample(self, images, scale_factor=2):
        """
        Upsample a batch of images using interpolation.
        
        Parameters:
        -----------
        images : numpy.ndarray
            Input images as numpy array with shape [ix, iy, number_of_images]
        scale_factor : int or float
            Scale factor for upsampling (default: 2)
            
        Returns:
        --------
        numpy.ndarray
            Upsampled images with shape [ix*scale_factor, iy*scale_factor, number_of_images]
        """
        ix, iy, num_images = images.shape
        new_ix, new_iy = int(ix * scale_factor), int(iy * scale_factor)
        upsampled_images = np.zeros((new_ix, new_iy, num_images), dtype=images.dtype)
        
        # Process each image in the batch
        for i in range(num_images):
            # Perform cubic interpolation for upsampling
            upsampled = self.interpolation(images[:, :, i], scale_factor)
            upsampled_images[:, :, i] = upsampled
        
        return upsampled_images
    
    def interpolation(self, image, scale_factor):
        """
        Performs cubic interpolation on a 2D image using RegularGridInterpolator to upscale it.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image as a 2D numpy array
        scale_factor : int or float
            Factor by which to scale the image (e.g., 2 for doubling the size)
        
        Returns:
        --------
        numpy.ndarray
            Upscaled image
        """
        # Get original dimensions
        height, width = image.shape
        
        # Calculate new dimensions
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        # Create coordinate grids for the original image
        y_orig = np.arange(0, height)
        x_orig = np.arange(0, width)
        
        # Create the interpolation function
        # RegularGridInterpolator expects points in format (y, x)
        interp_func = RegularGridInterpolator(
            (y_orig, x_orig), 
            image, 
            method='linear',
            bounds_error=False,
            fill_value=None  # Extrapolate values outside bounds
        )
        
        # Create meshgrid for the new coordinates
        y_new = np.linspace(0, height-1, new_height)
        x_new = np.linspace(0, width-1, new_width)
        Y_new, X_new = np.meshgrid(y_new, x_new, indexing='ij')
        
        # Stack coordinates for interpolation
        points = np.stack((Y_new.flatten(), X_new.flatten()), axis=-1)
        
        # Apply interpolation
        upscaled_flat = interp_func(points)
        
        # Reshape back to 2D
        upscaled_image = upscaled_flat.reshape((new_height, new_width))
        
        return upscaled_image

    def shift(self, data, shift_vectors):
        """
        Shift each image in the data array according to the provided shift vectors.
        
        Parameters:
        data (ndarray): Array of images with shape (nx, ny, nimages)
        shift_vectors (ndarray): Array of shift vectors with shape (nimages, 2)
        
        Returns:
        ndarray: Array of shifted images with same shape as input data
        """
        nx, ny, nimages = data.shape
        shifted_data = np.zeros_like(data)
        
        # Loop through each image
        for i in range(nimages):
            # Get current image and its shift vector
            img = data[:, :, i]
            shift = shift_vectors[i]  # [y_shift, x_shift]
            
            # Create coordinate matrices for the shifted image
            y, x = np.mgrid[0:nx, 0:ny]
            
            # Calculate source coordinates (where to sample from original image)
            source_y = y - shift[0]
            source_x = x - shift[1]
            
            # Use bilinear interpolation for smoother results
            # Create interpolator function
            from scipy.interpolate import RegularGridInterpolator
            
            # Define original grid points
            y_points = np.arange(0, nx)
            x_points = np.arange(0, ny)
            
            # Create interpolator (set 'fill_value' to handle out-of-bounds)
            interpolator = RegularGridInterpolator(
                (y_points, x_points), 
                img, 
                bounds_error=False, 
                fill_value=0
            )
            # Prepare points for interpolation
            points = np.stack((source_y.flatten(), source_x.flatten()), axis=-1)
            
            # Interpolate and reshape back to image dimensions
            shifted_img = interpolator(points).reshape(nx, ny)
            
            # Store the shifted image
            shifted_data[:, :, i] = shifted_img
        
        return shifted_data

    def getShiftVectors(self, data, plot=False):
        """
        Calculate the shift vectors between the images taken at different detector positions.

        Parameters
        ----------
        data (ndarray): The data array containing images from different detector positions. Shape: (nx, ny, num_detectors)
        plot (bool): If True, plot the shift vectors.

        Returns
        -------
        shift_vectors (ndarray): The calculated shift vectors for each detector position. Shape: (num_detectors, 2)
        
        """
        # max detector x and y (for plotting)
        max_det_x = np.max(self.detector_positions[:, 0])
        max_det_y = np.max(self.detector_positions[:, 1])
        # min detector x and y
        min_det_x = np.min(self.detector_positions[:, 0])
        min_det_y = np.min(self.detector_positions[:, 1])

        center_detector_index = int(len(self.detector_positions)/2) 

        # Get the center image
        center_image = data[:, :, center_detector_index]
        # Calculate phase correlation to find shift between ISM images 
        shift_vectors = np.zeros((len(self.detector_positions), 2))
        for j in range(len(self.detector_positions)):
            shift_vectors[j] = self.image_correlation(center_image, data[:, :, j])

        if plot:
            # Create figure with equal-sized subplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1]})
            # set font size, tick font size etc
            plt.rcParams.update({'font.size': 15})
            plt.rcParams['xtick.labelsize'] = 20
            plt.rcParams['xtick.major.size'] = 20
            plt.rcParams['ytick.labelsize'] = 20
            plt.rcParams['axes.labelsize'] = 20
            plt.rcParams['axes.titlesize'] = 15
            plt.rcParams['figure.titlesize'] = 15
            plt.rcParams['legend.fontsize'] = 15



            # Create more room between subplots
            plt.subplots_adjust(wspace=0.4)
            
            # First plot - shift vectors
            for i in range(len(shift_vectors)):
                axes[0].quiver(self.detector_positions[i, 0], self.detector_positions[i, 1], 
                            -shift_vectors[i, 0], -shift_vectors[i, 1], 
                            angles='xy', scale_units='xy', scale=0.2, color='r')
            

            # Plot detector positions
            axes[0].scatter(self.detector_positions[:, 0], self.detector_positions[:, 1], c='b', marker='o')
            
            # Write shift vector magnitudes
            # for i in range(len(shift_vectors)):
            #     axes[0].text(self.detector_positions[i, 0], self.detector_positions[i, 1], 
            #                 f"({shift_vectors[i, 0]:.2f}, {shift_vectors[i, 1]:.2f})", 
            #                 fontsize=8, color='r')
            
            # Set axis limits
            axes[0].set_xlim(min_det_x - 10, max_det_x + 10)
            axes[0].set_ylim(min_det_y - 10, max_det_y + 10)
            #axes[0].set_title("Shift vectors in the SPAD23 detector frame, \n with the shifts in number of pixels", loc='center')
            
            # Make the first plot square
            axes[0].set_aspect('equal')
            axes[0].set_xlabel("x position [µm]", fontsize=20)
            axes[0].set_ylabel("y position [µm]", fontsize=20)
            
            # Second plot - example image with shift vector
            # Combine images for reference
            image_combined = np.zeros(data[:, :, 0].shape)
            example_position = 3
            for j in [center_detector_index, example_position]:
                image_combined += data[:, :, j]
                #print(shift_vectors[j])
            
            # Display example image
            ax1 = axes[1].imshow(data[:,:,example_position], cmap='viridis', 
                                extent=[0, 20, 0, 20], origin='lower')
            
            # Add quiver for shift vector
            axes[1].quiver(10, 10, shift_vectors[example_position][1], shift_vectors[example_position][0], 
                        angles='xy', scale_units='xy', scale=1, color='r')
            
            # Set title with information
            # axes[1].set_title(
            #     f"Example scanning image with (scaled) shift vector.\n"
            #     f"Shift vector length: {np.round(np.sqrt(shift_vectors[example_position][0]**2 + shift_vectors[example_position][1]**2) * self.scan_data['step_size'][0], 2)} μm\n"
            #     f"Detector distance from center: {np.round(np.sqrt(self.detector_positions[example_position][0]**2 + self.detector_positions[example_position][1]**2) / self.sensor.magnification, 2)} μm",
            #     loc='center'
            # )
            
            # Labels
            axes[1].set_xlabel("x position [pix]", fontsize=20)
            axes[1].set_ylabel("y position [pix]", fontsize=20)
            
            # Make second plot square as well
            axes[1].set_aspect('equal')
            
            # Add colorbar with appropriate height
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(axes[1])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(ax1, cax=cax, label='Photon Count')
            
            # Display the figure
            plt.tight_layout()  # Adjust layout to prevent overlap

            plt.savefig("shift_vectors.png", dpi=300, bbox_inches='tight', format='png')

            plt.show()
        
        return shift_vectors
        
    def image_correlation(self, reference_image, target_image, max_shift=10):
        """
        Compute the shift between reference_image and target_image by directly
        shifting the target image across the reference image and finding the
        position with maximum correlation.
        
        Parameters:
        reference_image (ndarray): The reference image
        target_image (ndarray): The image to find shift for
        max_shift (int): Maximum pixel shift to test in each direction
        
        Returns:
        ndarray: [y_shift, x_shift] - the shift vector
        """
        # Get image dimensions
        h, w = reference_image.shape
        
        # Create zero-padded versions of both images to handle shifts
        pad_size = max_shift
        ref_padded = np.pad(reference_image, pad_size, mode='constant')
        target_padded = np.pad(target_image, pad_size, mode='constant')
        
        # Normalize images to zero mean and unit variance for better correlation
        ref_norm = (ref_padded - np.mean(reference_image)) / (np.std(reference_image) + 1e-10)
        target_norm = (target_padded - np.mean(target_image)) / (np.std(target_image) + 1e-10)
        
        # Initialize variables to track best correlation
        best_corr = -np.inf
        best_shift = np.array([0, 0])
        
        # Try different shifts and calculate correlation
        for y_shift in range(-max_shift, max_shift + 1):
            for x_shift in range(-max_shift, max_shift + 1):
                # Calculate the offset for target image
                y_offset = pad_size + y_shift
                x_offset = pad_size + x_shift
                
                # Extract the overlapping region from both images
                target_region = target_norm[pad_size:pad_size+h, pad_size:pad_size+w]
                ref_region = ref_norm[y_offset:y_offset+h, x_offset:x_offset+w]
                
                # Calculate correlation (Pearson correlation coefficient)
                # Flatten arrays for easier calculation
                ref_flat = ref_region.flatten()
                target_flat = target_region.flatten()
                
                # Skip if no valid overlap
                if ref_flat.size == 0:
                    continue
                
                # Calculate correlation coefficient
                corr = np.corrcoef(ref_flat, target_flat)[0, 1]
                
                # Handle NaN values which can occur if one image is constant
                if np.isnan(corr):
                    corr = -np.inf
                    
                # Update best correlation and shift
                if corr > best_corr:
                    best_corr = corr
                    best_shift = np.array([y_shift, x_shift])
        
        return best_shift

    
    def plot_ism_comparison(self, ism_results, emitters=None, fig=None, ax=None, crosssection=True):
        """
        Plot comparison between confocal and ISM images, with a third plot showing overlaid intensity profiles.
        
        Parameters
        ----------
        ism_results : dict
            Results dictionary from process_scan_data
        emitters : list, optional
            List of Emitter objects to overlay on the images
        fig : matplotlib.figure.Figure, optional
            Figure to use. If None, a new figure is created.
        ax : list of matplotlib.axes.Axes, optional
            List of axes to use. If None, new axes are created.
        crosssection : bool, optional
            If True, add a plot showing overlaid intensity profiles along x-axis through the middle of y-axis
            
        Returns
        -------
        tuple
            (fig, ax) - Figure and axes objects
        """
        # Create figure with 3 axes in a row
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw = {'width_ratios': [1, 1, 0.92], 'height_ratios': [1]})
        
        area_size = self.scan_data['area_size']
        confocal_image = ism_results['confocal_image']
        ism_image = ism_results['ism_image']
        
        # Plot confocal image
        im1 = ax[0].imshow(
            confocal_image,
            cmap='hot',
            extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2],
            origin='lower',
            interpolation='nearest'
        )
        ax[0].set_title('Conventional Image')
        ax[0].set_xlabel('x position (µm)')
        ax[0].set_ylabel('y position (µm)')
        plt.colorbar(im1, ax=ax[0], label='Photon Count', shrink=0.65)
        
        # Plot ISM image
        im2 = ax[1].imshow(
            ism_image,
            cmap='hot',
            extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2],
            origin='lower',
            interpolation='nearest'
        )
        ax[1].set_title('ISM Reconstruction')
        ax[1].set_xlabel('x position (µm)')
        ax[1].set_ylabel('y position (µm)')
        plt.colorbar(im2, ax=ax[1], label='Photon Count', shrink=0.65)
        
        # Plot emitter positions if provided
        if emitters is not None:
            for a in ax[:2]:  # Only for the first two plots
                x_positions = [emitter.x for emitter in emitters]
                y_positions = [emitter.y for emitter in emitters]
                a.scatter(x_positions, y_positions,
                        marker='x', color='red', s=100, alpha=0.8,
                        label='True emitter positions')
                a.legend(loc='upper right')
        
        # Add crosssection plot with both profiles overlaid
        if crosssection:
            # Get middle y-index
            mid_y_idx = confocal_image.shape[0] // 2
            
            # Get x positions for plotting
            x_positions_confocal = np.linspace(-area_size[0]/2, area_size[0]/2, confocal_image.shape[1])
            x_positions_ism = np.linspace(-area_size[0]/2, area_size[0]/2, ism_image.shape[1])
            
            # Plot normalized confocal crosssection
            confocal_profile = confocal_image[mid_y_idx,:]
            confocal_profile /= np.max(confocal_profile)  # Normalized
            ax[2].plot(x_positions_confocal, confocal_profile, 'b-', linewidth=2, label='Conventional')
            
            # Plot ISM crosssection on the same axes
            ism_profile = ism_image[mid_y_idx,:]
            ism_profile /= np.max(ism_profile)
            ax[2].plot(x_positions_ism, ism_profile, 'r-', linewidth=2, label='ISM')
            
            ax[2].set_title('Intensity Profiles Comparison')
            ax[2].set_xlabel('x position (µm)')
            ax[2].set_ylabel('Intensity')
            ax[2].grid(True, alpha=0.3)
            ax[2].legend(loc='upper right')
            ax[2].set_aspect(0.5)
            
            # Add markers for emitter positions in crosssection plot if provided
            if emitters is not None:
                for emitter in emitters:
                    ax[2].axvline(x=emitter.x, color='red', linestyle='--', alpha=0.5)
            # Compute the FWHM of the profiles
            fwhm_confocal = self.FWHM(confocal_profile, x_positions_confocal)
            fwhm_ism = self.FWHM(ism_profile, x_positions_ism)
            ax[2].text(-0.1, 0.2, f'FWHM (Conventional): {fwhm_confocal:.2f} μm')
            ax[2].text(-0.1, 0.24, f'FWHM (ISM): {fwhm_ism:.2f} μm')
        
        plt.tight_layout()

        # Compute the FWHM of the profiles


        return fig, ax
    
    def FWHM(self, intensity, x_values=None):
        """
        Calculate the Full Width at Half Maximum (FWHM) of a 1D intensity spectrum.
        
        Parameters:
        -----------
        intensity : array-like
            The intensity values of the spectrum
        x_values : array-like, optional
            The corresponding x-axis values. If None, indices will be used.
            
        Returns:
        --------
        fwhm : float
            The Full Width at Half Maximum
        """
        import numpy as np
        
        if x_values is None:
            x_values = np.arange(len(intensity))
        
        # Find the maximum intensity and calculate half maximum
        max_intensity = np.max(intensity)
        half_max = max_intensity / 2.0
        
        # Find indices where intensity is greater than half maximum
        above_half_max = intensity > half_max
        
        # Find first and last indices where intensity crosses half maximum
        if not np.any(above_half_max):
            return 0  # No values above half maximum
        
        # Find left and right crossing points
        left_idx = np.argmax(above_half_max)
        right_idx = len(above_half_max) - np.argmax(above_half_max[::-1]) - 1
        
        # Simple linear interpolation to find more precise crossing points
        if left_idx > 0:
            left_x = x_values[left_idx-1]
            left_y = intensity[left_idx-1]
            slope = (intensity[left_idx] - left_y) / (x_values[left_idx] - left_x)
            x_left = left_x + (half_max - left_y) / slope
        else:
            x_left = x_values[0]
        
        if right_idx < len(intensity)-1:
            right_x = x_values[right_idx]
            right_y = intensity[right_idx]
            slope = (intensity[right_idx+1] - right_y) / (x_values[right_idx+1] - right_x)
            x_right = right_x + (half_max - right_y) / slope
        else:
            x_right = x_values[-1]
        
        # Calculate FWHM
        fwhm = x_right - x_left
        return fwhm


    
    def visualize_detector_positions(self):
        """
        Visualize the detector positions used for ISM processing.
        
        Returns
        -------
        tuple
            (fig, ax) - Figure and axes objects
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot the detector positions
        ax.scatter(self.detector_positions[:, 0], self.detector_positions[:, 1],
                  s=100, alpha=0.7)
        
        # Add detector numbers for reference
        for i, (x, y) in enumerate(self.detector_positions):
            ax.text(x, y, str(i), fontsize=9, ha='center', va='center')
        
        # Add a circle showing the active area of each detector
        for x, y in self.detector_positions:
            circle = plt.Circle((x, y), self.sensor.pixel_radius, 
                               fill=False, color='r', linestyle='--', alpha=0.5)
            ax.add_patch(circle)
        
        ax.set_aspect('equal')
        ax.set_title('SPAD23 Detector Positions')
        ax.set_xlabel('x position (µm)')
        ax.set_ylabel('y position (µm)')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        return fig, ax