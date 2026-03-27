import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import EllipseCollection
from matplotlib.cm import ScalarMappable
import time
from collections import defaultdict
from project.model.coherence_from_data import coherence, auto_coherence, fit_coherence_function, calculate_G2_difference, generate_est_nr_emitters_map
from project.model.ISMprocessor import ISMProcessor
from project.model.detection import show_photons
from project.model.sample import Alexa647
import datetime
import random

class Setup(ABC):
    """"
    Base class for an imaging setup.
    
    Attributes
    ----------
    sensor: Sensor
        The sensor used in the setup.
    magnification: float
        The magnification of the setup.
    illumination: str
        The type of illumination used in the setup ('scanning' or 'widefield').
    """

    def __init__(self, sensor, magnification, illumination_mode):
            self.sensor = sensor
            self.magnification = magnification
            self.illumination_mode = illumination_mode
    
    # @abstractmethod
    # def acquire_data(self):
    #     """Method to acquire data based on the imaging mode."""
    #     pass

class ScanningSetup(Setup):
    """
    Scanning setup where the sample moves across the SPAD23 sensor.

    Attributes
    ----------
    scan_speed : float
        Speed of the sample movement (µm/s).
    step_size : float
        Step size in a stepwise scanning approach (µm).
    dwell_time : float
        Time spent at each position during scanning (ms).
    ism_processor : ISMProcessor
        Optional processor for Image Scanning Microscopy.
    """

    def __init__(self, sensor, magnification, scan_speed, step_size, dwell_time):
        super().__init__(sensor, magnification, illumination_mode="scanning")
        self.scan_speed = scan_speed
        self.step_size = step_size
        self.dwell_time = dwell_time
        self.scan_data = {}  # Will store data for each scan position
        self.detector_data = {}  # Will store detailed detector data for ISM processing
        self.ism_results = None  # Will store ISM processing results if enabled
        self.ism_processor = None  # Will be initialized when needed     

    def scan_area(self, emitters, area_size=(4, 4), positions=(80, 80), laser_power=100, 
                 beam_waist=0.3, seed=1, detection_efficiency=1, photon_statistics="sub_poisson", 
                 enable_noise=True, calculate_G2=True, calculate_g2=True, G2_min_photon_count = 10, max_delay=1000,
                 enable_ism=False, alpha=0.5, upsampling_factor=2, plot_results=False):
        """
        Scan an area with the SPAD23 sensor and collect photon data at each position.
        Optionally perform ISM processing on the collected data.
        
        Parameters
        ----------
        emitters : list
            List of Emitter objects to be scanned.
        area_size : tuple
            Size of the area to scan in micrometers (width, height).
        positions : tuple
            Number of positions to scan (x_positions, y_positions).
        laser_power : float
            Laser power in W/cm².
        beam_waist : float
            Beam waist of the laser in micrometers.
        seed : int
            Random seed for reproducibility.
        detection_efficiency : float
            Detection efficiency of the sensor (0-1).
        photon_statistics : str
            Type of photon statistics: "sub_poisson", "poisson", or "super_poisson".
        enable_noise : bool
            Enable sensor noise effects (dark counts, jitter, etc.).
        calculate_G2 : bool
            Calculate G2(τ) for each scanning position.
        calculate_g2 : bool
            Calculate g2(τ) for each scanning position.
        max_delay : float
            Maximum delay for G2(τ) and g2(τ) calculation in nanoseconds.
        enable_ism : bool
            Whether to perform ISM processing on the scanning data.
        alpha : float
            ISM pixel reassignment parameter (typically 0.5).
        upsampling_factor : int
            Factor by which to increase resolution in final ISM image.
        plot_results : bool
            Whether to plot intermediate results during processing.
        
        Returns
        -------
        dict
            Dictionary with scan results including:
            - Basic scan data (positions, photon counts, g2 values)
            - Detector data for each pixel (if ISM enabled)
            - ISM results (if ISM enabled)
            - Analysis maps (photon_count_map, G2_diff_map, nr_emitters_map)
        """
        
        # Initialize ISM processor if needed and not already initialized
        if enable_ism and self.ism_processor is None:
            from project.model.ISMprocessor import ISMProcessor
            self.ism_processor = ISMProcessor(sensor=self.sensor, alpha=alpha)
        
        # Calculate step sizes based on area and number of positions
        x_step = area_size[0] / positions[0]
        y_step = area_size[1] / positions[1]
        
        # Calculate starting positions (center the scan area)
        start_x = -area_size[0]/2 + x_step/2
        start_y = -area_size[1]/2 + y_step/2
        
        # Initialize data structures
        position_data = []
        g2_data = {}
        G2_data = {}
        self.G2_min_photon_count = G2_min_photon_count
        photon_count_map = np.zeros(positions)
        detector_data = {} if enable_ism else None  # Only track detailed detector data if ISM is enabled
        
        # Convert dwell time from ms to ns
        dwell_time_ns = self.dwell_time * 1e6
        
        print(f"Starting scan of {area_size[0]}×{area_size[1]} µm² area with {positions[0]}×{positions[1]} positions")
        print(f"Step size: {x_step:.3f} × {y_step:.3f} µm")
        if enable_ism:
            print(f"ISM processing enabled with alpha={alpha}")
        
        scan_start_time = time.time()
        total_positions = positions[0] * positions[1]
        positions_scanned = 0
        
        # Loop through all scanning positions
        for iy in range(positions[1]):
            y_pos = start_y + iy * y_step
            
            for ix in range(positions[0]):
                x_pos = start_x + ix * x_step
                positions_scanned += 1
                
                if positions_scanned % 10 == 0 or positions_scanned == total_positions:
                    progress = positions_scanned / total_positions * 100
                    elapsed = time.time() - scan_start_time
                    est_total = elapsed / (positions_scanned / total_positions)
                    remaining = est_total - elapsed
                    print(f"Progress: {progress:.1f}% ({positions_scanned}/{total_positions}), "
                        f"Est. remaining time: {remaining:.1f}s")
                
                # Current scanning position
                current_position = (x_pos, y_pos)
                
                # Clear sensor data from previous position
                self.sensor.clear()
                
                all_photons = []
                for emitter in emitters:
                    # Calculate distance from laser center to emitter
                    distance = np.sqrt((emitter.x - x_pos)**2 + (emitter.y - y_pos)**2)
                    # Calculate local laser intensity using Gaussian profile
                    # I(r) = I₀·exp(-2r²/w₀²)
                    local_intensity = laser_power * np.exp(-2 * (distance**2) / (beam_waist**2))
                    
                    # Skip emitters that receive negligible illumination (optimization)
                    if local_intensity < laser_power * 0.001:  # less than 0.1% of max intensity
                        continue

                    # Give local_intensity some variability. This is very important!! If I don't implement this, the photons will hit at the same time at
                    # some position if there is a contribution from 2 or more fluorophores. This causes a huge spike in g2(0)/G2(0) and is unrealistic.
                    local_intensity *= random.uniform(0.9, 1.1)

                    # Create temporary emitter with effective position
                    temp_emitter = type(emitter)(emitter.x - x_pos, emitter.y - y_pos)
                    temp_emitter.absorption_wavelength = emitter.absorption_wavelength
                    temp_emitter.emission_wavelength = emitter.emission_wavelength
                    temp_emitter.lifetime = emitter.lifetime
                    temp_emitter.extinction_coefficient = emitter.extinction_coefficient
                    temp_emitter.sigma = emitter.sigma
                    
                    # Generate photons with local laser intensity
                    photons = temp_emitter.generate_photons(
                        laser_power=local_intensity,  # Use local intensity for this emitter
                        time_interval=dwell_time_ns,
                        seed=seed,
                        detection_efficiency=detection_efficiency,
                        statistics=photon_statistics,
                        widefield=False
                    )
                    
                    if len(photons) > 0:
                        all_photons.append(photons)
                
                # If photons were generated, combine them and process with sensor
                if all_photons:
                    combined_photons = np.vstack(all_photons)
                    # Sort photons by timestamp
                    combined_photons = combined_photons[np.argsort(combined_photons[:, 2])]
                    
                    # Apply magnification and measure with sensor
                    magnified_photons = self.sensor.magnify(combined_photons)
                    
                    if enable_noise:
                        # Full measurement with noise effects
                        detected_photons, is_detected = self.sensor.measure(
                            magnified_photons, 
                            duration=dwell_time_ns,
                            seed=seed + 10000 + ix*positions[1] + iy,
                        )
                    else:
                        # Simplified measurement without noise
                        detected_photons, is_detected = self.sensor.measure(
                            magnified_photons, 
                            duration=dwell_time_ns,
                            seed=seed + 10000 + ix*positions[1] + iy,
                            enable_dark_counts=False,
                            enable_timestamp_jitter=False,
                            enable_deadtime=False,
                            enable_afterpulsing=False,
                            enable_crosstalk=False
                        )
                    #print(f"Detected {len(detected_photons)} photons at position ({ix}, {iy}). Theoretical deadtime limit: {dwell_time_ns/self.sensor.dead_time}")
                    
                    if plot_results:
                        if (ix,iy) in []: #[(5,5), (2,5), (7,7), (3,3)]:
                            #plot the photons
                            self.sensor.show()
                            #print(magnified_photons[:4])
                            #print(detected_photons[:4])
                            show_photons(magnified_photons, is_detected)
                            import matplotlib.animation as animation
                            from matplotlib.colors import Normalize
                            from matplotlib.collections import EllipseCollection
                            from matplotlib.cm import ScalarMappable

                            def animate_spad_detections_on_sensor(sensor, photons, is_detected=None, batch_size=5, 
                                                                interval=100, save_path=None, 
                                                                title="SPAD Detector - Photon Detection Animation",
                                                                show_sensor_data=None):
                                """
                                Creates an animated plot showing photons appearing in batches on the SPAD sensor layout.
                                Uses the sensor's show() method to display the actual detector pixels as background.
                                
                                Parameters
                                ----------
                                sensor : Sensor object
                                    The SPAD sensor object with pixel layout and show() method.
                                photons : np.ndarray
                                    A (number of photons, 3) array containing the (x, y, t) coordinates of each photon, 
                                    sorted by time.
                                is_detected : np.ndarray, optional
                                    A (number of photons, ) array that indicates which photon was detected (1) and 
                                    which not (0). If None, all photons are shown in black.
                                batch_size : int, default=5
                                    Number of photons to show per frame.
                                interval : int, default=100
                                    Time between frames in milliseconds.
                                save_path : str, optional
                                    If provided, saves the animation as GIF or MP4 (based on file extension).
                                title : str
                                    Title for the animation plot.
                                show_sensor_data : np.ndarray, optional
                                    Data to display on sensor pixels (e.g., photon counts). If None, uses sensor.photon_count.
                                
                                Returns
                                -------
                                matplotlib.animation.FuncAnimation
                                    The animation object.
                                """
                                
                                # Sort photons by timestamp if not already sorted
                                if len(photons) > 0:
                                    sort_indices = np.argsort(photons[:, 2])
                                    photons_sorted = photons[sort_indices]
                                    if is_detected is not None:
                                        is_detected_sorted = is_detected[sort_indices]
                                    else:
                                        is_detected_sorted = None
                                else:
                                    photons_sorted = photons
                                    is_detected_sorted = is_detected
                                
                                # Calculate number of frames
                                n_frames = (len(photons_sorted) + batch_size - 1) // batch_size if len(photons_sorted) > 0 else 1
                                
                                # Set up the figure and axis
                                fig, ax = plt.subplots(figsize=(12, 10))
                                
                                # Prepare sensor data to display
                                if show_sensor_data is None:
                                    data_to_show = sensor.photon_count
                                else:
                                    data_to_show = show_sensor_data
                                
                                # Set up sensor background using sensor's show method logic
                                vmin = np.amin(data_to_show) if len(data_to_show) > 0 else 0
                                vmax = np.amax(data_to_show) if len(data_to_show) > 0 else 1
                                if vmin == vmax:
                                    vmin -= 1
                                    vmax += 1
                                
                                cmap = plt.get_cmap('viridis')
                                norm = Normalize(vmin, vmax, clip=True)
                                colors = norm(data_to_show)
                                
                                # Create sensor pixel display
                                offsets = list(zip(sensor.pixel_coordinates[:, 0], sensor.pixel_coordinates[:, 1]))
                                sensor_collection = EllipseCollection(
                                    widths=sensor.pixel_radius * 2, 
                                    heights=sensor.pixel_radius * 2, 
                                    angles=0,
                                    units='xy', 
                                    facecolors=cmap(colors), 
                                    offsets=offsets,
                                    transOffset=ax.transData
                                )
                                ax.add_collection(sensor_collection)
                                
                                # Set axis limits based on sensor
                                ax.set_xlim((sensor.x_limits[0] - sensor.spacing / 2, sensor.x_limits[1] + sensor.spacing / 2))
                                ax.set_ylim((sensor.y_limits[0] - sensor.spacing / 2, sensor.y_limits[1] + sensor.spacing / 2))
                                ax.set_aspect('equal', adjustable='box')
                                
                                # Add colorbar for sensor
                                cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
                                cbar.set_label('Photon Count', fontsize=12)
                                
                                # Labels and title
                                ax.set_xlabel(r"x [$\mu m]$", fontsize=14)
                                ax.set_ylabel(r"y [$\mu m]$", fontsize=14)
                                ax.set_title(title, fontsize=16, fontweight='bold')
                                
                                # Initialize empty scatter plot for photons
                                photon_scat = ax.scatter([], [], s=5, alpha=0.2, zorder=10)  # zorder=10 to show on top
                                
                                # Info text
                                info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=11,
                                                verticalalignment='top', bbox=dict(boxstyle='round', 
                                                facecolor='white', alpha=0.9), zorder=11)
                                
                                def animate(frame):
                                    if len(photons_sorted) == 0:
                                        return photon_scat, info_text
                                        
                                    # Calculate photons to show
                                    n_photons = min((frame + 1) * batch_size, len(photons_sorted))
                                    
                                    if n_photons == 0:
                                        return photon_scat, info_text
                                    
                                    current_photons = photons_sorted[:n_photons]
                                    
                                    # Determine colors for photons
                                    if is_detected_sorted is not None:
                                        current_detected = is_detected_sorted[:n_photons]
                                        photon_colors = ['red' if detected else 'black' for detected in current_detected]
                                        detected_count = np.sum(current_detected)
                                    else:
                                        photon_colors = ['black'] * n_photons
                                        detected_count = n_photons
                                    
                                    # Update photon scatter plot
                                    photon_scat.set_offsets(current_photons[:, :2])
                                    photon_scat.set_color(photon_colors)
                                    
                                    # Update info text
                                    current_time = current_photons[-1, 2] if n_photons > 0 else 0
                                    info_text.set_text(f'Time: {current_time:.1f} ns\n'
                                                    f'Detected: {detected_count}/{len(current_photons)}\n'
                                                    f'Batch: {frame + 1}/{n_frames}\n'
                                                    f'Total photons: {n_photons}')
                                    
                                    return photon_scat, info_text
                                
                                # Create animation
                                anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                                            interval=interval, blit=True, repeat=True)
                                
                                # Add legend for photons
                                if is_detected_sorted is not None:
                                    from matplotlib.patches import Patch
                                    legend_elements = [
                                        Patch(facecolor='red', label='Detected Photons'),
                                        Patch(facecolor='black', label='Undetected Photons')
                                    ]
                                    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.85))
                                
                                plt.tight_layout()
                                
                                # Save if requested
                                if save_path:
                                    print(f"Saving animation to {save_path}...")
                                    if save_path.lower().endswith('.gif'):
                                        anim.save(save_path, writer='pillow', fps=1000//interval)
                                    elif save_path.lower().endswith('.mp4'):
                                        anim.save(save_path, writer='ffmpeg', fps=1000//interval)
                                    else:
                                        print("Warning: Unsupported file format. Use .gif or .mp4")
                                
                                return anim


                            # Create sensor
                            sensor = self.sensor

                            # Create animation
                            anim = animate_spad_detections_on_sensor(
                                sensor, 
                                magnified_photons, 
                                is_detected, 
                                batch_size=500,
                                interval=150,
                                save_path="spad_sensor_demo.gif",
                                title="SPAD Sensor - Photon Detection Animation"
                            )
                            
                            plt.show()

                    # Store photon count for this position
                    photon_count_map[ix, iy] = len(detected_photons)

                    # For ISM, we need to store photon data by detector
                    if enable_ism:
                        # Create a dictionary to store photons detected by each detector
                        detector_photons = {}
                        for det_id in range(self.sensor.nr_pixels):
                            # Filter photons for this detector
                            det_photons = detected_photons[detected_photons[:, 0] == det_id] if len(detected_photons) > 0 else np.empty((0,1))
                            detector_photons[det_id] = det_photons
                        
                        # Store by scan position
                        detector_data[(ix, iy)] = detector_photons

                    # Calculate G2(τ) if requested
                    if calculate_G2 and len(detected_photons) > 0:
                        G2data, bins = auto_coherence(detected_photons[:,1],
                                                interval=dwell_time_ns, bin_size=0.1, nr_steps=800, normalize=False)
                        G2_data[(ix, iy)] = (G2data, bins)
                        #if (ix,iy) in [(5,5), (2,5)]:
                            #print g2 data
                            #print(f"G2 data for position ({ix}, {iy}): {G2data[:5]}")
                            #save detected photons as npy
                            #np.save(f"detected_photons_{ix}_{iy}.npy", detected_photons)
                    
                    # Calculate g2(τ) if requested
                    if calculate_g2 and len(detected_photons) > 0:
                        g2data, bins = auto_coherence(detected_photons[:,1],
                                                interval=dwell_time_ns, bin_size=0.1, nr_steps=800, normalize=True)
                        g2_data[(ix, iy)] = (g2data, bins)
                    
                # Store position data
                position_data.append({
                    'position': (ix, iy),
                    'coordinates': current_position,
                    'photon_count': photon_count_map[ix, iy] if len(all_photons) > 0 else 0
                })
        
        scan_time = time.time() - scan_start_time
        print(f"Scan completed in {scan_time:.2f} seconds")
        print(f"Total number of photons across image: {np.sum(photon_count_map)}")
        
        # Store results in the scan_data attribute
        self.scan_data = {
            'area_size': area_size,
            'positions': positions,
            'step_size': (x_step, y_step),
            'dwell_time': self.dwell_time,
            'photon_count_map': photon_count_map,
            'position_data': position_data,
            'detector_data': detector_data,
            'g2_data': g2_data,
            'G2_data': G2_data
        }
        
        # Store detector data for ISM
        if enable_ism:
            self.detector_data = detector_data
        
        # Calculate extra analysis maps
        # G2 difference map (antibunching metric)
        G2_diff_map = None
        if calculate_G2:
            G2_diff_map = self._make_G2_difference_map(G2_data, start_index=0)
            self.scan_data['G2_diff_map'] = G2_diff_map
        
        # Estimated number of emitters map
        nr_emitters_map = None
        sigma_emitters_map = None
        if calculate_g2:
            nr_emitters_map, sigma_emitters_map = generate_est_nr_emitters_map(
                self.scan_data,
                self.detector_data,
                min_photon_count=10,
                method='without_k',
                initial_guess=np.array([3, 2]),
                subtract_autocoherences=True,
                laser_power=laser_power,
                verbose=False)
            #_,_,nr_emitters_map, sigma_emitters_map = self.plot_est_nr_emitters(emitters=emitters, show_emitters=True, min_photon_count=10)
            self.scan_data['nr_emitters_map'] = nr_emitters_map
            self.scan_data['sigma_emitters_map'] = sigma_emitters_map
            print("NR EMITTERS MAP")
            print(nr_emitters_map)
        
        # Perform ISM processing if enabled
        if enable_ism and self.ism_processor is not None:
            print("Performing ISM processing...")
            ism_start_time = time.time()
            
            self.ism_results = self.ism_processor.process_scan_data_new(
                self.scan_data,
                self.detector_data,
                plot=plot_results
            )
            
            ism_time = time.time() - ism_start_time
            print(f"ISM processing completed in {ism_time:.2f} seconds")
            
            # Add ISM results to return data
            self.scan_data['ism_results'] = self.ism_results
        
        # Return combined results
        return self.scan_data
    
    def _calculate_g2(self, photon_data, max_delay):
        """
        Calculate g2(τ) (intensity correlation) from photon arrival times.
        
        Parameters
        ----------
        photon_data : np.ndarray
            Array of [pixel, timestamp] for each detected photon.
        max_delay : float
            Maximum delay to consider (ns).
            
        Returns
        -------
        tuple
            (delays, g2_values) - arrays of delays and corresponding g2 values.
        """
        # Extract timestamps and sort them
        timestamps = photon_data[:, 1]
        timestamps.sort()
        
        if len(timestamps) < 10:
            # Not enough photons for meaningful calculation
            return None
        
        # Create histogram of time differences
        delays = []
        for i in range(len(timestamps)):
            # Calculate delays with all other timestamps within max_delay
            for j in range(i+1, len(timestamps)):
                delay = timestamps[j] - timestamps[i]
                if delay > max_delay:
                    break
                delays.append(delay)
        
        if not delays:
            return None
            
        # Create histogram with 100 bins or fewer if few data points
        num_bins = min(100, len(delays) // 10)
        if num_bins < 5:
            num_bins = 5
            
        hist, bin_edges = np.histogram(delays, bins=num_bins, range=(0, max_delay))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Normalize to get g2(τ)
        # For proper normalization we need the total measurement time and count rate
        measurement_time = np.max(timestamps) - np.min(timestamps)
        count_rate = len(timestamps) / measurement_time if measurement_time > 0 else 0
        
        # Expected coincidences for uncorrelated photons
        expected_coincidences = count_rate**2 * measurement_time * (bin_edges[1] - bin_edges[0])
        
        # Normalize histogram
        g2_values = hist / expected_coincidences if expected_coincidences > 0 else hist
        
        return bin_centers, g2_values
    
    def _make_G2_difference_map(self, G2_data, start_index=0, tau_min=50, tau_max=None):
        # Create empty map for g2 difference
        diff_map = np.zeros(self.scan_data['positions']) + np.nan
        # Track calculation statistics
        calc_success = 0
        calc_failed = 0
        skipped_low_counts = 0
        for (ix, iy), (correlation_data,bins) in G2_data.items():
            # Skip positions with too few photon hits
            if self.scan_data['photon_count_map'][ix, iy] < self.G2_min_photon_count:
                skipped_low_counts += 1
                continue
                
            if correlation_data is not None:
                try:
                    #print(correlation_data)
                    # Skip positions with insufficient data points
                    if len(correlation_data) < 5 or np.all(np.isnan(correlation_data)):
                        #print(correlation_data)
                        print(f"Skipping position ({ix}, {iy}) due to insufficient data points.")
                        calc_failed += 1
                        continue
                    
                    # Calculate g2 difference using our function
                    result = calculate_G2_difference(correlation_data, 
                                                start_index=0, 
                                                tau_min=tau_min, 
                                                tau_max=tau_max)
                    
                    # Store result in map
                    diff_map[ix, iy] = result["difference"]
                    calc_success += 1
                        
                except Exception as e:
                    calc_failed += 1
                    # For debugging, uncomment the following line:
                    #print(f"Calculation failed at position ({ix}, {iy}): {e}")

        # Replace empty values with zeros
        diff_map[np.isnan(diff_map)] = 0
        
        print(f"Calculation summary: {calc_success} successful calculations, {calc_failed} failed calculations, "
            f"{skipped_low_counts} skipped due to low photon count (<{self.G2_min_photon_count})")
        
        return diff_map
    
    def plot_photon_count_map(self, emitters=None, cmap='viridis', fig=None, ax=None, 
                            show_emitters=True, emitter_marker='x', emitter_color='green', 
                            emitter_size=100, emitter_alpha=0.8):
        """
        Plot the photon count map from the scan with optional overlay of true emitter positions.
        
        Parameters
        ----------
        emitters : list, optional
            List of Emitter objects whose true positions will be plotted.
            If None, no emitter positions will be shown.
        cmap : str
            Colormap to use for the plot.
        fig : matplotlib.figure.Figure
            Figure object to use for plotting. If None, a new one is created.
        ax : matplotlib.axes.Axes
            Axes object to use for plotting. If None, a new one is created.
        show_emitters : bool
            Whether to show the emitter positions on the plot.
        emitter_marker : str
            Marker style for emitter positions.
        emitter_color : str
            Color for emitter markers.
        emitter_size : int
            Size of emitter markers.
        emitter_alpha : float
            Transparency of emitter markers.
            
        Returns
        -------
        tuple
            (fig, ax) - Figure and axes objects.
        """
        if not self.scan_data:
            print("No scan data available. Run scan_area first.")
            return None, None
            
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            
        photon_count_map = self.scan_data['photon_count_map']
        area_size = self.scan_data['area_size']


        plt.rcParams['font.size'] = 20          # Base font size
        plt.rcParams['axes.titlesize'] = 14     # Title font size
        plt.rcParams['axes.labelsize'] = 15     # X and Y label font size
        plt.rcParams['xtick.labelsize'] = 15    # X tick label font size
        plt.rcParams['ytick.labelsize'] = 15    # Y tick label font size
        plt.rcParams['legend.fontsize'] = 18    # Legend font size
        plt.rcParams['figure.titlesize'] = 16   # Figure title font size
                
        im = ax.imshow(
            photon_count_map,  # Transpose to match coordinate system
            cmap=cmap,
            extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2],
            origin='lower',
            interpolation='nearest'
        )

        # Plot true emitter positions if provided
        if emitters is not None and show_emitters:
            x_positions = [emitter.x for emitter in emitters]
            y_positions = [emitter.y for emitter in emitters]
            # Plot emitters with a distinct marker
            ax.scatter(x_positions, y_positions, 
                    marker=emitter_marker, 
                    color=emitter_color, 
                    s=emitter_size, 
                    alpha=emitter_alpha,
                    label='True emitter positions')
            
            # Add a legend
            ax.legend(loc='upper right')
        
        ax.set_xlabel('x position (µm)')
        ax.set_ylabel('y position (µm)')
        ax.set_title('Photon Count Map')
        plt.rcParams.update({'font.size': 30})        
        cbar = plt.colorbar(im, ax=ax, label='Photon Count', fraction=0.046, pad=0.04)

        #plt.colorbar(img, label="Emitters", fraction=0.046, pad=0.04)

        
        return fig, ax
    
    def plot_g2_map(self, delay_idx=1, cmap='viridis', fig=None, ax=None):
        """
        Plot the g2(τ) value map for a specific delay across all scan positions.
        
        Parameters
        ----------
        delay_idx : int
            Index of the delay to plot g2 values for.
        cmap : str
            Colormap to use for the plot.
        fig : matplotlib.figure.Figure
            Figure object to use for plotting. If None, a new one is created.
        ax : matplotlib.axes.Axes
            Axes object to use for plotting. If None, a new one is created.
            
        Returns
        -------
        tuple
            (fig, ax) - Figure and axes objects.
        """
        if not self.scan_data or 'g2_data' not in self.scan_data:
            print("No g2 data available. Run scan_area with calculate_g2=True first.")
            return None, None
            
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            
        positions = self.scan_data['positions']
        area_size = self.scan_data['area_size']
        g2_data = self.scan_data['g2_data']
        
        # Create an empty map for g2 values
        g2_map = np.zeros(positions) + np.nan
        
        # Fill in the map with g2 values at the selected delay
        for (ix, iy), g2_values in g2_data.items():
            if g2_values is not None:
                delays, values = g2_values
                if delay_idx < len(values):
                    g2_map[ix, iy] = values[delay_idx]
        
        im = ax.imshow(
            g2_map,  # Transpose to match coordinate system
            cmap=cmap,
            extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2],
            origin='lower',
            interpolation='nearest'
        )
        
        delay_value = None
        for g2_values in g2_data.values():
            if g2_values is not None:
                delays, values = g2_values
                if delay_idx < len(delays):
                    delay_value = delays[delay_idx]
                    break
        
        title = f'g2(τ) Map at τ = {delay_value:.2f} ns' if delay_value is not None else 'g2(τ) Map'
        ax.set_xlabel('x position (µm)')
        ax.set_ylabel('y position (µm)')
        ax.set_title(title)
        
        plt.colorbar(im, ax=ax, label='g2(τ)')
        
        return fig, ax
    
    def plot_g2_curves(self, positions=None, fig=None, ax=None, plot_fit=True, laser_power=None):
        """
        Plot g2(τ) curves for selected positions.
        
        Parameters
        ----------
        positions : list
            List of (x, y) tuples indicating positions to plot g2 curves for.
            If None, plots for up to 5 random positions with data.
        fig : matplotlib.figure.Figure
            Figure object to use for plotting. If None, a new one is created.
        ax : matplotlib.axes.Axes
            Axes object to use for plotting. If None, a new one is created.
            
        Returns
        -------
        tuple
            (fig, ax) - Figure and axes objects.
        """
        if not self.scan_data or 'g2_data' not in self.scan_data:
            print("No g2 data available. Run scan_area with calculate_g2=True first.")
            return None, None
            
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        g2_data = self.scan_data['g2_data']
        
        # If positions not specified, pick some random ones with data
        if positions is None:
            positions = []
            pos_with_data = [pos for pos, g2 in g2_data.items() if g2 is not None]
            if pos_with_data:
                # Take up to 5 random positions
                indices = np.random.choice(
                    len(pos_with_data), 
                    size=min(5, len(pos_with_data)), 
                    replace=False
                )
                positions = [pos_with_data[i] for i in indices]
        # Plot g2 curves for each position
        for pos in positions:
            if pos in g2_data and g2_data[pos] is not None:
                values, delays = g2_data[pos]
                ax.plot(delays[1:], values[1:], label=f'Position {pos}')
            else:
                ax.plot([], [], label=f'Position {pos} (No Data)', linestyle='--', color='gray')

            if plot_fit:
                # Apply fitting function
                fit, popt, pcov = fit_coherence_function(delays[1:], values[1:], 
                                                                method='without_k', 
                                                                initial_guess=np.array([3,2]),
                                                                example_emitter = Alexa647(x=0,y=0),
                                                                laser_power = laser_power,
                                                                debug=False
                )
                # Extract number of emitters and uncertainty
                n = np.round(popt[0], 2)
                sigma = np.sqrt(np.diag(pcov))[0]
                ax.plot(delays[1:], fit)
                print(f"For position {pos}, n: {n} and sigma: {sigma}")

            # Also plot a fit of the g2 curve.
            
        
        ax.set_xlabel('τ (ns)')
        ax.set_ylabel('g2(τ)')
        ax.set_title('Second-Order Correlation Function')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig, ax
    
    # Add method to display ISM results
    def plot_ism_results(self, emitters=None, fig=None, ax=None, crosssection=False):
        """
        Plot the results of ISM processing.
        
        Parameters
        ----------
        emitters : list, optional
            List of Emitter objects to overlay on the images
        fig : matplotlib.figure.Figure, optional
            Figure to use. If None, a new figure is created.
        ax : list of matplotlib.axes.Axes, optional
            List of axes to use. If None, new axes are created.
            
        Returns
        -------
        tuple
            (fig, ax) - Figure and axes objects
        """
        if not hasattr(self, 'ism_results'):
            print("No ISM results available. Run perform_ism_scan first.")
            return None, None
            
        return self.ism_processor.plot_ism_comparison(self.ism_results, emitters, fig, ax, crosssection=crosssection)
    

    def plot_est_nr_emitters(self, method='without_k', initial_guess=np.array([3, 2]), 
                         min_photon_count=100, cmap='viridis', fig=None, ax=None,
                         emitters=None, show_emitters=True, emitter_marker='x', 
                         emitter_color='red', emitter_size=100, emitter_alpha=0.8, verbose=False):
        """
        Plot a map of the estimated number of emitters based on fitting g2 curves.
        
        Parameters
        ----------
        method : str
            Method to use for fitting the coherence function ('with_k' or other methods 
            supported by fit_coherence_function).
        initial_guess : np.ndarray
            Initial guess for the fit parameters [n, k].
        min_photon_count : int
            Minimum number of photon hits required to attempt fitting the g2 curve.
        cmap : str
            Colormap to use for the plot.
        fig : matplotlib.figure.Figure
            Figure object to use for plotting. If None, a new one is created.
        ax : matplotlib.axes.Axes
            Axes object to use for plotting. If None, a new one is created.
        emitters : list, optional
            List of Emitter objects whose true positions will be plotted.
        show_emitters : bool
            Whether to show the emitter positions on the plot.
        emitter_marker : str
            Marker style for emitter positions.
        emitter_color : str
            Color for emitter markers.
        emitter_size : int
            Size of emitter markers.
        emitter_alpha : float
            Transparency of emitter markers.
            
        Returns
        -------
        tuple
            (fig, ax, n_map, sigma_map) - Figure, axes, and result maps.
        """
        # if not self.scan_data or 'g2_data' not in self.scan_data:
        #     print("No g2 data available. Run scan_area with calculate_g2=True first. Note: g2, not the unnormalized G2.")
        #     return None, None, None, None
            
        # if fig is None or ax is None:
        #     fig, ax = plt.subplots(figsize=(10, 8))
            
        # positions = self.scan_data['positions']
        # area_size = self.scan_data['area_size']
        # g2_data = self.scan_data['g2_data']
        # photon_count_map = self.scan_data['photon_count_map']
        
        # # Create empty maps for estimated number of emitters and fit uncertainty
        # n_map = np.zeros(positions) + np.nan
        # sigma_map = np.zeros(positions) + np.nan
        
        # # Track fitting statistics
        # fit_success = 0
        # fit_failed = 0
        # skipped_low_counts = 0
        
        # # Fit g2 curves for each position and extract number of emitters
        # for (ix, iy), g2_values in g2_data.items():
        #     # Skip positions with too few photon hits
        #     if photon_count_map[ix, iy] < min_photon_count:
        #         skipped_low_counts += 1
        #         continue
                
        #     if g2_values is not None:
        #         values, bins = g2_values
                
        #         try:
        #             # Skip positions with insufficient data points
        #             if len(values) < 5 or np.all(np.isnan(values)):
        #                 fit_failed += 1
        #                 continue
                    
        #             # Apply fitting function
        #             fit, popt, pcov = fit_coherence_function(bins, values, 
        #                                                             method=method, 
        #                                                             initial_guess=initial_guess,
        #                                                             example_emitter=self.sensor.example_emitter,
        #                                                             laser_power=self.sensor.laser_power)
                    
        #             # Extract number of emitters and uncertainty
        #             n = np.round(popt[0], 2)
        #             sigma = np.sqrt(np.diag(pcov))[0]
                    
        #             # Add sanity check for the fitted values
        #             if n > 0 and n < 100:
        #                 if sigma > n:  # Reasonable range
        #                     print(f"Fitting with high uncertainty at position ({ix}, {iy}): n={n}, sigma={sigma}.")
        #                 # Store results in maps
        #                 n_map[ix, iy] = n
        #                 sigma_map[ix, iy] = sigma
        #                 fit_success += 1
        #             else:
        #                 print(f"Fitting failed at position ({ix}, {iy}): n={n}, sigma={sigma}. Trying without first datapoint.")
        #                                         # Attempt to fit but without the first datapoint at zero lag
        #                 fit, popt, pcov = fit_coherence_function(bins[1:], values[1:],
        #                                                             method=method, 
        #                                                             initial_guess=initial_guess,
        #                                                             example_emitter=self.sensor.example_emitter,
        #                                                             laser_power=self.sensor.laser_power)
        #                 n = np.round(popt[0], 2)
        #                 sigma = np.sqrt(np.diag(pcov))[0]


        #                 # Add sanity check for the fitted values
        #                 if n > 0 and n < 100 and sigma < n:  # Reasonable range
        #                     print(f"Fitting succeeded at position ({ix}, {iy}) without first datapoint: n={n}, sigma={sigma}.")
        #                     # Store results in maps
        #                     n_map[ix, iy] = n
        #                     sigma_map[ix, iy] = sigma
        #                     fit_success += 1
        #                 else:
        #                     print(f"Fitting failed again without first datapoint at position ({ix}, {iy}): n={n}, sigma={sigma}")
        #                     fit_failed += 1
        #                 #fit_failed += 1
                        
        #         except Exception as e:
        #             try:
        #                 # Attempt to fit but without the first datapoint at zero lag
        #                 fit, popt, pcov = fit_coherence_function(bins[1:], values[1:],
        #                                                             method=method, 
        #                                                             initial_guess=initial_guess,
        #                                                             laser_power=self.sensor.laser_power)
        #                 n = np.round(popt[0], 2)
        #                 sigma = np.sqrt(np.diag(pcov))[0]

        #                 # Add sanity check for the fitted values
        #                 if n > 0 and n < 100 and sigma < n:  # Reasonable range
        #                     # Store results in maps
        #                     n_map[ix, iy] = n
        #                     sigma_map[ix, iy] = sigma
        #                     fit_success += 1
        #                 else:
        #                     print(f"Fitting failed again without first datapoint at position ({ix}, {iy}): n={n}, sigma={sigma}")
        #                     fit_failed += 1
        #             except:
        #                 fit_failed += 1
        #             # For debugging, uncomment the following line:
        #             # print(f"Fitting failed at position ({ix}, {iy}): {e}")

        # #Replace empty values with zeros
        # n_map[np.isnan(n_map)] = 0
        # sigma_map[np.isnan(sigma_map)] = 0

        area_size = self.scan_data['area_size']
        n_map, sigma_map = generate_est_nr_emitters_map(self.scan_data, self.detector_data, min_photon_count, method=method, initial_guess=initial_guess, laser_power = self.scan_data['laser_power'], verbose=verbose)
        
        # Plot the map of estimated number of emitters
        im = ax.imshow(
            n_map,  # Transpose to match coordinate system
            cmap=cmap,
            extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2],
            origin='lower',
            interpolation='nearest'
        )
        
        # Plot true emitter positions if provided
        if emitters is not None and show_emitters:
            x_positions = [emitter.x for emitter in emitters]
            y_positions = [emitter.y for emitter in emitters]
            
            # Plot emitters with a distinct marker
            ax.scatter(x_positions, y_positions, 
                    marker=emitter_marker, 
                    color=emitter_color, 
                    s=emitter_size, 
                    alpha=emitter_alpha,
                    label='True emitter positions')
            
            # Add a legend
            ax.legend(loc='upper right')
        
        ax.set_xlabel('x position (µm)')
        ax.set_ylabel('y position (µm)')
        ax.set_title(f'Estimated Number of Emitters (min. {min_photon_count} photons)')
        
        cbar = plt.colorbar(im, ax=ax, label='Number of Emitters')
        
        return fig, ax, n_map, sigma_map  # Return maps for further analysis
    
    def plot_G2_difference_map(self, tau_min=50, tau_max=None, 
                            min_photon_count=100, cmap='viridis', fig=None, ax=None,
                            emitters=None, show_emitters=True, emitter_marker='x', 
                            emitter_color='red', emitter_size=100, emitter_alpha=0.8):
        """
        Plot a map of the difference between G²(∞) and G²(0) across a scan area.
        
        Parameters
        ----------
        tau_min : int
            Minimum delay index to include in G²(∞) calculation.
        tau_max : int or None
            Maximum delay index to include in G²(∞) calculation. If None, uses full data length.
        min_photon_count : int
            Minimum number of photon hits required to calculate g2 difference.
        cmap : str
            Colormap to use for the plot.
        fig : matplotlib.figure.Figure
            Figure object to use for plotting. If None, a new one is created.
        ax : matplotlib.axes.Axes
            Axes object to use for plotting. If None, a new one is created.
        emitters : list, optional
            List of Emitter objects whose true positions will be plotted.
        show_emitters : bool
            Whether to show the emitter positions on the plot.
        emitter_marker : str
            Marker style for emitter positions.
        emitter_color : str
            Color for emitter markers.
        emitter_size : int
            Size of emitter markers.
        emitter_alpha : float
            Transparency of emitter markers.
            
        Returns
        -------
        tuple
            (fig, ax, diff_map) - Figure, axes, and g2 difference map.
        """
        if not self.scan_data or 'G2_data' not in self.scan_data:
            print("No G2 data available. Run scan_area with calculate_G2=True first. Note: G2, not the normalized g2.")
            return None, None, None
            
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            
        positions = self.scan_data['positions']
        area_size = self.scan_data['area_size']
        G2_data = self.scan_data['G2_data']
        photon_count_map = self.scan_data['photon_count_map']
        
        # Create empty map for g2 difference
        diff_map = np.zeros(positions) + np.nan
        
        # Calculate g2 difference for each position
        #print(g2_data)
        # g2data is a dicitonary with keys (0,3) and values (g2data, bins)
        
        # Calculate G2 difference map
        diff_map = self._make_G2_difference_map(G2_data, start_index=0, tau_min=tau_min, tau_max=tau_max)
        
        # Plot the map of G2 differences
        im = ax.imshow(
            diff_map,  # Transpose to match coordinate system
            cmap=cmap,
            extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2],
            origin='lower',
            interpolation='nearest'
        )
        
        # Plot true emitter positions if provided
        if emitters is not None and show_emitters:
            x_positions = [emitter.x for emitter in emitters]
            y_positions = [emitter.y for emitter in emitters]
            
            # Plot emitters with a distinct marker
            ax.scatter(x_positions, y_positions, 
                    marker=emitter_marker, 
                    color=emitter_color, 
                    s=emitter_size, 
                    alpha=emitter_alpha,
                    label='True emitter positions')
            
            # Add a legend
            ax.legend(loc='upper right')
        
        ax.set_xlabel('x position (µm)')
        ax.set_ylabel('y position (µm)')
        ax.set_title(f'G²(∞) - G²(0) Difference Map (min. {min_photon_count} photons)')
        
        cbar = plt.colorbar(im, ax=ax, label='G²(∞) - G²(0) Difference')
        
        return fig, ax, diff_map  # Return map for further analysis
    
    def plot_psf_comparison(self, ism_results, G2_diff_map, emitters=None, fig=None, ax=None, crosssection=True):
            """
            Plot comparison between confocal, ISM images, and G2 difference, with plots showing overlaid intensity profiles.
            
            Parameters
            ----------
            ism_results : dict
                Results dictionary from process_scan_data
            G2_diff_map : ndarray
                G2 difference map from scan_data
            emitters : list, optional
                List of Emitter objects to overlay on the images
            fig : matplotlib.figure.Figure, optional
                Figure to use. If None, a new figure is created.
            ax : list of matplotlib.axes.Axes, optional
                List of axes to use. If None, new axes are created.
            crosssection : bool, optional
                If True, add plots showing overlaid intensity profiles along x-axis through the middle of y-axis
                
            Returns
            -------
            tuple
                (fig, ax) - Figure and axes objects
            """
            # Create figure with 4 axes in a 2x2 grid
            if fig is None or ax is None:
                fig, ax = plt.subplots(2, 2, figsize=(16, 16))
                ax = ax.flatten()  # Flatten for easier indexing
            
            area_size = self.scan_data['area_size']
            confocal_image = ism_results['confocal_image']
            ism_image = ism_results['ism_image']
            G2_diff_map = G2_diff_map
            
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
            plt.colorbar(im1, ax=ax[0], label='Photon Count', shrink=0.85)
            
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
            plt.colorbar(im2, ax=ax[1], label='Photon Count', shrink=0.85)
            
            # Plot G2 difference map
            im3 = ax[2].imshow(
                G2_diff_map,
                cmap='viridis',
                extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2],
                origin='lower',
                interpolation='nearest'
            )
            ax[2].set_title('G2 Difference Map')
            ax[2].set_xlabel('x position (µm)')
            ax[2].set_ylabel('y position (µm)')
            plt.colorbar(im3, ax=ax[2], label='G2 Difference', shrink=0.85)
            
            # Plot emitter positions if provided
            if emitters is not None:
                for a in ax[:3]:  # For the first three plots (image plots)
                    x_positions = [emitter.x for emitter in emitters]
                    y_positions = [emitter.y for emitter in emitters]
                    a.scatter(x_positions, y_positions,
                            marker='x', color='red', s=100, alpha=0.8,
                            label='True emitter positions')
                    a.legend(loc='upper right')
            
            # Add crosssection plot with all profiles overlaid
            if crosssection:
                # Get middle y-index
                mid_y_idx = confocal_image.shape[0] // 2
                
                # Get x positions for plotting
                x_positions_confocal = np.linspace(-area_size[0]/2, area_size[0]/2, confocal_image.shape[1])
                x_positions_ism = np.linspace(-area_size[0]/2, area_size[0]/2, ism_image.shape[1])
                x_positions_g2 = np.linspace(-area_size[0]/2, area_size[0]/2, G2_diff_map.shape[1])
                
                # Plot normalized confocal crosssection
                confocal_profile = confocal_image[mid_y_idx,:]
                confocal_profile /= np.max(confocal_profile)  # Normalized
                ax[3].plot(x_positions_confocal, confocal_profile, 'b-', linewidth=2, label='Conventional')
                
                # Plot ISM crosssection on the same axes
                ism_profile = ism_image[mid_y_idx,:]
                ism_profile /= np.max(ism_profile)
                ax[3].plot(x_positions_ism, ism_profile, 'r-', linewidth=2, label='ISM')
                
                # Plot G2 difference crosssection
                g2_profile = G2_diff_map[mid_y_idx,:]
                g2_profile /= np.max(np.abs(g2_profile)) if np.max(np.abs(g2_profile)) > 0 else 1  # Normalized
                ax[3].plot(x_positions_g2, g2_profile, 'g-', linewidth=2, label='G2 Difference')
                
                ax[3].set_title('Intensity Profiles Comparison')
                ax[3].set_xlabel('x position (µm)')
                ax[3].set_ylabel('Normalized Intensity')
                ax[3].grid(True, alpha=0.3)
                ax[3].legend(loc='upper right')
                
                # Add markers for emitter positions in crosssection plot if provided
                if emitters is not None:
                    for emitter in emitters:
                        ax[3].axvline(x=emitter.x, color='red', linestyle='--', alpha=0.5)
                
                # Compute the FWHM of the profiles
                fwhm_confocal = self.ism_processor.FWHM(confocal_profile, x_positions_confocal)
                fwhm_ism = self.ism_processor.FWHM(ism_profile, x_positions_ism)
                try:
                    fwhm_g2 = self.ism_processor.FWHM(g2_profile, x_positions_g2)
                    g2_fwhm_text = f'FWHM (G2): {fwhm_g2:.2f} μm'
                except:
                    g2_fwhm_text = 'FWHM (G2): N/A'
                    
                ax[3].text(0.02, 0.15, f'FWHM (Conventional): {fwhm_confocal:.2f} μm', transform=ax[3].transAxes)
                ax[3].text(0.02, 0.10, f'FWHM (ISM): {fwhm_ism:.2f} μm', transform=ax[3].transAxes)
                ax[3].text(0.02, 0.05, g2_fwhm_text, transform=ax[3].transAxes)
            
            plt.tight_layout()
            return fig, ax

class WidefieldSetup(Setup):
    """
    Widefield setup where the entire field of view is imaged at once.

    Attributes
    ----------
    exposure_time : float
        Time for capturing an image (ns).
    """

    def __init__(self, sensor, magnification = 1, exposure_time=100000):
        super().__init__(sensor, magnification, illumination_mode="widefield")
        self.exposure_time = exposure_time

    def acquire_data(self):
        """Simulate acquiring widefield data."""
        print(f"Capturing widefield image with {self.exposure_time} ms exposure time.")
