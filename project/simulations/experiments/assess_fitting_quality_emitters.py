import numpy as np
import matplotlib.pyplot as plt

import project.model.coherence_from_data as coherence
from project.model.detection import show_photons, Spad23, Spad512, merge_photons
from project.model.plot_functions import show_emitter_positions
from project.model.sample import Alexa647

#25-3-2025
# Experiment: Simulate measurements with 1-5 emitters placed randomly in the detector FOV
# to test how well the fitting works for different numbers of emitters.
# Plot the estimated nr of emitters on y-axis and the actual number of emitters on x-axis.

np.random.seed(42)

# Set the global font to be DejaVu Sans, size 10 (or any other font you prefer)
plt.rcParams['font.family'] = 'serif'  # Options: 'serif', 'sans-serif', 'monospace'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman']
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']

# Font sizes for different elements
plt.rcParams['font.size'] = 15          # Base font size
plt.rcParams['axes.titlesize'] = 14     # Title font size
plt.rcParams['axes.labelsize'] = 15     # X and Y label font size
plt.rcParams['xtick.labelsize'] = 15    # X tick label font size
plt.rcParams['ytick.labelsize'] = 15    # Y tick label font size
plt.rcParams['legend.fontsize'] = 18    # Legend font size
plt.rcParams['figure.titlesize'] = 16   # Figure title font size

# Font weight
plt.rcParams['axes.labelweight'] = 'normal'  # Options: 'normal', 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# Figure and axes settings
plt.rcParams['figure.figsize'] = (8, 6)     # Default figure size (width, height)
plt.rcParams['figure.dpi'] = 100            # Figure resolution
plt.rcParams['savefig.dpi'] = 300           # Saved figure resolution (high quality for thesis)
plt.rcParams['savefig.format'] = 'pdf'      # Default save format (PDF is vector format, good for thesis)
plt.rcParams['savefig.bbox'] = 'tight'      # Remove extra whitespace when saving

# Line and marker settings
plt.rcParams['lines.linewidth'] = 2.0       # Default line width
plt.rcParams['lines.markersize'] = 6        # Default marker size
plt.rcParams['lines.markeredgewidth'] = 1.0 # Marker edge width

# Axes settings
plt.rcParams['axes.linewidth'] = 1.2        # Axes border line width
plt.rcParams['axes.spines.top'] = False     # Remove top spine
plt.rcParams['axes.spines.right'] = False   # Remove right spine
plt.rcParams['axes.grid'] = False            # Enable grid by default
plt.rcParams['grid.alpha'] = 0.3            # Grid transparency
plt.rcParams['grid.linewidth'] = 0.8        # Grid line width

# Tick settings
plt.rcParams['xtick.major.size'] = 5        # X major tick size
plt.rcParams['xtick.minor.size'] = 3        # X minor tick size
plt.rcParams['ytick.major.size'] = 5        # Y major tick size
plt.rcParams['ytick.minor.size'] = 3        # Y minor tick size
plt.rcParams['xtick.major.width'] = 1.2     # X major tick width
plt.rcParams['xtick.minor.width'] = 0.8     # X minor tick width
plt.rcParams['ytick.major.width'] = 1.2     # Y major tick width
plt.rcParams['ytick.minor.width'] = 0.8     # Y minor tick width
plt.rcParams['xtick.direction'] = 'in'      # Tick direction: 'in', 'out', 'inout'
plt.rcParams['ytick.direction'] = 'in'

# Legend settings
plt.rcParams['legend.frameon'] = True       # Legend frame
plt.rcParams['legend.framealpha'] = 0.9     # Legend frame transparency
plt.rcParams['legend.fancybox'] = True      # Rounded corners for legend
plt.rcParams['legend.numpoints'] = 1        # Number of points in legend for line plots

# LaTeX settings (optional - for high-quality mathematical expressions)
plt.rcParams['text.usetex'] = False         # Set to True if you have LaTeX installed
plt.rcParams['mathtext.default'] = 'regular'  # Math font style

# Color settings - you can define a custom color palette
thesis_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=thesis_colors)
# 1. Over whole detector at once
def sim_whole_detector(positions, nr_emitters, laser, interval, bin_size, eta, seed, nr_steps=200, dashboard=False, debug=False):
    """
    Simulates a measurement with a SPAD23 sensor.

    Parameters
    ----------
    positions : np.array
        Array of emitter positions (x, y coordinates).
    nr_emitters : int
        Number of emitters to simulate.
    laser : float
        Laser power in W/cm².
    interval : int
        Time interval in ns.
    bin_size : float
        Bin size for coherence calculation.
    eta : float
        Detection efficiency.
    seed : int
        Seed for random number generator.
    """
    if debug:
        print("Debugging SPAD23_simulated_measurement")
        print("Parameters:"
        "\nnr_emitters:", nr_emitters,
        "\nlaser:", laser,
        "\ninterval:", interval,
        "\nbin_size:", bin_size,
        "\neta:", eta,
        "\nseed:", seed)

    ######### INITIALIZATION ##########
    s = Spad23(magnification=120, nr_pixel_rows=5, pixel_radius=10.3)

    ######### PIPELINE ##########
    # 1) Generating photons from the emitters
    emitters = []
    for j in range(nr_emitters):
        e = Alexa647(x=positions[j, 0], y=positions[j, 1])
        e.generate_photons(laser_power=laser, time_interval=interval, seed=(j+1)*seed, detection_efficiency=eta, widefield=False)
        emitters.append(e)


    # 2) #TODO: Translating sample plane to imaging plane 
    photons = s.magnify(merge_photons(emitters), debug=False)

    # 3) Measuring the photons at the detector
    measurement, is_detected = s.measure(photons=photons, duration=interval, seed=seed, enable_dark_counts=False, enable_timestamp_jitter=False, enable_deadtime=False, enable_afterpulsing=False, enable_crosstalk=False, debug=False)
    
    #incl. noise
    #measurement, is_detected = s.measure(photons=photons, duration=interval, seed=seed, enable_dark_counts=True, enable_timestamp_jitter=True, enable_deadtime=True, enable_afterpulsing=True, enable_crosstalk=True, debug=False)

    measured_timestamps = measurement[:, 1]
    
    # 4) Calculating coherence and expected number of emitters
    pixel_coherence, bins = coherence.auto_coherence(measured_timestamps, interval=interval, bin_size=bin_size, nr_steps=nr_steps, normalize=False)
    pixel_coherence_nor, bins = coherence.auto_coherence(measured_timestamps, interval=interval, bin_size=bin_size, nr_steps=nr_steps, normalize=False)

    ###################################### test method ######################
    pixel_autocoherence_array = []
    for i in range(s.nr_pixels):
        #print(s.data_per_pixel[i])
        print(s.data_per_pixel[i])
        pixel_autocoherence, bins = coherence.auto_coherence(s.data_per_pixel[i], interval=interval, bin_size=bin_size, nr_steps=nr_steps, normalize=False)
        # Place zeros in all bins before the deadtime
        pixel_autocoherence_array.append(pixel_autocoherence)
    
    pixel_neighbourhood_coherence = pixel_coherence - np.sum(pixel_autocoherence_array, axis=0)
    # Normalize the result
    # Calculate the average value for the last 100 bins
    pixel_neighbourhood_coherence /= pixel_neighbourhood_coherence[-100:].mean(axis=0)
    #pixel_neighbourhood_coherence /= np.max(pixel_neighbourhood_coherence)

    pixel_coherence = pixel_neighbourhood_coherence
    ############################################################################


    #pixel_coherence /= np.max(pixel_coherence)
    bins = bins[1:]
    pixel_coherence_nor = pixel_coherence_nor[1:]
    pixel_neighbourhood_coherence = pixel_neighbourhood_coherence[1:]
    pixel_coherence = pixel_coherence[1:]
    plot_curve = False
    if plot_curve:
        plt.figure()
        plt.plot(bins, pixel_coherence_nor, color = 'b')# label='Pixel Coherence')
        #plt.plot(bins, pixel_neighbourhood_coherence, label='Pixel Autocoherence')
        #plt.plot(bins, np.sum(pixel_autocoherence_array, axis=0)[1:], color='r')
        plt.xlabel(r'$\ell$ [ns]', fontsize=25)
        plt.ylabel(r'$G^{(2)}[\ell]$', fontsize=25)
        #plt.ylim(0,
        #plt.title('Pixel Coherence vs Time')
        #plt.legend()
        plt.show()

    #pixel_coherence = pixel_neighbourhood_coherence

    # Plot the photons (optional - uncomment if needed)
    # s.show()
    # show_emitter_positions(positions, s.x_limits[-1], s.y_limits[-1])
    # show_photons(photons, is_detected)

    # Fit the pixel coherence
    fit, popt, pcov = coherence.fit_coherence_function(bins, pixel_coherence, method='without_k', initial_guess=np.array([3, 2]), example_emitter=emitters[0], laser_power = laser)
    # Compute standard deviation of the fitted value for n
    sigma = np.sqrt(np.diag(pcov))[0]
    n = np.round(popt[0], 2)
    print(n)
    return n, sigma


def generate_random_positions(nr_emitters, fov_limits, seed):
    """
    Generate random positions within the detector FOV.
    
    Parameters
    ----------
    nr_emitters : int
        Number of emitters to place.
    fov_limits : tuple
        (x_min, x_max, y_min, y_max) limits of the field of view.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    positions : np.array
        Array of (x, y) positions.
    """
    np.random.seed(seed)
    x_min, x_max, y_min, y_max = fov_limits
    
    positions = np.zeros((nr_emitters, 2))
    positions[:, 0] = np.random.uniform(x_min, x_max, nr_emitters)  # x coordinates
    positions[:, 1] = np.random.uniform(y_min, y_max, nr_emitters)  # y coordinates
    
    return positions


#######################################################
## Simulation for 1-5 emitters with random locations
#######################################################

# Define FOV limits (adjust based on your detector specifications)
# For SPAD23 with magnification 150 and pixel radius 10.3
fov_limits = (-0.5, 0.5, -0.5, 0.5)  # in micrometers

# Test different numbers of emitters
nr_emitters_list = [1,2,3,4,5,6,7,8,9,10]
nr_measurements_per_condition = 25  # Number of random configurations per emitter count

estimated_emitters_mean = []
estimated_emitters_std = []
actual_emitters = []

for nr_emitters in nr_emitters_list:
    print(f"Testing {nr_emitters} emitters...")
    
    measurements = []
    
    # Generate multiple random configurations
    for config_idx in range(nr_measurements_per_condition):
        # Generate random positions for this configuration
        positions = generate_random_positions(nr_emitters, fov_limits, seed=config_idx*10)
        print(positions)
        # Run multiple measurements with different seeds for each configuration
        config_measurements = []
        base_seed = config_idx * 100 + 100
        
        for seed_offset in range(1):  # 5 measurements per configuration
            seed = base_seed + seed_offset
            try:
                n, _ = sim_whole_detector(positions, nr_emitters=nr_emitters, 
                                        laser=10 * 10**3, interval=20 * 10**5, 
                                        bin_size=0.1, eta=1.0, seed=seed, 
                                        nr_steps=800)
                config_measurements.append(n)
            except Exception as e:
                print(f"Error in simulation for {nr_emitters} emitters, config {config_idx}, seed {seed}: {e}")
                continue
        
        # Average the measurements for this configuration
        if config_measurements:
            measurements.append(np.mean(config_measurements))
    
    # Calculate statistics across all configurations
    if measurements:
        mean_estimated = np.mean(measurements)
        std_estimated = np.std(measurements)
        
        estimated_emitters_mean.append(mean_estimated)
        estimated_emitters_std.append(std_estimated)
        actual_emitters.append(nr_emitters)
        
        print(f"  Actual: {nr_emitters}, Estimated: {mean_estimated:.2f} ± {std_estimated:.2f}")

# Create error bar plot
plt.figure(figsize=(10, 6))
plt.errorbar(actual_emitters, estimated_emitters_mean, yerr=estimated_emitters_std, 
             fmt='o', capsize=5, markersize=8, linewidth=2, color='blue')

# Add ideal line (y = x)
plt.plot([0, 6], [0, 6], 'r--', alpha=0.7, label='Ideal (y = x)')

plt.xlabel("Actual Number of Emitters")
plt.ylabel("Estimated Number of Emitters")
#plt.title("Emitter Estimation Performance for Random Positions")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.xlim(0.5, 5.5)
plt.ylim(0, max(estimated_emitters_mean) + max(estimated_emitters_std) + 0.5)

# Add text annotations showing the estimation accuracy
# for i, (actual, estimated, std) in enumerate(zip(actual_emitters, estimated_emitters_mean, estimated_emitters_std)):
#     accuracy = abs(estimated - actual) / actual * 100
#     plt.annotate(f'{accuracy:.1f}% error', 
#                 xy=(actual, estimated), 
#                 xytext=(5, 5), 
#                 textcoords='offset points',
#                 fontsize=9, alpha=0.7)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
for actual, estimated, std in zip(actual_emitters, estimated_emitters_mean, estimated_emitters_std):
    error = abs(estimated - actual) / actual * 100
    print(f"Actual: {actual} | Estimated: {estimated:.2f} ± {std:.2f} | Error: {error:.1f}%")