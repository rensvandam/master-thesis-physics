import numpy as np
import matplotlib.pyplot as plt

import project.model.coherence_from_data as coherence
from project.model.detection import show_photons, Spad23, Spad512, merge_photons
from project.model.sample import Alexa647



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

#25-3-2025
# Experiment: Simualate measurements with 1 emitter for different deadtimes and intensities
# Plot the incoming photons on the x axis, the detected photons on the y axis for different deadtimes.


# 1. Over whole detector at once
def sim_whole_detector(fraction, nr_emitters, laser, interval, bin_size, eta, seed, deadtime=50, nr_steps=200, dashboard=False, debug=False):
    """
    Simulates a measurement with a SPAD23 sensor.

    Parameters
    ----------
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
    s = Spad23(magnification=150, nr_pixel_rows=5, pixel_radius=10.3, dead_time = deadtime)

    if nr_emitters == 1:
        positions = np.array([[0, 0]])
        #positions = np.array([[-100, -100]])
    elif nr_emitters == 2:
        positions = np.array([[0, 0], [0, 0]])
    elif nr_emitters == 3:
        positions = np.array([[0, 0], [0, 0], [0, 0]])
    elif nr_emitters == 4:
        positions = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])

    ######### PIPELINE ##########
    # 1) Generating photons from the emitters
    emitters = []
    for j in range(nr_emitters):
        e = Alexa647(x=positions[j, 0], y=positions[j, 1])
        current_laser_power = laser*(1-fraction) if j == 0 else laser
        #print("Laser power: ", current_laser_power)
        e.generate_photons(laser_power=current_laser_power, time_interval=interval, seed=(j+1)*seed, detection_efficiency=eta, widefield=False)
        emitters.append(e)

    # 2) #TODO: Translating sample plane to imaging plane 
    photons = s.magnify(merge_photons(emitters), debug=False)

    # 3) Measuring the photons at the detector
    measurement, detected = s.measure(photons=photons, duration=interval, enable_dark_counts=False, enable_timestamp_jitter=False, enable_deadtime=True, enable_afterpulsing=False, enable_crosstalk=False, seed=seed, debug=False)
    #measurement, detected = s.measure(photons=photons, duration=interval, enable_dark_counts=False, enable_timestamp_jitter=False, enable_deadtime=True, enable_afterpulsing=False, enable_crosstalk=False, seed=seed, debug=False)

    measured_timestamps = measurement[:, 1]
    # 4) Calculating coherence and expected number of emitters
    #print(measured_timestamps)
    #print(len(measured_timestamps))
    # Calculating the autocoherence over a neighbourhood
    estimated_emitters_array = []
    pixel_coherence, bins = coherence.auto_coherence(measured_timestamps, interval=interval, bin_size=bin_size, nr_steps=nr_steps, normalize=True)
        #estimated_nr_emitters = expected_number_of_emitters(pixel_autocoh, bin_size, 1, 1)
        #estimated_emitters_array.append(estimated_nr_emitters)
    #estimated_emitters_array = np.array(estimated_emitters_array)

    # Fit the pixel coherence
    #fit, popt, pcov = coherence.fit_coherence_function(bins, pixel_coherence, method='without_k', initial_guess=np.array([3]))
    # Compute standard deviation of the fitted value for n
    #sigma = np.sqrt(np.diag(pcov))[0]
    #n = np.round(popt[0], 2)

    # Incoming photons on pixels
    detected = len(detected)
    # Measured
    measured = len(measurement)

    return detected, measured
    

#######################################################
## Simulation for 2 emitters with different intensities
#######################################################

# deadtimes = np.linspace(0, 100, 19)
# detected_photon_intensities = []
# incoming_photon_intensities = []
# for deadtime in deadtimes:
#     print("Deadtime: ", deadtime)
#     # Collect 10 measurements with different seeds
#     measurements_detected = []
#     measurements_incoming = []
#     # Use 100 different seeds for each fraction
#     for seed in range(10):
#         incoming,detected = sim_whole_detector(fraction=0,nr_emitters=1, laser=330 * 10**3, 
#                                    interval=10**5, bin_size=0.1, eta=1.0, 
#                                    seed=seed, deadtime=deadtime, nr_steps=200)
#         print(incoming,detected)
#         measurements_detected.append(detected)
#         measurements_incoming.append(incoming)
#     # Calculate mean and standard deviation of the 10 measurements
#     incoming_mean = np.mean(measurements_incoming)
#     detected_mean = np.mean(measurements_detected)
#     detected_photon_intensities.append(detected_mean)
#     incoming_photon_intensities.append(incoming_mean)

# # Plot
# plt.figure(figsize=(10, 6))
# plt.plot(deadtimes, detected_photon_intensities, label='Detected photons', marker='o')
# plt.plot(deadtimes, incoming_photon_intensities, label='Incoming photons', marker='o')
# plt.xlabel("Deadtime [ns]")
# plt.ylabel("Number of photons")
# plt.title("Deadtime Effect on Photon Detection")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()

#######################################################
## Simulation for 3 emitters, 2 with same intensity 
## and 1 with different intensity
#######################################################

# fractions = np.linspace(0, 1, 9)
# estimated_emitters = []
# standard_deviations = []
# for fraction in fractions:
#     # Collect 10 measurements with different seeds
#     measurements = []
#     # Use 100 different seeds for each fraction
#     for seed in range(10):
#         n, _ = sim_whole_detector(fraction, nr_emitters=3, laser=330 * 10**3, 
#                                    interval=10**5, bin_size=0.1, eta=1.0, 
#                                    seed=seed, nr_steps=200)
#         measurements.append(n)
#     # Calculate mean and standard deviation of the 10 measurements
#     mean_emitters = np.mean(measurements)
#     std_emitters = np.std(measurements)
#     estimated_emitters.append(mean_emitters)
#     standard_deviations.append(std_emitters)

# # Create error bar plot
# plt.figure(figsize=(10, 6))
# plt.errorbar(fractions, estimated_emitters, yerr=standard_deviations, fmt='o', capsize=5)
# plt.xlabel("Intensity difference")
# plt.ylabel("Estimated number of emitters")
# plt.title("Emitter Estimation For different Emitter Intensties")
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()

###############################################
###############################################
###############################################
from tqdm import tqdm
# Extended simulation with varying laser intensity and deadtime
def run_comprehensive_simulation():
    # Parameters
    deadtimes = np.linspace(0, 100, 9)  # 10 deadtime values from 0 to 100 ns
    laser_intensities = np.linspace(5 * 10**3, 300 * 10**3, 8)  # 8 laser intensities from 5k to 300k
    laser_intensities = [5, 10, 20, 40, 60, 80, 100]
    laser_intensities = [intensity * 10**3 for intensity in laser_intensities]  # Convert to photons/s
    seeds_per_point = 5  # Number of random seeds per parameter combination
    
    # Results storage
    results = []
    
    # Run simulations for all combinations
    total_combinations = len(deadtimes) * len(laser_intensities)
    
    print(f"Running simulations for {len(deadtimes)} deadtimes × {len(laser_intensities)} intensities...")
    
    # For each deadtime
    for deadtime in tqdm(deadtimes, desc="Deadtimes"):
        # For each laser intensity
        for laser in laser_intensities:
            # Collect measurements with different seeds
            measurements_detected = []
            measurements_incoming = []
            
            for seed in range(seeds_per_point):
                incoming, detected = sim_whole_detector(
                    fraction=0, 
                    nr_emitters=1, 
                    laser=laser,
                    interval=2*10**6, 
                    bin_size=0.1, 
                    eta=1.0,
                    seed=seed, 
                    deadtime=deadtime, 
                    nr_steps=200
                )
                measurements_detected.append(detected)
                measurements_incoming.append(incoming)
            
            # Calculate mean of the measurements
            incoming_mean = np.mean(measurements_incoming)
            detected_mean = np.mean(measurements_detected)
            
            # Store results
            results.append({
                'deadtime': deadtime,
                'laser': laser,
                'incoming': incoming_mean,
                'detected': detected_mean,
                'efficiency': detected_mean / incoming_mean if incoming_mean > 0 else 0
            })
    
    return results

# Run the simulation
results = run_comprehensive_simulation()

# Organize results by deadtime for plotting
deadtime_groups = {}
for result in results:
    deadtime = result['deadtime']
    if deadtime not in deadtime_groups:
        deadtime_groups[deadtime] = []
    deadtime_groups[deadtime].append(result)

# Create the comprehensive plot
plt.figure(figsize=(12, 10))

# Create the 100% efficiency diagonal line (ideal detector)
max_value = max([r['incoming'] for r in results]) * 1.1
diagonal_line = np.linspace(0, max_value, 100)
#plt.plot(diagonal_line, diagonal_line, 'k--', alpha=0.7, label='100% Efficiency (Ideal)')

# Create a custom colormap for deadtime values
colors = plt.cm.viridis(np.linspace(0, 1, len(deadtime_groups)))

# Plot each deadtime as a separate curve
for i, (deadtime, group) in enumerate(sorted(deadtime_groups.items())):
    # Sort by incoming photon count
    group.sort(key=lambda x: x['incoming'])
    
    # Extract data for this deadtime
    incoming = [r['incoming'] for r in group]
    detected = [r['detected'] for r in group]
    
    # Plot this deadtime curve
    plt.plot(incoming, detected, 'o-', color=colors[i], 
             markersize=6, linewidth=2, 
             label=f'τ = {deadtime:.1f} ns')

# Calculate theoretical curves for non-paralyzable model
# x_range = np.linspace(0, max_value, 500)
# for deadtime in sorted(deadtime_groups.keys()):
#     if deadtime > 0:
#         # Convert ns to seconds
#         tau = deadtime * 1e-9
#         y_nonpara = x_range / (1 + x_range * tau)
#         plt.plot(x_range, y_nonpara, ':', color='gray', alpha=0.3)

# Add efficiency lines
# efficiencies = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
# for eff in efficiencies:
#     x_vals = np.linspace(0, max_value, 100)
#     y_vals = x_vals * eff
#     plt.plot(x_vals, y_vals, 'k:', alpha=0.3)
#     plt.text(max_value * 0.9, max_value * 0.9 * eff, f"{eff*100:.0f}%", 
#              fontsize=8, alpha=0.5)

# Set labels and title
plt.xlabel('Incoming Photons (counts)', fontsize=20)
plt.ylabel('Detected Photons (counts)', fontsize=20)
#plt.title('Detector Response with Varying Deadtime and Laser Intensity', fontsize=14)

# Add legend with smaller font and more columns
plt.legend(loc='upper left', fontsize=18, ncol=2)

# Add grid
#plt.grid(True, linestyle='--', alpha=0.5)

# Add text explanation
# plt.figtext(0.25, 0.15, 
#             "Deadtime Plot with Variable Laser Intensity:\n" +
#             "• Each curve represents a different deadtime value\n" +
#             "• Along each curve, laser intensity increases\n" +
#             #"• Dotted lines show theoretical non-paralyzable model\n" +
#             "• As deadtime increases, saturation occurs at lower intensities",
#             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.savefig('deadtime_laser_intensity_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a 3D surface plot for better visualization
from mpl_toolkits.mplot3d import Axes3D

# Organize data for 3D plotting
deadtime_values = sorted(list(deadtime_groups.keys()))
laser_values = sorted(list(set([r['laser'] for r in results])))

# Create a 2D grid for the surface
X, Y = np.meshgrid(laser_values, deadtime_values)
Z = np.zeros_like(X)

# Fill in Z values (efficiencies)
for i, deadtime in enumerate(deadtime_values):
    for j, laser in enumerate(laser_values):
        # Find the matching result
        for result in results:
            if result['deadtime'] == deadtime and result['laser'] == laser:
                Z[i, j] = result['efficiency'] * 100  # Convert to percentage
                break

# Create 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

# Add colorbar
cbar = fig.colorbar(surf)
cbar.set_label('Detection Efficiency (%)')

# Set labels
ax.set_xlabel('Laser Intensity (photons/s)')
ax.set_ylabel('Deadtime (ns)')
ax.set_zlabel('Detection Efficiency (%)')
ax.set_title('Detection Efficiency as a Function of Deadtime and Laser Intensity')

# Improve axis scaling
ax.set_box_aspect((2, 1, 1))

plt.tight_layout()
plt.savefig('deadtime_efficiency_3d_surface.png', dpi=300, bbox_inches='tight')
plt.show()