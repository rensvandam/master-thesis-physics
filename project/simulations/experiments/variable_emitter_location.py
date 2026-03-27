import numpy as np
import matplotlib.pyplot as plt

import project.model.coherence_from_data as coherence
from project.model.detection import show_photons, Spad23, Spad512, merge_photons
from project.model.plot_functions import show_emitter_positions
from project.model.sample import Alexa647

#25-3-2025
# Experiment: Simualate measurements with emitters on different locations, such that the PSF's have different overlaps.
# Plot a visualisation of the PSF's on the detector;
# Plot the estimated nr of emitters on y-axis and the % of the PSF of the second emitter that is on the detector on x-axis.


# 1. Over whole detector at once
def sim_whole_detector(positions, nr_emitters, laser, interval, bin_size, eta, seed, nr_steps=200, dashboard=False, debug=False):
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
    s = Spad23(magnification=150, nr_pixel_rows=5, pixel_radius=10.3)

    # if nr_emitters == 1:
    #     positions = np.array([[0, 0]])
    #     #positions = np.array([[-100, -100]])
    # elif nr_emitters == 2:
    #     positions = np.array([[0, 0], [0, 0]])
    # elif nr_emitters == 3:
    #     positions = np.array([[0, 0], [0, 0], [0, 0]])
    # elif nr_emitters == 4:
    #     positions = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])

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
    measured_timestamps = measurement[:, 1]
    # 4) Calculating coherence and expected number of emitters
    # print(measured_timestamps)
    # print(len(measured_timestamps))
    # Calculating the autocoherence over a neighbourhood
    estimated_emitters_array = []
    pixel_coherence, bins = coherence.auto_coherence(measured_timestamps, interval=interval, bin_size=bin_size, nr_steps=nr_steps, normalize=True)
        #estimated_nr_emitters = expected_number_of_emitters(pixel_autocoh, bin_size, 1, 1)
        #estimated_emitters_array.append(estimated_nr_emitters)
    #estimated_emitters_array = np.array(estimated_emitters_array)

    # Plot the photons
    # s.show()
    # show_emitter_positions(positions, s.x_limits[-1], s.y_limits[-1])
    # show_photons(photons, is_detected)

    # Fit the pixel coherence
    fit, popt, pcov = coherence.fit_coherence_function(bins, pixel_coherence, method='with_k', initial_guess=np.array([3, 2]))
    # Compute standard deviation of the fitted value for n
    sigma = np.sqrt(np.diag(pcov))[0]
    n = np.round(popt[0], 2)
    return n, sigma
    

#######################################################
## Simulation for 2 emitters with different locations: 
## 1 centered and one offset (in either x, y or diagonal direction)
#######################################################

# Generate positions: offset from 0 to 1 um.
positions_sets = []
offsets = np.linspace(0, 1, 10)
for offset in offsets:
      # X-direction:
      positions = np.array([[0, 0], [0, offset]])
      # Y-direction:
      # positions = np.array([[0, 0], [offset, 0]])
      # Diagnonal:
      # positions = np.array([[0, 0], [offset / np.sqrt(2), offset / np.sqrt(2)]])
      positions_sets.append(positions)

estimated_emitters = []
standard_deviations = []
for positions in positions_sets:
    measurements = []
    # Use 100 different seeds for each fraction. make seed dependent on position
    print(f"Position: {positions}")
    random = np.random.RandomState(42)
    randomnumber = random.randint(0, 100)
    for seed in range(randomnumber, randomnumber + 10):
        n, _ = sim_whole_detector(positions, nr_emitters=2, laser=330 * 10**3, 
                                   interval=10**5, bin_size=0.1, eta=1.0, 
                                   seed=seed, nr_steps=200)
        measurements.append(n)
    # Calculate mean and standard deviation of the 10 measurements
    mean_emitters = np.mean(measurements)
    std_emitters = np.std(measurements)
    estimated_emitters.append(mean_emitters)
    standard_deviations.append(std_emitters)

# Create error bar plot
plt.figure(figsize=(10, 6))
plt.errorbar(offsets, estimated_emitters, yerr=standard_deviations, fmt='o', capsize=5)
plt.xlabel("Offset [um]")
plt.ylabel("Estimated number of emitters")
plt.title("Emitter Estimation with second emitter offset from center")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

