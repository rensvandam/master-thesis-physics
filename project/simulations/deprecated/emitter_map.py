#%%

import numpy as np
import matplotlib.pyplot as plt

import project.model.plot_functions as plotting
import project.model.coherence_from_data as coherence

from project.model.sample import Alexa647
from project.model.detection import Sensor, Spad512, Spad23, merge_photons, show_photons
from project.model.emitter_density_map import get_map
from project.model.coherence_analytical import expected_number_of_emitters

def SPAD512_simulated_measurement(nr_emitters, laser, interval, bin_size, eta, seed, dashboard=False):
    """
    Simulates a measurement with a SPAD512 sensor
    """
    nr_pixel_rows = 16
    nr_pixel_columns = 16
    ######### INITIALIZATION ##########
    s = Spad512(magnification=1, nr_pixel_rows=nr_pixel_rows, nr_pixel_columns=nr_pixel_columns, pixel_radius=5)
    laser = 330 * 10 ** 3  # W / cm2
    interval = 10 ** 5  # ns
    eta = 1
    bin_size = 0.1
    positions = np.random.default_rng(seed).uniform(0, s.nr_pixel_rows * s.spacing, size=(nr_emitters, 2))

    ######### PIPELINE ##########

    # 1) Generating photons from the emitters
    emitters = []
    for j in range(nr_emitters):
        e = Alexa647(x=positions[j, 0], y=positions[j, 1])
        e.generate_photons(laser_power=laser, time_interval=interval, seed=j * seed, detection_efficiency=eta)
        emitters.append(e)

    # 2) #TODO: Translating sample plane to imaging plane 
    photons = s.magnify(merge_photons(emitters))

    # 3) Measuring the photons at the detector
    measurement, is_detected = s.measure(photons, duration=10_000, seed=seed)

    # 4) Calculating coherence and expected number of emitters
    coherence_array = coherence.auto_coherence_per_neighbourhood(s, interval, bin_size)
    estimated_nr_emitters = expected_number_of_emitters(coherence_array, bin_size, 1, 1)

    if dashboard:
        return s, emitters, photons, measurement, is_detected, estimated_nr_emitters

    ######## PLOTTING #########

    # Initialize plot
    plt.rcParams.update({'font.size': 14, 'figure.figsize': (7, 5.5)})  # 'figure.figsize': (7, 4.8)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # First subplot: Sensor intensity with emitter positions
    s.show(title = "Sensor Intensity & Emitter Positions", ax=axs[0])  # Pass axs[0] to s.show()
    plotting.show_emitter_positions(positions, s.x_limits[-1], s.y_limits[-1], ax=axs[0])

    # Second subplot: Estimated number of emitters from coherence
    s.show(data_to_show=get_map(estimated_nr_emitters, s, round_to_int=False).flatten(), title="Estimated number of emitters", ax=axs[1])

    plt.tight_layout()
    plt.savefig("./report/discussion/sensor_intensity2.svg")
    plt.show()
    s.clear()

def SPAD23_simulated_measurement(nr_emitters, laser, interval, bin_size, eta, seed, lag=0, dashboard=False, debug=False):
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
        "\nseed:", seed,
        "\nlag:", lag)

    ######### INITIALIZATION ##########
    s = Spad23(magnification=150, nr_pixel_rows=5, pixel_radius=10.3)
    # Setting random positions of the emitters
    width = s.nr_pixel_rows * s.spacing * 20
    height = s.nr_pixel_rows * s.spacing * np.sqrt(3)/2 * 20
    positions = np.random.default_rng(seed=seed).uniform(
        [-width/2, -height/2], [width/2, height/2], size=(nr_emitters, 2))
    
    if nr_emitters == 1:
        positions = np.array([[0, 0]])
    elif nr_emitters == 2:
        positions = np.array([[0, 0], [0, 0]])
    #positions = np.array([[50, 0]])
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
    measurement, is_detected = s.measure(photons=photons, duration=interval, seed=seed, debug=False)

    # 4) Calculating coherence and expected number of emitters

    # Calculating the autocoherence over a neighbourhood
    pixel_area_autocoh = coherence.auto_coherence_per_neighbourhood(s, interval, bin_size, kernel_size = 1)
    estimated_emitters_array = []
    for i in range(s.nr_pixels):
        estimated_nr_emitters = expected_number_of_emitters(pixel_area_autocoh[i], bin_size, 1, 1)
        estimated_emitters_array.append(estimated_nr_emitters)
    estimated_emitters_array = np.array(estimated_emitters_array)

    # Calculating the coherence matrix between pixels
    coherence_matrix = coherence.pixel_to_pixel_coherence(s, interval, bin_size, lag=lag, normalize=False)
    if dashboard:
        return s, emitters, photons, measurement, is_detected, estimated_emitters_array, coherence_matrix

    ######## PLOTTING #########

    # Showing the photon count per pixel

    # Initialize plot
    plt.rcParams.update({'font.size': 8, 'figure.figsize': (7, 5.5)})
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))

    # First subplot: Sensor intensity with emitter positions
    s.show(title=f"Emitted: {len(photons)}. On pixel: {np.sum(is_detected).astype(int)}."
                      f"Detected: {len(measurement)}.", ax=axs[0])
    # Showing the emitter positions
    plotting.show_emitter_positions(positions, s.x_limits[-1], s.y_limits[-1], ax=axs[0])
    plotting.show_pixel_numbers(s, ax=axs[0])
    # Second subplot: Photon hits
    s.show(title=f"Emitted: {len(photons)}. On pixel: {np.sum(is_detected).astype(int)}."
                      f"Detected: {len(measurement)}.", ax=axs[1])
    show_photons(photons, is_detected, ax=axs[1])

    # Third subplot: Estimated number of emitters from coherence
    s.show(data_to_show=estimated_emitters_array, title="Estimated number of emitters", ax=axs[2])

    # Fourth subplot: pixel to pixel coherence
    cax = axs[3].matshow(coherence_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    # Add colorbar
    fig.colorbar(cax)
    axs[3].set_xticks(range(23))
    axs[3].set_yticks(range(23))
    axs[3].set_xticklabels(range(1, 24))
    axs[3].set_yticklabels(range(1, 24))

    plotting.show_coherence_to_distance(s, lag, coherence_matrix, ax=axs[4])

    plt.tight_layout()
    plt.savefig("./report/discussion/sensor_intensity2.svg")
    plt.show()
    s.clear()

def main():
    #SPAD512_simulated_measurement(15)
    SPAD23_simulated_measurement(2, 330*10**3, 10**5, 0.1, 1, 65, lag = 0, dashboard=False, debug=True)


if __name__ == "__main__":
    main()
