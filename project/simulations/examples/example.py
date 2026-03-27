import numpy as np
import matplotlib.pyplot as plt

from project.model.coherence_from_data import auto_coherence, show_coherence, coherence
from project.model.detection import show_photons, Spad23, Spad512, merge_photons
from project.model.sample import Alexa647


def basic_simulation_spad23():
    # All parameters are defined by default. You can change one or multiple arguments if you want. Check out the class
    # Spad23.
    sensor = Spad23(magnification=150)

    measurement_time = 10**5  # ns
    N = 2
    emitter_list = []
    for i in range(N):
        emitter = Alexa647(x=i * 100, y=i * 50)  # An Alexa647 fluorophore at a certain coordinate.
        emitter.generate_photons(laser_power=330 * 10 ** 3, time_interval=measurement_time, widefield=False, seed=6 * i)
        emitter_list.append(emitter)
    photons = merge_photons(emitter_list)

    # Magnify the signal
    photons = sensor.magnify(photons)

    # Do the measurement
    measurement, is_detected = sensor.measure(photons=photons, duration=measurement_time, seed=52)

    # Show the results
    sensor.show(title=f"Emitted: {len(photons)}. On pixel: {np.sum(is_detected).astype(int)}."
                      f"Detected: {len(measurement)}.")
    show_photons(photons, is_detected)
    plt.show()

    # Now we have the measurement data. Let's analyze it by calculating the coherence.
    # The measurement can be accessed through: sensor.data_by_time, sensor.data_per_pixel and sensor.photon_count.
    measured_timestamps = measurement[:, 1]
    coherence_vals, bins = auto_coherence(measured_timestamps, interval=measurement_time, bin_size=0.1, nr_steps=200, normalize=True)
    show_coherence(bins, coherence_vals, show_fit=False, auto_scale=True)

    # We can also calculate the coherence between different pixels.
    # For example, the coherence between pixel 10 and 11.
    coherence_vals, bins = coherence(sensor.data_per_pixel[10], sensor.data_per_pixel[11], interval=measurement_time,
                                     bin_size=0.5, nr_steps=200, normalize=True)
    show_coherence(bins, coherence_vals, show_fit=False, auto_scale=True)

    # Don't forget to clean the sensor if you want to re-use it for a new measurement.
    sensor.clear()


def basic_simulation_spad512():
    measurement_time = 10**2

    # For this example, we take a 10x10 spad array.
    sensor = Spad512(nr_pixel_rows=10, pixel_radius=5, spacing=20, magnification=50)

    # TODO: there needs to be a Setup class or something like that, where the translation between sample coordinates and
    #  Sensor coordinates should be done. Also, the scanning procedure for the Spad23 could be implemented there for
    #  example. Now, the emitter positions are given in the Sensor coordinate frame.
    emitter_1 = Alexa647(x=91, y=89.3)
    # This is a widefield setup, so widefield=True.
    emitter_1.generate_photons(laser_power=330 * 10 ** 3, time_interval=measurement_time, widefield=True, seed=55)

    emitter_2 = Alexa647(x=89.6, y=90.6)
    emitter_2.generate_photons(330 * 10 ** 3, measurement_time, widefield=True, seed=56)

    photons = merge_photons([emitter_1, emitter_2])

    photons = sensor.magnify(photons)
    measurement, is_detected = sensor.measure(photons, seed=10, duration=measurement_time)
    sensor.show()
    show_photons(photons, is_detected)
    plt.show()


def optimal_magnification():
    # Set the global font to be DejaVu Sans, size 10 (or any other font you prefer)
    plt.rcParams['font.family'] = 'serif'  # Options: 'serif', 'sans-serif', 'monospace'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman']
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']

    # Font sizes for different elements
    plt.rcParams['font.size'] = 20          # Base font size
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
    thesis_colors = ["#1f77b4", '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    thesis_colors = ['blue', 'red', 'green', 'purple']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=thesis_colors)


    dtime_data = []
    for deadtime in [5,20,35,50]:
        measurement_time = 1000000
        sensor = Spad23(dead_time=0)
        emitter = Alexa647(0, 0)
        photons = emitter.generate_photons(10 * 10 ** 3, measurement_time, seed=5, widefield=False)

        nr_steps = 150
        #magnification = (np.arange(nr_steps) + 1) * 2
        #magnification[0] = 1
        # magnification ranges from 10 to 300
        magnification = np.linspace(30, 300, nr_steps)
        percentage_geometry = np.zeros(nr_steps)
        percentage_dead_time = np.zeros(nr_steps)

        for i in range(nr_steps):
            sensor.magnification = magnification[i]
            current_photons = sensor.magnify(photons)
            measurement, is_detected = sensor.measure(current_photons, duration=measurement_time, seed=2)
            percentage_geometry[i] = np.sum(is_detected) / len(photons)
            percentage_dead_time[i] = len(measurement) / np.sum(is_detected)

            # if i == 0 or i % 25 == 24:
            #     sensor.show(title=f"M: {magnification[i]}. Emitted: {len(photons)}. "
            #                       f"On pixel: {np.sum(is_detected).astype(int)}. Detected: {len(measurement)}.")
            #     show_photons(current_photons, is_detected)
            #     plt.show()

            #     print(f"Number of photons detected: {len(measurement)}")

            # If magnification is 120
            # if magnification[i] == 120:
            #     print(f"Special case for magnification 120: {len(measurement)} photons detected.")
            # if magnification[i] == 112:
            #     print(f"Special case for magnification 112: {len(measurement)} photons detected.")

            sensor.clear()

        print(magnification[np.argmax(percentage_dead_time * percentage_geometry)])
        print(np.amax(percentage_geometry * percentage_dead_time))
        print(magnification[np.argmax(percentage_dead_time[35:] * percentage_geometry[35:])])
        print(np.amax(percentage_geometry[35:] * percentage_dead_time[35:]))

        NA=1
        pixel_distance = 23 #micrometers
        wavelength = 0.64 #micrometers
        sampling_factor = (pixel_distance/magnification)/(wavelength/(4*NA))
        print("Sampling max")
        print(np.max(sampling_factor))
        print(np.argmax(sampling_factor))
        #dtime_data.append((magnification, percentage_geometry, percentage_dead_time))

        # print the sampling factor for which the percentage_dead_time*percentage_geometry is the highest
        print(f"HIGHEST FOR DEADTIME {deadtime} ns")
        print(sampling_factor[np.argmax(percentage_dead_time * percentage_geometry)])
        print("highest up to index 80")
        arg = np.argmax(percentage_dead_time[25:] * percentage_geometry[25:])
        print(arg)
        print(sampling_factor[np.argmax(percentage_dead_time[(arg+25):] * percentage_geometry[(arg+25):])])
        print(sampling_factor)
        dtime_data.append((sampling_factor, percentage_geometry, percentage_dead_time))
    #plt.plot(magnification, percentage_geometry, label="Emission -> Geometry")
    #plt.plot(magnification, percentage_dead_time, label="Geometry -> Dead time")
    #plt.plot(magnification, percentage_dead_time * percentage_geometry, label="Total")

    #plt.plot(dtime_data[0][0], dtime_data[0][1] * dtime_data[0][2], label="0 ns")
    #plt.plot(dtime_data[1][0], dtime_data[1][1] * dtime_data[1][2], label="50 ns")

    plt.plot(dtime_data[0][0], dtime_data[0][1] * dtime_data[0][2], label="5 ns")
    plt.plot(dtime_data[1][0], dtime_data[1][1] * dtime_data[1][2], label="20 ns")
    plt.plot(dtime_data[2][0], dtime_data[2][1] * dtime_data[2][2], label="35 ns")
    plt.plot(dtime_data[3][0], dtime_data[3][1] * dtime_data[3][2], label="50 ns")

    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.xlabel("S (Sampling factor)")
    plt.ylabel("Fraction detected")
    plt.show()

    

    print(magnification[np.argmax(percentage_dead_time[35:] * percentage_geometry[35:])])
    print(np.amax(percentage_geometry[35:] * percentage_dead_time[35:]))
    print(percentage_geometry * percentage_dead_time)

def main():
    #basic_simulation_spad23()
    optimal_magnification()


if __name__ == "__main__":
    main()
