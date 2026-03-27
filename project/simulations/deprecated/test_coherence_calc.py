import numpy as np
import matplotlib.pyplot as plt

import project.model.coherence_from_data as coherence
from project.model.detection import show_photons, Spad23, Spad512, merge_photons
from project.model.sample import Alexa647

# Compute coherence curves and fit them (g2(tau)) for SPAD23 for 1 emitter in three ways:

# 1. Over whole detector at once
# 2. For every pixel seperately over its neighbours and return seperate fits
# 3. For every pixel seperately without neighbours and return seperate fits

# 1. Over whole detector at once
def sim_whole_detector(nr_emitters, laser, interval, bin_size, eta, seed, nr_steps=200, dashboard=False, debug=False, **kwargs):
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
    kwargs : dict
        Additional keyword arguments for the SPAD and measurement functions.
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

    # Extract kwargs for spad and measure and get rid of prefix
    spad_kwargs = {k.replace('spad_', ''): v for k, v in kwargs.items() if k.startswith('spad_')}
    measure_kwargs = {k.replace('measure_', ''): v for k, v in kwargs.items() if k.startswith('measure_')}
    
    ######### INITIALIZATION ##########
    s = Spad23(magnification=150, nr_pixel_rows=5, pixel_radius=10.3, **spad_kwargs)

    if nr_emitters == 1:
        #positions = np.array([[0, 0]])
        positions = np.array([[0.3123, 0.1134]])
    elif nr_emitters == 2:
        #positions = np.array([[0, 0], [0, 0]])
        positions = np.array([[0.2123, 0.1634], [0.12342, -0.1234]])
    elif nr_emitters == 3:
        positions = np.array([[0, 0], [0, 0], [0, 0]])
    elif nr_emitters == 4:
        positions = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])

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
    measurement, is_detected = s.measure(photons=photons, duration=interval, seed=seed, debug=debug, **measure_kwargs)
    measured_timestamps = measurement[:, 1]
    # 4) Calculating coherence and expected number of emitters
    print(measured_timestamps)
    print(len(measured_timestamps))
    # Calculating the autocoherence over a neighbourhood
    estimated_emitters_array = []
    pixel_coherence, bins = coherence.auto_coherence(measured_timestamps, interval=interval, bin_size=bin_size, nr_steps=nr_steps, normalize=True)
        #estimated_nr_emitters = expected_number_of_emitters(pixel_autocoh, bin_size, 1, 1)
        #estimated_emitters_array.append(estimated_nr_emitters)
    #estimated_emitters_array = np.array(estimated_emitters_array)

    if dashboard:
        return pixel_coherence, bins, s, emitters, photons, measurement, is_detected, estimated_emitters_array
    
def sim_pixels_autocoherence(nr_emitters, laser, interval, bin_size, eta, seed, nr_steps=200, dashboard=False, debug=False, **kwargs):
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

    # Extract kwargs for spad and measure
    spad_kwargs = {k.replace('spad_', ''): v for k, v in kwargs.items() if k.startswith('spad_')}
    measure_kwargs = {k.replace('measure_', ''): v for k, v in kwargs.items() if k.startswith('measure_')}
    
    ######### INITIALIZATION ##########
    s = Spad23(magnification=150, nr_pixel_rows=5, pixel_radius=10.3, **spad_kwargs)

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
        e.generate_photons(laser_power=laser, time_interval=interval, seed=(j+1)*seed, detection_efficiency=eta, widefield=False)
        emitters.append(e)

    # 2) #TODO: Translating sample plane to imaging plane 
    photons = s.magnify(merge_photons(emitters), debug=False)

    # 3) Measuring the photons at the detector
    measurement, is_detected = s.measure(photons=photons, duration=interval, seed=seed, debug=False, **measure_kwargs)
    measured_timestamps = measurement[:, 1]
    # 4) Calculating coherence and expected number of emitters

    # Calculating the autocoherence for every pixel.
    estimated_emitters_array = []
    pixel_autocoherence_array = []
    
    #print(s.data_per_pixel[4])
    #print(len(s.data_per_pixel[4]))
    for i in range(s.nr_pixels):
        #print(len(s.data_per_pixel[i]))
        pixel_autocoherence, bins = coherence.auto_coherence(s.data_per_pixel[i], interval=interval, bin_size=bin_size, nr_steps=nr_steps, normalize=True)
        #print(pixel_autocoherence)
        pixel_autocoherence_array.append(pixel_autocoherence)
        #estimated_nr_emitters = coherence.expected_number_of_emitters(pixel_autocoherence, bin_size, 1, 1)
        #estimated_emitters_array.append(estimated_nr_emitters)
    
    if dashboard:
        return pixel_autocoherence_array, bins, s, emitters, photons, measurement, is_detected, estimated_emitters_array
    

def sim_neighbourhood_autocoherence_weighed(nr_emitters, laser, interval, bin_size, eta, seed, nr_steps=200, dashboard=False, debug=False, **kwargs):
    """
    Implements the simulation of the coherence calculation for the SPAD23 sensor. The coherence is calculated for every pixel with its neighbours; the centre pixel (in terms of intensity) is determined and from that pixel the other pixel coherences are add up and weighed by distance.

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

    # Extract kwargs for spad and measure
    spad_kwargs = {k.replace('spad_', ''): v for k, v in kwargs.items() if k.startswith('spad_')}
    measure_kwargs = {k.replace('measure_', ''): v for k, v in kwargs.items() if k.startswith('measure_')}
    
    ######### INITIALIZATION ##########
    s = Spad23(magnification=150, nr_pixel_rows=5, pixel_radius=10.3, **spad_kwargs)

    if nr_emitters == 1:
        positions = np.array([[0, 0]])
        #positions = np.array([[-100, -100]])
    elif nr_emitters == 2:
        #positions = np.array([[0, 0], [0, 0]])
        positions = np.array([[0.2123, 0.1634], [0.12342, -0.1234]])
    elif nr_emitters == 3:
        positions = np.array([[0, 0], [0, 0], [0, 0]])
    elif nr_emitters == 4:
        positions = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])

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
    measurement, is_detected = s.measure(photons=photons, duration=interval, seed=seed, debug=False, **measure_kwargs)
    measured_timestamps = measurement[:, 1]
    # 4) Calculating coherence and expected number of emitters
    estimated_emitters_array = []

    # Calculating the autocoherence over a neighbourhood
    pixel_neighbourhood_coherence_array, bins = coherence.coherence_per_neighbourhood(s, interval, bin_size, nr_steps=nr_steps, kernel_size = 1)

    # Add up all pixel coherences and weigh them by distance from the centre pixel (pixel 11)
    pixel_neighbourhood_coherence_array_weighed = np.zeros_like(pixel_neighbourhood_coherence_array[0])
    for i in range(s.nr_pixels):
        if i == 10:
            continue
        #print("Distance:", np.exp(-np.linalg.norm(s.pixel_coordinates[i] - s.pixel_coordinates[10])))
        #print("Alternative distance:", np.linalg.norm(s.pixel_coordinates[i] - s.pixel_coordinates[10]))
        pixel_neighbourhood_coherence_array_weighed += pixel_neighbourhood_coherence_array[i] * np.linalg.norm(s.pixel_coordinates[i] - s.pixel_coordinates[10])
    # Compute final coherence by normalizing
    pixel_neighbourhood_coherence_array_weighed /= np.max(pixel_neighbourhood_coherence_array_weighed)
    #print(pixel_neighbourhood_coherence_array_weighed)
    if dashboard:
        return pixel_neighbourhood_coherence_array_weighed, bins, s, emitters, photons, measurement, is_detected, estimated_emitters_array

def sim_neighbourhood_autocoherence(nr_emitters, laser, interval, bin_size, eta, seed, nr_steps=200, dashboard=False, debug=False, **kwargs):
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

    # Extract kwargs for spad and measure
    spad_kwargs = {k.replace('spad_', ''): v for k, v in kwargs.items() if k.startswith('spad_')}
    measure_kwargs = {k.replace('measure_', ''): v for k, v in kwargs.items() if k.startswith('measure_')}
    
    ######### INITIALIZATION ##########
    s = Spad23(magnification=150, nr_pixel_rows=5, pixel_radius=10.3, **spad_kwargs)

    if nr_emitters == 1:
        positions = np.array([[0, 0]])
        #positions = np.array([[-100, -100]])

    elif nr_emitters == 2:
        positions = np.array([[0, 0], [0, 0]])
    elif nr_emitters == 3:
        positions = np.array([[0, 0], [0, 0], [0, 0]])
        #positions = np.array([[-50, 0], [50, 0], [0, -50]])
    elif nr_emitters == 4:
        positions = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])

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
    measurement, is_detected = s.measure(photons=photons, duration=interval, seed=seed, debug=False, **measure_kwargs)
    measured_timestamps = measurement[:, 1]
    # 4) Calculating coherence and expected number of emitters
    estimated_emitters_array = []
    
    # Calculating the autocoherence over a neighbourhood
    pixel_neighbourhood_coherence_array, bins = coherence.coherence_per_neighbourhood(s, interval, bin_size, nr_steps = nr_steps, kernel_size = 1)
    if dashboard:
        return pixel_neighbourhood_coherence_array, bins, s, emitters, photons, measurement, is_detected, estimated_emitters_array
    
def sim_neighborhood_coherence_pixelautocoherence_substracted(nr_emitters, laser, interval, bin_size, eta, seed, nr_steps=200, dashboard=False, debug=False, **kwargs):
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

    # Extract kwargs for spad and measure
    spad_kwargs = {k.replace('spad_', ''): v for k, v in kwargs.items() if k.startswith('spad_')}
    measure_kwargs = {k.replace('measure_', ''): v for k, v in kwargs.items() if k.startswith('measure_')}
    
    ######### INITIALIZATION ##########
    s = Spad23(magnification=150, nr_pixel_rows=5, pixel_radius=10.3, **spad_kwargs)

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
        e.generate_photons(laser_power=laser, time_interval=interval, seed=(j+1)*seed, detection_efficiency=eta, widefield=False)
        emitters.append(e)

    # 2) #TODO: Translating sample plane to imaging plane 
    photons = s.magnify(merge_photons(emitters), debug=False)

    # 3) Measuring the photons at the detector
    measurement, is_detected = s.measure(photons=photons, duration=interval, seed=seed, debug=False, **measure_kwargs)
    measured_timestamps = measurement[:, 1]
    # 4) Calculating coherence and expected number of emitters
    estimated_emitters_array = []

    # Compute the whole-detector coherence
    pixel_coherence, bins = coherence.auto_coherence(measured_timestamps, interval=interval, bin_size=bin_size, nr_steps=nr_steps, normalize=False)

    # Calculating the autocoherence for each pixel
    estimated_emitters_array = []
    pixel_autocoherence_array = []
    for i in range(s.nr_pixels):
        pixel_autocoherence, bins = coherence.auto_coherence(s.data_per_pixel[i], interval=interval, bin_size=bin_size, nr_steps=nr_steps, normalize=False)
        pixel_autocoherence_array.append(pixel_autocoherence)
    
    # Subtract a weighted sum of the pixel autocoherences from the whole detector coherence
    #pixel_autocoherence_weighed = np.sum(pixel_autocoherence_array, axis=0) / s.nr_pixels

    pixel_neighbourhood_coherence = pixel_coherence - np.sum(pixel_autocoherence_array, axis=0)
    # Normalize the result
    pixel_neighbourhood_coherence /= np.max(pixel_neighbourhood_coherence)
    print(f"Result: {pixel_neighbourhood_coherence}")

    if dashboard:
        return pixel_neighbourhood_coherence, bins, s, emitters, photons, measurement, is_detected, estimated_emitters_array
