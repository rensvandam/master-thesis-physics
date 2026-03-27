import os
import pickle
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from abc import ABC, abstractmethod
import hashlib
import h5py

from project.model.coherence_from_data import auto_coherence, show_coherence, coherence, generate_est_nr_emitters_map
from project.model.detection import show_photons, Spad23, Spad512, merge_photons
from project.model.sample import Alexa647
from project.model.setup import Setup, ScanningSetup, WidefieldSetup
from project.model.helper_functions import generate_random_positions, get_readable_filename, get_sim_hash

def run_scanning_experiment(
    # Sensor parameters
    magnification=150,
    pixel_radius=10.3,
    spacing=23,
    crosstalk=0.0014, #(0.14%)
    afterpulsing=0.001, #(0.1%)
    jitter=0.12,
    dead_time=50, #50 # nanoseconds (ns)
    dark_count_rate=100, # dark counts per pixel per second (cps)
    
    # Scan parameters
    scan_speed=1,
    step_size=0.05, # micrometers (μm)
    dwell_time=0.1, # milliseconds (ms)
    
    # Simulation parameters
    area_size=(0.5, 0.5),
    positions=(10, 10),
    laser_power=50E3,
    beam_waist=0.3,
    emitter_density=5,
    emitters_manual=False,
    seed=25,
    enable_noise=False,
    detection_efficiency=1,
    max_delay=800,
    
    # Output options
    save_data=True,
    show_plots=True
):
    """
    Run a scanning experiment with sensible defaults.
    All parameters have defaults so you can call it with no arguments.

    Parameters:
    - emitter_density: Number of emitters per square micrometer. Emitters are placed randomly by default.
    - emitters_manual: List of tuples with emitter positions [(x1, y1), (x2, y2), ...] in micrometers. Default: False.
    etc.

    Returns:
    - setup: ScanningSetup object with the scan data.
    - photon_count_map: 2D array of photon counts per pixel.
    - G2_diff_map: 2D array of G2 difference values.
    - nr_emitters_map: 2D array of estimated number of emitters per pixel.
    - metadata: Dictionary with metadata about the scan.
    """

    print(f"Starting scanning experiment: {area_size[0]}×{area_size[1]} μm area...")
    
    # Create sensor
    sensor = Spad23(
        magnification=magnification,
        pixel_radius=pixel_radius,
        spacing=spacing,
        crosstalk=crosstalk,
        afterpulsing=afterpulsing,
        jitter=jitter,
        dead_time=dead_time,
        dark_count_rate=dark_count_rate
    )
    
    # Create scanning setup
    setup = ScanningSetup(
        sensor=sensor,
        magnification=magnification,
        scan_speed=scan_speed,
        step_size=step_size,
        dwell_time=dwell_time
    )
    
    bounds = (-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2)

    # Generate emitters
    if emitters_manual:
        emitter_positions = emitters_manual
        for i, (x, y) in enumerate(emitters_manual):
            assert bounds[0] <= x <= bounds[1], f"Emitter {i} x-coordinate {x} is outside bounds [{bounds[0]}, {bounds[1]}]"
            assert bounds[2] <= y <= bounds[3], f"Emitter {i} y-coordinate {y} is outside bounds [{bounds[2]}, {bounds[3]}]"
        emitters = [Alexa647(x=pos[0], y=pos[1]) for pos in emitter_positions]
        print(f"Manually created {len(emitters)} emitter(s). All {len(emitters_manual)} emitters are within bounds {bounds}")

    else:
        try:
            emitter_positions = generate_random_positions(bounds=bounds, density=emitter_density, seed=seed, avoid_boundaries=True)
            emitters = [Alexa647(x=pos[0], y=pos[1]) for pos in emitter_positions]
            print(f"Generated {len(emitters)} emitters")
        except:
            print("Failed to generate emitters with bounds. Trying without bounds.")
            emitter_positions = generate_random_positions(bounds=bounds, density=emitter_density, seed=seed, avoid_boundaries=False)
            emitters = [Alexa647(x=pos[0], y=pos[1]) for pos in emitter_positions]
            print(f"Generated {len(emitters)} emitters")

    # Run simulation
    print("Running scan...")
    scan_data = setup.scan_area(
        emitters=emitters,
        area_size=area_size,
        positions=positions,
        laser_power=laser_power,
        beam_waist=beam_waist,
        seed=seed,
        enable_noise=enable_noise,
        detection_efficiency=detection_efficiency,
        calculate_g2=True,
        calculate_G2=True,
        max_delay=max_delay,
        enable_ism=True,
        plot_results=show_plots
    )
    setup.scan_data = scan_data

    timestamps_by_position = scan_data['detector_data']
    #print("TIMESTAMPS")
    #print(timestamps_by_position)

    nr_emitters_map = scan_data['nr_emitters_map']
    sigma_emitters_map = scan_data['sigma_emitters_map']
    #nr_emitters_map, sigma_map = generate_est_nr_emitters_map(scan_data, min_photon_count=10, method='with_k', initial_guess=np.array([3, 2]), verbose=True)
    G2_diff_map = scan_data['G2_diff_map']
    #_, _, G2_diff_map = setup.plot_G2_difference_map()
    
    # Generate plots
    if show_plots:
        print("Generating plots...")
        
        # Photon count map
        fig, ax = plt.subplots(figsize=(12, 8))
        setup.plot_photon_count_map(fig=fig, ax=ax, emitters=emitters, cmap='hot', show_emitters=True)
        
        # g2 map
        fig, ax = plt.subplots(figsize=(12, 8))
        setup.plot_g2_map(fig=fig, ax=ax, delay_idx=5, cmap='viridis')
        
        # # g2 curves
         #fig, ax = plt.subplots(figsize=(10, 5))
         #setup.plot_g2_curves(fig=fig, ax=ax, positions=[(15, 9)], plot_fit=True, laser_power=laser_power)

        # # g2 curves
        # fig, ax = plt.subplots(figsize=(10, 5))
        # setup.plot_g2_curves(fig=fig, ax=ax, positions=[(16, 11)], plot_fit=True, laser_power=laser_power)

        # # g2 curves
        # fig, ax = plt.subplots(figsize=(10, 5))
        # setup.plot_g2_curves(fig=fig, ax=ax, positions=[(15, 10)], plot_fit=True, laser_power=laser_power)
        
        # # g2 curves
        # fig, ax = plt.subplots(figsize=(10, 5))
        # setup.plot_g2_curves(fig=fig, ax=ax, positions=[(10, 1)], plot_fit=True, laser_power=laser_power)
        
        # # g2 curves
        # fig, ax = plt.subplots(figsize=(10, 5))
        # setup.plot_g2_curves(fig=fig, ax=ax, positions=[(16, 7)], plot_fit=True, laser_power=laser_power)
        
        # Number of emitters estimation
        fig, ax = plt.subplots(figsize=(10, 5))
        # _, _, _, _ = setup.plot_est_nr_emitters(
        #     fig=fig, ax=ax, emitters=emitters, show_emitters=True, min_photon_count=10
        # )
        img = ax.imshow(nr_emitters_map, cmap='inferno', interpolation='nearest')
        plt.colorbar(img, label="Emitters", fraction=0.046, pad=0.04)

        
        # G2 difference map
        fig, ax = plt.subplots(figsize=(10, 5))
        _, _, G2_diff_map = setup.plot_G2_difference_map()
        
        # PSF comparison
        setup.plot_psf_comparison(scan_data['ism_results'], scan_data['G2_diff_map'], emitters=emitters)
        
        plt.tight_layout()
        if save_data:
            plt.savefig('scanning_results.png', dpi=150, bbox_inches='tight')
        plt.show()

    # Save metadata (just the important stuff)
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    metadata = {
        'date': date,
        'area_size': area_size,
        'pixel_size': step_size,
        'positions': positions,
        'dwell_time': dwell_time,
        'emitter_count': len(emitters),
        'emitter_positions': [(e.x, e.y) for e in emitters],
        'laser_power': laser_power,
        'detection_efficiency': detection_efficiency,
        'magnification': magnification,
        'seed': seed,
        'enable_noise': enable_noise,
        'dead_time': dead_time,
        'dark_count_rate' : dark_count_rate,
        'crosstalk' : crosstalk,
        'afterpulsing' : afterpulsing,
        'jitter' : jitter
    }
    
    # Save data
    if save_data:
        print("Saving data...")
        data_dir = f"./project/data/{date}"
        os.makedirs(data_dir, exist_ok=True)
                
        filename = f"{data_dir}/sim_{get_readable_filename(metadata)}.h5"
        with h5py.File(filename, 'w') as f:
            f['photon_counts'] = scan_data['ism_results']['ism_image']#scan_data['photon_count_map']
            f['nr_emitters'] = nr_emitters_map  
            f['G2_diff'] = scan_data['ism_results']['G2_ism_image']#G2_diff_map
            f['timestamps'] = timestamps_by_position
            f['original_intensity_image'] = scan_data['photon_count_map']
            f['original_G2_image'] = scan_data['G2_diff_map']

            f.attrs.update(metadata)  # metadata as attributes
        
        print(f"Data saved to {filename}")
   

    #return setup, scan_data['photon_count_map'], G2_diff_map, nr_emitters_map, metadata
    return setup, scan_data['ism_results']['ism_image'], scan_data['ism_results']['G2_ism_image'], nr_emitters_map, metadata


# Run a test experiment with default parameters
if __name__ == "__main__":
    run_scanning_experiment()



    #run_scanning_experiment()