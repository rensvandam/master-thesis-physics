from ..examples.run_scanning_experiment import run_scanning_experiment
from project.model.localization import localize
import matplotlib.pyplot as plt

#darkcountrates = [0,1,10,100,1000,2000,5000,10000] # 100cps reported
#darkcountrates = [0,1000]
#laser_power = [i*25E3 for i in range(1, 12)]
laser_power = [50E3, 100E3, 150E3, 200E3, 250E3, 300E3]
variable = laser_power
variable_name = 'Laser Power'
use_saved_data = True

iterations_per_variable = 1
seed = 25

errors = [0] * len(variable)
estimated_locations = [0] * len(variable)
for i, value in enumerate(variable):
    laser_power = value
    if use_saved_data:
        


    else:
        print(f"Running simulation with {variable_name}: {value}")
        setup, I_meas, Gd_meas, est_emitters, metadata = run_scanning_experiment(
            enable_noise=False,
            laser_power=value,
            area_size = (1, 1),
            positions = (20, 20),
            emitter_density=5,
            show_plots=False,
            save_data=False,
            seed=seed
        )

    # Localize
    print("Localizing emitters...")
    localization = localize(I_meas.T, Gd_meas.T, est_emitters.T, metadata, plot=False, psf_file = 'project/data/psf.json', verbose=True)
    estimated_locations[i] = localization['emitters']
    true_locations = metadata['emitter_positions']
    errors[i] = localization['RSS']

    # Plot
    # Plot Gd_meas and emitter positions

    pixel_size = metadata['pixel_size']
    num_pixels = metadata['positions'][0]
    area_size = metadata['area_size']

    plt.figure(figsize=(10, 6))
    plt.imshow(Gd_meas, cmap='viridis', origin='lower', interpolation='nearest', extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2])
    plt.colorbar(label='Gd_meas')
    plt.title(f"Gd_meas for {variable_name}: {value:.2e}")
    plt.scatter([loc[1] for loc in metadata['emitter_positions']], [loc[0] for loc in metadata['emitter_positions']], c='red', s=100, alpha=0.7, label='True', marker='o')
    est_x = [loc['x']*pixel_size - pixel_size*num_pixels/2 for loc in estimated_locations[i]]
    est_y = [loc['y']*pixel_size - pixel_size*num_pixels/2 for loc in estimated_locations[i]]
    plt.scatter(est_y, est_x, c='blue', s=100, alpha=0.7, label='Estimated', marker='x')

    plt.figure(figsize=(10, 6))
    plt.imshow(est_emitters, cmap='viridis', origin='lower', interpolation='nearest', extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2])
    plt.colorbar(label='Gd_meas')
    plt.title(f"est_emitters for {variable_name}: {value:.2e}")
    plt.scatter([loc[1] for loc in metadata['emitter_positions']], [loc[0] for loc in metadata['emitter_positions']], c='red', s=100, alpha=0.7, label='True', marker='o')
    est_x = [loc['x']*pixel_size - pixel_size*num_pixels/2 for loc in estimated_locations[i]]
    est_y = [loc['y']*pixel_size - pixel_size*num_pixels/2 for loc in estimated_locations[i]]
    plt.scatter(est_y, est_x, c='blue', s=100, alpha=0.7, label='Estimated', marker='x')

    plt.figure(figsize=(10, 6))
    plt.imshow(I_meas, cmap='viridis', origin='lower', interpolation='nearest', extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2])
    plt.colorbar(label='I_meas')
    plt.title(f"I_meas for {variable_name}: {value:.2e}")
    plt.scatter([loc[1] for loc in metadata['emitter_positions']], [loc[0] for loc in metadata['emitter_positions']], c='red', s=100, alpha=0.7, label='True', marker='o')
    est_x = [loc['x']*pixel_size - pixel_size*num_pixels/2 for loc in estimated_locations[i]]
    est_y = [loc['y']*pixel_size - pixel_size*num_pixels/2 for loc in estimated_locations[i]]
    plt.scatter(est_y, est_x, c='blue', s=100, alpha=0.7, label='Estimated', marker='x')


#plot darkcount vs localization error
plt.figure(figsize=(10, 6))
plt.title(f"{variable_name} vs Localization Error")
plt.plot(variable, errors)
plt.show()

# Plot estimated vs true locations for each dark count rate
n_rates = len(variable)
cols = min(4, n_rates)  # Max 4 columns
rows = (n_rates + cols - 1) // cols  # Ceiling division

pixel_size = metadata['pixel_size']
num_pixels = metadata['positions'][0]

plt.figure(figsize=(4*cols, 4*rows))

print("True locations:", metadata['emitter_positions'])
for i, value in enumerate(variable):
    plt.subplot(rows, cols, i+1)
    
    true_locs = metadata['emitter_positions']  # Assuming this is the same for all runs
    est_locs = estimated_locations[i]
    
    print("Estimated locations:", est_locs)
    
    # Plot true locations as red circles
    if len(true_locs) > 0:
        true_x = [loc[0] for loc in true_locs]
        true_y = [loc[1] for loc in true_locs]
        plt.scatter(true_x, true_y, c='red', s=100, alpha=0.7, label='True', marker='o')
    
    # Plot estimated locations as blue crosses
    if len(est_locs) > 0:
        est_x = [loc['x']*pixel_size - pixel_size*num_pixels/2 for loc in est_locs]
        est_y = [loc['y']*pixel_size - pixel_size*num_pixels/2 for loc in est_locs]
        plt.scatter(est_x, est_y, c='blue', s=100, alpha=0.7, label='Estimated', marker='x')
    
    plt.title(f"{variable_name}: {value:.2e}\nError: {errors[i]:.3f}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

plt.tight_layout()
plt.show()