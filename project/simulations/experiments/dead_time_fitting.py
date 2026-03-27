import numpy as np
import matplotlib.pyplot as plt

import project.model.coherence_from_data as coherence
from project.model.detection import show_photons, Spad23, Spad512, merge_photons
from project.model.sample import Alexa647, expected_excitation_time

#1-4-2025
#What to do:
# Investigate fitting to the deadtime effects to gather information.
# How?
# Plot on the y-axis the coefficient of an exponential fitted to the deadtime decrease after t_dead_time and on the x-axis the excitation rate of the fit, so for range of rates (laser powers)

# 1. Over whole detector at once
def sim_whole_detector(dead_time, nr_emitters, laser, interval, bin_size, eta, seed, nr_steps=200, dashboard=False, debug=False):
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
    s = Spad23(magnification=150, nr_pixel_rows=5, pixel_radius=10.3, dead_time=dead_time)

    if nr_emitters == 1:
        positions = np.array([[0, 0]])
        #positions = np.array([[-100, -100]])
    elif nr_emitters == 2:
        positions = np.array([[0, 0], [0, 0]])
    elif nr_emitters == 3:
        positions = np.array([[0, 0], [0, 0], [0, 0]])
    elif nr_emitters == 4:
        positions = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    elif nr_emitters == 5:
        positions = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])

    ######### PIPELINE ##########
    # 1) Generating photons from the emitters
    emitters = []
    for j in range(nr_emitters):
        e = Alexa647(x=positions[j, 0], y=positions[j, 1])
        e.generate_photons(laser_power=laser, time_interval=interval, seed=(j+1)*seed, detection_efficiency=eta, widefield=False)
        emitters.append(e)

    print("Excitation coefficient: ",e.extinction_coefficient)
    # 2) #TODO: Translating sample plane to imaging plane 
    photons = s.magnify(merge_photons(emitters), debug=False)

    # 3) Measuring the photons at the detector
    measurement, is_detected = s.measure(photons=photons, duration=interval, seed=seed, debug=False)
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

    ##########################
    # Fit the deadtime decay
    # fit, popt, pcov = coherence.fit_deadtime_decay(bins, pixel_coherence, initial_guess=np.array([0.5, 0.5]))

    # sigma = np.sqrt(np.diag(pcov))[0]
    # b = np.round(popt[0], 2) # decay amplitude
    # c = np.round(popt[1], 2) # decay rate

    # Calculate expected excitation rate
    exc_rate = 1/expected_excitation_time(e.extinction_coefficient, e.absorption_wavelength,
                                                       laser)
    print(f"Expected excitation rate: {exc_rate} ns")

    ########################
    # Fit the pixel coherence
    fit, popt, pcov = coherence.fit_coherence_function(bins, pixel_coherence, method='without_k', initial_guess=np.array([3]))
    # Compute standard deviation of the fitted value for n
    sigma = np.sqrt(np.diag(pcov))[0]
    n = np.round(popt[0], 2)
    return n, sigma, bins, pixel_coherence, exc_rate
    

#######################################################
## Simulation for 1 emitters with different excitation rates
#######################################################

# deadtime = 50 # deadtime in ns
# powers = np.linspace(10, 500, 9) # excitation rates in /ns
# amplitudes_vs_power = []
# decay_rates_vs_power = []
# for power in powers:
#     decay_amplitudes = []
#     decay_rates = []
#     # Use 100 different seeds for each fraction
#     for seed in range(10):
#         b, c, bins, coherence = sim_whole_detector(deadtime, nr_emitters=1, laser=power* 10**3, 
#                                    interval=10**5, bin_size=0.1, eta=1.0, 
#                                    seed=seed, nr_steps=200)
#         decay_amplitudes.append(b)
#         decay_rates.append(c)
#     # Calculate mean and standard deviation of the 10 measurements
#     mean_amp = np.mean(decay_amplitudes)
#     mean_rate = np.mean(decay_rates)
#     amplitudes_vs_power.append(mean_amp)
#     decay_rates_vs_power.append(mean_rate)

# # Create error bar plot
# plt.figure(figsize=(10, 6))
# plt.plot(powers, decay_rates_vs_power, label='Decay Amplitude', marker='o')
# #plt.errorbar(powers, estimated_emitters, yerr=standard_deviations, fmt='o', capsize=5)
# plt.xlabel("Deadtime [ns]")
# plt.ylabel("Estimated number of emitters")
# plt.title("Emitter Estimation with Variable Deadtime")
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()

#########################################################
#########################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Assuming sim_whole_detector has been run and you have b, c, bins, coherence

# Function to create the plot with the trimmed data and fit
# Function to create the plot with the trimmed data and fit
def plot_fit_after_deadtime(bins, coherence, deadtime=50):
    # Find index of bins just after deadtime
    start_idx = np.where(bins > deadtime)[0][0]
    
    # Trim the data to only include points after the deadtime
    trimmed_bins = bins[start_idx:]
    trimmed_coherence = coherence[start_idx:]
    
    # Define exponential decay function for fitting
    def exp_decay(x, amplitude, decay_rate, offset):
        return offset + amplitude * np.exp(-decay_rate * (x - deadtime))
    
    
    # Initial parameter guesses
    p0 = [0.2, 0.03, 1.0]  # amplitude, decay_rate, offset
    
    # Perform the fit on the trimmed data
    try:
        popt, pcov = curve_fit(exp_decay, trimmed_bins, trimmed_coherence, p0=p0)
        amplitude, decay_rate, offset = popt
        
        # Generate fit curve
        fit_x = np.linspace(deadtime, max(bins), 1000)
        fit_y = exp_decay(fit_x, *popt)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot the full data in light blue
        plt.plot(bins, coherence, 'b-', alpha=0.5, linewidth=1, label='Full coherence data')
        
        # Plot the trimmed data in darker blue
        plt.plot(trimmed_bins, trimmed_coherence, 'b-', alpha=0.8, linewidth=2, label='Post-deadtime data')
        
        # Plot the fit in red
        plt.plot(fit_x, fit_y, 'r-', linewidth=2, 
                 label=f'Fit: {offset:.2f} + {amplitude:.2f}·exp(-{decay_rate:.3f}·(τ-{deadtime}))')
        
        # Add labels and title
        plt.xlabel('τΔt [ns]')
        plt.ylabel('g^(2)(τ)')
        plt.title('Coherence fit after deadtime')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Add text box with fit parameters
        plt.text(0.05, 0.05, 
                 f'Amplitude: {amplitude:.3f}\nDecay rate: {decay_rate:.3f}\nOffset: {offset:.3f}', 
                 transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Display vertical line at deadtime
        plt.axvline(x=deadtime, color='g', linestyle='--', alpha=0.5, label='Deadtime')
        
        plt.tight_layout()
        return popt, pcov
    
    except Exception as e:
        print(f"Fitting error: {e}")
        plt.figure(figsize=(10, 6))
        plt.plot(bins, coherence, 'b-', label='Coherence data')
        plt.axvline(x=deadtime, color='g', linestyle='--', label='Deadtime')
        plt.xlabel('τΔt [ns]')
        plt.ylabel('g^(2)(τ)')
        plt.title('Coherence data (fitting failed)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        return None, None

# Example usage:
# deadtime = 50
# b, c, bins, coh, exc_rate = sim_whole_detector(deadtime, nr_emitters=3, laser=500* 10**3, 
#                                    interval=10**5, bin_size=0.1, eta=1.0, 
#                                    seed=345, nr_steps=1200)

# popt, pcov = plot_fit_after_deadtime(bins, coh, deadtime=50)
# plt.show()

# To use this with your loop where you're collecting data for different powers:
def process_simulations_with_exc_rates(powers, deadtime=50):
    # Lists to store mean values for each power level
    mean_decay_amplitudes = []
    mean_decay_rates = []
    mean_offsets = []
    mean_exc_rates = []
    
    # Lists to store standard deviations for error bars
    std_decay_amplitudes = []
    std_decay_rates = []
    std_offsets = []
    std_exc_rates = []
    
    for power in powers:
        # Lists to store values for this power level across different seeds
        power_amplitudes = []
        power_decay_rates = []
        power_offsets = []
        power_exc_rates = []
        
        for seed in range(10):
            # Note the updated unpacking to include exc_rate
            try:
                b, c, bins, coherence, exc_rate = sim_whole_detector(deadtime, nr_emitters=3, laser=power*10**3,
                                                    interval=10**5, bin_size=0.1, eta=1.0,
                                                    seed=seed, nr_steps=1200)
                
                # Store the excitation rate
                power_exc_rates.append(exc_rate)
                
                # Fit only the post-deadtime data
                popt, _ = plot_fit_after_deadtime(bins, coherence, deadtime)
                
                if popt is not None:
                    amplitude, decay_rate, offset = popt
                    power_amplitudes.append(amplitude)
                    power_decay_rates.append(decay_rate)
                    power_offsets.append(offset)
                    
                    # Save the plot for this specific run if desired
                    plt.savefig(f'exc_rate_{exc_rate:.2f}_seed_{seed}_fit.png')
                    plt.close()
            except Exception as e:
                print(f"Error in simulation with power {power}, seed {seed}: {e}")
        
        # Calculate means and standard deviations for this power level
        if power_amplitudes:
            mean_decay_amplitudes.append(np.mean(power_amplitudes))
            mean_decay_rates.append(np.mean(power_decay_rates))
            mean_offsets.append(np.mean(power_offsets))
            mean_exc_rates.append(np.mean(power_exc_rates))
            
            std_decay_amplitudes.append(np.std(power_amplitudes))
            std_decay_rates.append(np.std(power_decay_rates))
            std_offsets.append(np.std(power_offsets))
            std_exc_rates.append(np.std(power_exc_rates))
        else:
            # If no successful fits, append NaN
            mean_decay_amplitudes.append(np.nan)
            mean_decay_rates.append(np.nan)
            mean_offsets.append(np.nan)
            mean_exc_rates.append(np.nan)
            
            std_decay_amplitudes.append(np.nan)
            std_decay_rates.append(np.nan)
            std_offsets.append(np.nan)
            std_exc_rates.append(np.nan)
    
    # Create x-error bars for excitation rate variation (if needed)
    # Plot the decay rates vs excitation rate WITH ERROR BARS
    plt.figure(figsize=(10, 6))
    plt.errorbar(mean_exc_rates, mean_decay_rates, 
                 yerr=std_decay_rates, 
                 xerr=std_exc_rates,
                 fmt='o-', capsize=5, elinewidth=1, markeredgewidth=1,
                 label='Decay rate')
    
    plt.xlabel("Excitation rate [/ns]")
    plt.ylabel("Decay rate [1/ns]")
    plt.title("Decay rate vs. excitation rate")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot the amplitude vs excitation rate WITH ERROR BARS
    plt.figure(figsize=(10, 6))
    plt.errorbar(mean_exc_rates, mean_decay_amplitudes, 
                 yerr=std_decay_amplitudes, 
                 xerr=std_exc_rates,
                 fmt='o-', capsize=5, elinewidth=1, markeredgewidth=1,
                 label='Amplitude')
    
    plt.xlabel("Excitation rate [/ns]")
    plt.ylabel("Amplitude")
    plt.title("Amplitude vs. excitation rate")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot the offset vs excitation rate WITH ERROR BARS
    plt.figure(figsize=(10, 6))
    plt.errorbar(mean_exc_rates, mean_offsets, 
                 yerr=std_offsets, 
                 xerr=std_exc_rates,
                 fmt='o-', capsize=5, elinewidth=1, markeredgewidth=1,
                 label='Offset')
    
    plt.xlabel("Excitation rate [/ns]")
    plt.ylabel("Offset")
    plt.title("Offset vs. excitation rate")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Create a table of results
    results = {
        'Excitation Rate': mean_exc_rates,
        'Excitation Rate Std': std_exc_rates,
        'Amplitude': mean_decay_amplitudes,
        'Amplitude Std': std_decay_amplitudes,
        'Decay Rate': mean_decay_rates,
        'Decay Rate Std': std_decay_rates,
        'Offset': mean_offsets,
        'Offset Std': std_offsets
    }
    
    return results


# Single coherence plot for a specific simulation
def plot_single_coherence(deadtime=50, power=100, seed=0):
    n, sigma, bins, coh, exc_rate = sim_whole_detector(deadtime, nr_emitters=3, laser=power*10**3,
                                         interval=10**5, bin_size=0.1, eta=1.0,
                                         seed=seed, nr_steps=1200)
    
    # Find index of bins just after deadtime
    start_idx = np.where(bins > deadtime)[0][0]
    
    # Trim the data to only include points after the deadtime
    trimmed_bins = bins[start_idx:]
    trimmed_coherence = coh[start_idx:]
    
    # Define exponential decay function for fitting
    def exp_decay(x, amplitude, decay_rate, offset):
        return offset + amplitude * np.exp(-decay_rate * (x - deadtime))
    
    # Initial parameter guesses
    p0 = [0.2, 0.03, 1.0]  # amplitude, decay_rate, offset
    
    # Perform the fit on the trimmed data
    popt, pcov = curve_fit(exp_decay, trimmed_bins, trimmed_coherence, p0=p0)
    amplitude, decay_rate, offset = popt
    
    # Generate fit curve
    fit_x = np.linspace(deadtime, max(bins), 1000)
    fit_y = exp_decay(fit_x, *popt)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot the full data in blue
    plt.plot(bins, coh, 'b-', alpha=0.7, label='Coherence data')
    
    # Plot the fit in red
    plt.plot(fit_x, fit_y, 'r-', linewidth=2, 
             label=f'Fit: {offset:.2f} + {amplitude:.2f}·exp(-{decay_rate:.3f}·(τ-{deadtime}))')
    
    # Add labels and title
    plt.xlabel('τΔt [ns]')
    plt.ylabel('g^(2)(τ)')
    plt.title(f'Coherence for excitation rate = {exc_rate:.2f} /ns')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add text box with fit parameters
    plt.text(0.05, 0.05, 
             f'Amplitude: {amplitude:.3f}\nDecay rate: {decay_rate:.3f}\nOffset: {offset:.3f}\nExcitation rate: {exc_rate:.3f} /ns', 
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Display vertical line at deadtime
    plt.axvline(x=deadtime, color='g', linestyle='--', alpha=0.5, label='Deadtime')
    
    plt.tight_layout()
    plt.show()
    
    return exc_rate, amplitude, decay_rate, offset


# Example usage:
powers = np.linspace(10, 600, 9)
results = process_simulations_with_exc_rates(powers, deadtime=50)

# To plot a single coherence function with fit:
#exc_rate, amplitude, decay_rate, offset = plot_single_coherence(deadtime=50, power=100, seed=0)


####################################################
#####################################################
### Fitting the excitation rate vs decay rate data
#####################################################

# Logarithmic function: a + b*log(x + c)
# The +c term inside the log prevents issues with log(0)
def log_function(x, a, b, c):
    return a + b * np.log(x + c)

# Saturation function: a * (1 - exp(-b * x)) + c
def saturation_function(x, a, b, c):
    return a * (1 - np.exp(-b * x)) + c

# Hill-type function: a + (b * x^n) / (c^n + x^n)
def hill_function(x, a, b, c, n):
    return a + (b * x**n) / (c**n + x**n)

# Function to fit and plot the data
def fit_decay_rate_vs_excitation(excitation_rates, decay_rates, decay_std=None):
    """
    Fit decay rate vs. excitation rate data with multiple models and plot results
    
    Parameters:
    -----------
    excitation_rates : array-like
        The excitation rates (x values)
    decay_rates : array-like
        The decay rates (y values)
    decay_std : array-like, optional
        Standard deviation of decay rates for error bars
    """
    # Create a smooth x array for plotting the fits
    x_smooth = np.linspace(0, max(excitation_rates) * 1.1, 1000)
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Plot the original data with error bars if provided
    if decay_std is not None:
        plt.errorbar(excitation_rates, decay_rates, yerr=decay_std, fmt='o', 
                    capsize=5, label='Data', color='#1f77b4')
    else:
        plt.scatter(excitation_rates, decay_rates, label='Data', color='#1f77b4')
    
    # Fit with logarithmic function
    try:
        log_params, log_cov = curve_fit(log_function, excitation_rates, decay_rates, 
                                       p0=[0, 0.05, 0.1], maxfev=10000)
        log_y = log_function(x_smooth, *log_params)
        plt.plot(x_smooth, log_y, 'r-', label=f'Log fit: {log_params[0]:.3f} + {log_params[1]:.3f}*log(x + {log_params[2]:.3f})')
        
        # Calculate R² for logarithmic fit
        log_residuals = decay_rates - log_function(excitation_rates, *log_params)
        log_ss_res = np.sum(log_residuals**2)
        log_ss_tot = np.sum((decay_rates - np.mean(decay_rates))**2)
        log_r_squared = 1 - (log_ss_res / log_ss_tot)
        
        print(f"Logarithmic fit parameters: a={log_params[0]:.4f}, b={log_params[1]:.4f}, c={log_params[2]:.4f}")
        print(f"Logarithmic fit R²: {log_r_squared:.4f}")
    except Exception as e:
        print(f"Logarithmic fit failed: {e}")
    
    # Fit with saturation function
    try:
        sat_params, sat_cov = curve_fit(saturation_function, excitation_rates, decay_rates, 
                                       p0=[0.1, 2.0, 0.05], maxfev=10000)
        sat_y = saturation_function(x_smooth, *sat_params)
        plt.plot(x_smooth, sat_y, 'g-', label=f'Saturation fit: {sat_params[2]:.3f} + {sat_params[0]:.3f}*(1-exp(-{sat_params[1]:.3f}*x))')
        
        # Calculate R² for saturation fit
        sat_residuals = decay_rates - saturation_function(excitation_rates, *sat_params)
        sat_ss_res = np.sum(sat_residuals**2)
        sat_ss_tot = np.sum((decay_rates - np.mean(decay_rates))**2)
        sat_r_squared = 1 - (sat_ss_res / sat_ss_tot)
        
        print(f"Saturation fit parameters: a={sat_params[0]:.4f}, b={sat_params[1]:.4f}, c={sat_params[2]:.4f}")
        print(f"Saturation fit R²: {sat_r_squared:.4f}")
    except Exception as e:
        print(f"Saturation fit failed: {e}")
    
    # Fit with Hill function
    try:
        hill_params, hill_cov = curve_fit(hill_function, excitation_rates, decay_rates, 
                                         p0=[0.05, 0.1, 0.5, 1.0], maxfev=10000)
        hill_y = hill_function(x_smooth, *hill_params)
        plt.plot(x_smooth, hill_y, 'm-', label=f'Hill fit: a={hill_params[0]:.3f}, b={hill_params[1]:.3f}, c={hill_params[2]:.3f}, n={hill_params[3]:.3f}')
        
        # Calculate R² for Hill fit
        hill_residuals = decay_rates - hill_function(excitation_rates, *hill_params)
        hill_ss_res = np.sum(hill_residuals**2)
        hill_ss_tot = np.sum((decay_rates - np.mean(decay_rates))**2)
        hill_r_squared = 1 - (hill_ss_res / hill_ss_tot)
        
        print(f"Hill fit parameters: a={hill_params[0]:.4f}, b={hill_params[1]:.4f}, c={hill_params[2]:.4f}, n={hill_params[3]:.4f}")
        print(f"Hill fit R²: {hill_r_squared:.4f}")
    except Exception as e:
        print(f"Hill fit failed: {e}")
    
    # Also try a simple power law fit: y = a * x^b + c
    def power_function(x, a, b, c):
        return a * x**b + c
    
    try:
        power_params, power_cov = curve_fit(power_function, excitation_rates, decay_rates, 
                                          p0=[0.1, 0.5, 0.05], maxfev=10000)
        power_y = power_function(x_smooth, *power_params)
        plt.plot(x_smooth, power_y, 'y-', label=f'Power fit: {power_params[0]:.3f}*x^{power_params[1]:.3f} + {power_params[2]:.3f}')
        
        # Calculate R² for power fit
        power_residuals = decay_rates - power_function(excitation_rates, *power_params)
        power_ss_res = np.sum(power_residuals**2)
        power_ss_tot = np.sum((decay_rates - np.mean(decay_rates))**2)
        power_r_squared = 1 - (power_ss_res / power_ss_tot)
        
        print(f"Power law fit parameters: a={power_params[0]:.4f}, b={power_params[1]:.4f}, c={power_params[2]:.4f}")
        print(f"Power law fit R²: {power_r_squared:.4f}")
    except Exception as e:
        print(f"Power law fit failed: {e}")
    
    plt.xlabel('Excitation rate [/ns]')
    plt.ylabel('Decay rate [1/ns]')
    plt.title('Decay rate vs. excitation rate with different fit models')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Return the fit parameters and R² values in a dictionary
    fit_results = {
        'log_params': log_params if 'log_params' in locals() else None,
        'log_r_squared': log_r_squared if 'log_r_squared' in locals() else None,
        'sat_params': sat_params if 'sat_params' in locals() else None,
        'sat_r_squared': sat_r_squared if 'sat_r_squared' in locals() else None,
        'hill_params': hill_params if 'hill_params' in locals() else None,
        'hill_r_squared': hill_r_squared if 'hill_r_squared' in locals() else None,
        'power_params': power_params if 'power_params' in locals() else None,
        'power_r_squared': power_r_squared if 'power_r_squared' in locals() else None
    }
    
    return fit_results


# Run the fitting function
#fit_results = fit_decay_rate_vs_excitation(results['Excitation Rate'], results['Decay Rate'], results['Decay Rate Std'])
#plt.show()

#####################################################
#####################################################

# Task: relation between #emitters and peak of coherence at deadtime?

def analyze_peak_vs_emitters(emitter_range, deadtime=50, laser_power=400, repeats=100):
    """
    Analyze how the deadtime peak amplitude changes with the number of emitters.
    
    Parameters:
    -----------
    emitter_range : array-like
        Range of emitter numbers to simulate
    deadtime : float
        Deadtime value in ns
    laser_power : float
        Laser power to use for all simulations
    repeats : int
        Number of times to repeat each simulation for error estimation
    
    Returns:
    --------
    dict
        Dictionary containing results of the analysis
    """
    peak_amplitudes = []
    peak_amplitudes_std = []
    exc_rates = []
    exc_rates_std = []
    
    for n_emitters in emitter_range:
        print(f"Simulating with {n_emitters} emitters...")
        
        # Store results for this emitter count
        amplitudes = []
        rates = []
        
        for seed in range(repeats):
            # Run the simulation
            _, _, bins, coherence, exc_rate = sim_whole_detector(
                dead_time=deadtime, 
                nr_emitters=n_emitters,
                laser=laser_power*10**3,
                interval=10**5, 
                bin_size=0.1, 
                eta=1.0,
                seed=seed, 
                nr_steps=1200
            )
            
            # Find the index closest to the deadtime
            deadtime_idx = np.abs(bins - deadtime).argmin()
            
            # Find the maximum value near the deadtime (±5 bins)
            start_idx = max(0, deadtime_idx - 5)
            end_idx = min(len(coherence), deadtime_idx + 5)
            peak_region = coherence[start_idx:end_idx]
            
            # Get the peak amplitude
            peak_amplitude = np.max(peak_region)
            
            # Store results
            amplitudes.append(peak_amplitude)
            rates.append(exc_rate)
        
        # Calculate statistics
        peak_amplitudes.append(np.mean(amplitudes))
        peak_amplitudes_std.append(np.std(amplitudes))
        exc_rates.append(np.mean(rates))
        exc_rates_std.append(np.std(rates))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(emitter_range, peak_amplitudes, yerr=peak_amplitudes_std, 
                 fmt='o-', capsize=5, label='Peak amplitude')
    
    plt.xlabel('Number of emitters')
    plt.ylabel('Deadtime peak amplitude')
    plt.title('Deadtime peak amplitude vs. number of emitters')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Try to fit with different functions
    x_smooth = np.linspace(min(emitter_range), max(emitter_range), 100)
    
    # Linear fit
    try:
        def linear_func(x, a, b):
            return a*x + b
            
        linear_params, linear_cov = curve_fit(linear_func, emitter_range, peak_amplitudes)
        plt.plot(x_smooth, linear_func(x_smooth, *linear_params), 'r-', 
                 label=f'Linear: {linear_params[0]:.3f}*x + {linear_params[1]:.3f}')
                 
        # Calculate R² for linear fit
        linear_residuals = peak_amplitudes - linear_func(emitter_range, *linear_params)
        linear_ss_res = np.sum(linear_residuals**2)
        linear_ss_tot = np.sum((peak_amplitudes - np.mean(peak_amplitudes))**2)
        linear_r_squared = 1 - (linear_ss_res / linear_ss_tot)
    except Exception as e:
        print(f"Linear fit failed: {e}")
        linear_params = None
        linear_r_squared = None
    
    # Power law fit
    try:
        def power_func(x, a, b, c):
            return a * x**b + c
            
        power_params, power_cov = curve_fit(power_func, emitter_range, peak_amplitudes)
        plt.plot(x_smooth, power_func(x_smooth, *power_params), 'g-', 
                 label=f'Power: {power_params[0]:.3f}*x^{power_params[1]:.3f} + {power_params[2]:.3f}')
                 
        # Calculate R² for power fit
        power_residuals = peak_amplitudes - power_func(emitter_range, *power_params)
        power_ss_res = np.sum(power_residuals**2)
        power_ss_tot = np.sum((peak_amplitudes - np.mean(peak_amplitudes))**2)
        power_r_squared = 1 - (power_ss_res / power_ss_tot)
    except Exception as e:
        print(f"Power fit failed: {e}")
        power_params = None
        power_r_squared = None
    
    # Inverse relationship fit (1/x)
    try:
        def inverse_func(x, a, b, c):
            return a / (x + b) + c
            
        inverse_params, inverse_cov = curve_fit(inverse_func, emitter_range, peak_amplitudes)
        plt.plot(x_smooth, inverse_func(x_smooth, *inverse_params), 'm-', 
                 label=f'Inverse: {inverse_params[0]:.3f}/(x+{inverse_params[1]:.3f}) + {inverse_params[2]:.3f}')
                 
        # Calculate R² for inverse fit
        inverse_residuals = peak_amplitudes - inverse_func(emitter_range, *inverse_params)
        inverse_ss_res = np.sum(inverse_residuals**2)
        inverse_ss_tot = np.sum((peak_amplitudes - np.mean(peak_amplitudes))**2)
        inverse_r_squared = 1 - (inverse_ss_res / inverse_ss_tot)
    except Exception as e:
        print(f"Inverse fit failed: {e}")
        inverse_params = None
        inverse_r_squared = None
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Create a second plot with individual coherence curves for different emitter counts
    plt.figure(figsize=(12, 8))
    
    # Choose a few representative emitter counts to visualize
    sample_indices = np.linspace(0, len(emitter_range)-1, min(5, len(emitter_range)), dtype=int)
    
    for idx in sample_indices:
        n_emitters = emitter_range[idx]
        
        # Run a simulation with this number of emitters
        _, _, bins, coherence, _ = sim_whole_detector(
            dead_time=deadtime, 
            nr_emitters=n_emitters,
            laser=laser_power*10**3,
            interval=10**5, 
            bin_size=0.1, 
            eta=1.0,
            seed=0, 
            nr_steps=1200
        )
        
        plt.plot(bins, coherence, '-', label=f'{n_emitters} emitters')
    
    plt.axvline(x=deadtime, color='k', linestyle='--', label='Deadtime')
    plt.xlabel('τΔt [ns]')
    plt.ylabel('g^(2)(τ)')
    plt.title('Coherence function for different numbers of emitters')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Prepare and return results
    results = {
        'n_emitters': emitter_range,
        'peak_amplitudes': peak_amplitudes,
        'peak_amplitudes_std': peak_amplitudes_std,
        'exc_rates': exc_rates,
        'exc_rates_std': exc_rates_std,
        'fits': {
            'linear': {
                'params': linear_params,
                'r_squared': linear_r_squared
            },
            'power': {
                'params': power_params,
                'r_squared': power_r_squared
            },
            'inverse': {
                'params': inverse_params,
                'r_squared': inverse_r_squared
            }
        }
    }
    
    # Print fit quality summary
    print("\nFit Quality Summary:")
    if linear_r_squared is not None:
        print(f"Linear fit R²: {linear_r_squared:.4f}")
    if power_r_squared is not None:
        print(f"Power law fit R²: {power_r_squared:.4f}")
    if inverse_r_squared is not None:
        print(f"Inverse fit R²: {inverse_r_squared:.4f}")
    
    # Determine best fit
    r_squared_values = {
        'Linear': linear_r_squared if linear_r_squared is not None else 0,
        'Power law': power_r_squared if power_r_squared is not None else 0,
        'Inverse': inverse_r_squared if inverse_r_squared is not None else 0
    }
    
    best_model = max(r_squared_values, key=r_squared_values.get)
    print(f"\nBest fit model: {best_model} (R² = {r_squared_values[best_model]:.4f})")
    
    # Physical interpretation
    print("\nPhysical interpretation:")
    if best_model == 'Linear':
        print("The linear relationship suggests that each emitter contributes independently")
        print("to the deadtime peak amplitude. This indicates that emitters are not interacting")
        print("in a way that affects the deadtime behavior.")
    elif best_model == 'Power law':
        exponent = power_params[1]
        if abs(exponent - 1.0) < 0.2:
            print("The power law fit is close to linear, suggesting independent contributions")
            print("from each emitter with minimal interaction effects.")
        elif exponent > 1.0:
            print(f"The power law with exponent > 1 ({exponent:.2f}) suggests that additional")
            print("emitters enhance the deadtime effect super-linearly, possibly due to")
            print("cooperative interactions between emitters.")
        else:
            print(f"The power law with exponent < 1 ({exponent:.2f}) suggests a sub-linear")
            print("increase in deadtime peak amplitude, indicating possible saturation effects")
            print("or competition between emitters.")
    elif best_model == 'Inverse':
        print("The inverse relationship suggests that the deadtime peak amplitude approaches")
        print("a saturation value as the number of emitters increases. This could indicate")
        print("a fundamental limit in the detection system or competition between emitters.")
    
    return results

# Example usage:
# Define a range of emitter numbers to test
#emitter_range = np.array([1, 2, 3, 4, 5])
# Run the analysis
#results = analyze_peak_vs_emitters(emitter_range, deadtime=50, laser_power=300, repeats=100)

# To examine specific relationships in more detail, you can try different ranges
# For example, to focus on lower counts:
# emitter_range_small = np.array([1, 2, 3])
# results_small = analyze_peak_vs_emitters(emitter_range_small, deadtime=50, laser_power=100)

# Or to explore higher counts:
# emitter_range_large = np.arange(5, 21, 5)  # [5, 10, 15, 20]
# results_large = analyze_peak_vs_emitters(emitter_range_large, deadtime=50, laser_power=100)