import os
import json
import datetime
import numpy as np
import h5py
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import time

from ..examples.run_scanning_experiment import run_scanning_experiment
from project.model.localization import localize
from ..evaluation.evaluate_localization import compute_distance_metrics, compute_distance_metrics_batch

from ..evaluation.evaluate_localization_parallel import run_evaluation_parallel


def run_noise_parameter_sweep_parallel(
    # Fixed simulation parameters
    laser_power=200E3,  # 300kWr
    emitter_density=5,  # 5 emitters per μm²
    
    # Noise parameter levels to sweep
    dead_time_levels=[0, 25, 50, 100, 200],  # nanoseconds
    afterpulsing_levels=[0, 0.0005, 0.001, 0.002, 0.005],  # probability
    crosstalk_levels=[0, 0.0007, 0.0014, 0.0028, 0.0056],  # probability
    dark_count_rate_levels=[0, 50, 100, 200, 500],  # counts per pixel per second
    
    # Sweep configuration
    n_runs_per_param=10,
    base_params=None,
    n_workers=None,
    verbose=True
):
    """
    Run parallel noise parameter sweep evaluation.
    
    Tests the effect of different noise sources on detection performance
    while keeping laser power and emitter density fixed.
    
    Parameters:
    - laser_power: Fixed laser power (W)
    - emitter_density: Fixed emitter density (emitters per μm²)
    - dead_time_levels: List of dead time values to test (ns)
    - afterpulsing_levels: List of afterpulsing probabilities to test
    - crosstalk_levels: List of crosstalk probabilities to test
    - dark_count_rate_levels: List of dark count rates to test (cps per pixel)
    - n_runs_per_param: Number of Monte Carlo runs per parameter combination
    - base_params: Base parameters (will be updated with sweep parameters)
    - n_workers: Number of parallel workers (None = auto-detect)
    - verbose: Print progress information
    
    Returns:
    - noise_sweep_results: Dictionary with results for each parameter combination
    """
    
    if base_params is None:
        base_params = {
            'area_size': (0.5, 0.5),
            'positions': (10, 10),
            'detection_efficiency': 1.0,
            'save_data': False,
            'show_plots': False,
            'enable_noise': True,  # Enable noise for this sweep
            # Sensor parameters
            'magnification': 150,
            'pixel_radius': 10.3,
            'spacing': 23,
            'beam_waist': 0.3,
            'max_delay': 1000,
        }
    
    # Define noise parameters to sweep. skip if levels is none
    noise_params = {}
    if dead_time_levels is not None:
        noise_params['dead_time'] = dead_time_levels
    if afterpulsing_levels is not None:
        noise_params['afterpulsing'] = afterpulsing_levels
    if crosstalk_levels is not None:
        noise_params['crosstalk'] = crosstalk_levels
    if dark_count_rate_levels is not None:
        noise_params['dark_count_rate'] = dark_count_rate_levels
    
    noise_sweep_results = {}
    total_combinations = sum(len(levels) for levels in noise_params.values())
    current_combo = 0
    
    if verbose:
        print(f"Starting noise parameter sweep with {total_combinations} combinations...")
        print(f"Fixed parameters:")
        print(f"  Laser power: {laser_power/1000:.0f}kW")
        print(f"  Emitter density: {emitter_density} emitters/μm²")
        print(f"Noise parameters to sweep:")
        for param_name, levels in noise_params.items():
            print(f"  {param_name}: {levels}")
        print(f"Runs per combination: {n_runs_per_param}")
        if n_workers is None:
            print(f"Workers: auto-detect (max {cpu_count()})")
        else:
            print(f"Workers: {n_workers}")
    
    total_start_time = time.time()
    
    # Sweep each noise parameter individually
    for noise_param, levels in noise_params.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Sweeping {noise_param.upper()}")
            print(f"{'='*60}")
        
        for level in levels:
            current_combo += 1
            
            if verbose:
                print(f"\nCombination {current_combo}/{total_combinations}:")
                print(f"  Parameter: {noise_param}")
                print(f"  Level: {level}")
            
            # Create parameters for this combination
            sweep_params = base_params.copy()
            sweep_params['laser_power'] = laser_power
            sweep_params['emitter_density'] = emitter_density
            
            # Set all noise parameters to baseline (0) except the one being swept
            sweep_params['dead_time'] = 0
            sweep_params['afterpulsing'] = 0
            sweep_params['crosstalk'] = 0
            sweep_params['dark_count_rate'] = 0
            
            # Set the current noise parameter to the test level
            sweep_params[noise_param] = level
            
            # Run parallel evaluation for this parameter combination
            combo_start = time.time()
            try:
                results_dir, summary_data = run_evaluation_parallel(
                    n_runs=n_runs_per_param,
                    base_params=sweep_params,
                    save_individual=True,
                    n_workers=n_workers,
                    verbose=verbose
                )
                combo_time = time.time() - combo_start
                
                # Store results with descriptive key
                key = f"{noise_param}_{level}"
                noise_sweep_results[key] = {
                    'noise_parameter': noise_param,
                    'noise_level': level,
                    'laser_power': laser_power,
                    'emitter_density': emitter_density,
                    'summary_data': summary_data,
                }
                
                if verbose:
                    print(f"  Time: {combo_time:.1f}s")
                    if summary_data.get('mean_detection_rate') is not None:
                        print(f"  Detection rate: {summary_data['mean_detection_rate']:.3f} ± {summary_data['std_detection_rate']:.3f}")
                    if summary_data.get('mean_localization_error') is not None:
                        print(f"  Localization error: {summary_data['mean_localization_error']:.3f} ± {summary_data['std_localization_error']:.3f}")
                    if summary_data.get('mean_distance') is not None:
                        print(f"  Mean distance: {summary_data['mean_distance']:.3f} ± {summary_data['std_distance']:.3f}")
                        
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {str(e)}")
                # Store error information
                key = f"{noise_param}_{level}"
                noise_sweep_results[key] = {
                    'noise_parameter': noise_param,
                    'noise_level': level,
                    'laser_power': laser_power,
                    'emitter_density': emitter_density,
                    'error': str(e),
                    'execution_time': time.time() - combo_start
                }
    
    total_time = time.time() - total_start_time
    
    # Save noise sweep summary
    save_noise_sweep_summary_parallel(noise_sweep_results, total_time, laser_power, emitter_density)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"NOISE PARAMETER SWEEP COMPLETED!")
        print(f"{'='*60}")
        print(f"Total time: {total_time/3600:.1f} hours")
        print(f"Results saved for {len(noise_sweep_results)} parameter combinations")
        
        # Print summary statistics
        successful_runs = [k for k, v in noise_sweep_results.items() if 'error' not in v]
        failed_runs = [k for k, v in noise_sweep_results.items() if 'error' in v]
        
        print(f"Successful runs: {len(successful_runs)}")
        if failed_runs:
            print(f"Failed runs: {len(failed_runs)}")
            print(f"Failed combinations: {failed_runs}")
    
    return noise_sweep_results


def save_noise_sweep_summary_parallel(noise_results, total_time, laser_power, emitter_density):
    """
    Save summary of noise parameter sweep results.
    
    Parameters:
    - noise_results: Dictionary with sweep results
    - total_time: Total execution time in seconds
    - laser_power: Fixed laser power used
    - emitter_density: Fixed emitter density used
    """
    import json
    import pandas as pd
    
    # Create results directory
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Prepare summary data
    summary_data = []
    for key, result in noise_results.items():
        if 'error' not in result:
            summary_data.append({
                'combination': key,
                'noise_parameter': result['noise_parameter'],
                'noise_level': result['noise_level'],
                'laser_power': result['laser_power'],
                'emitter_density': result['emitter_density'],
                'mean_detection_rate': result.get('mean_detection_rate'),
                'std_detection_rate': result.get('std_detection_rate'),
                'mean_localization_error': result.get('mean_localization_error'),
                'std_localization_error': result.get('std_localization_error'),
                'mean_distance': result.get('mean_distance'),
                'std_distance': result.get('std_distance'),
            })
    
    # Save complete results as JSON
    import hashlib
    # hash string of 6 characters
    hashstr = hashlib.md5(str(noise_results).encode()).hexdigest()[:6]

    json_path = f"./project/data/noise_sweep_{date}_{hashstr}.json"
    with open(json_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for key, result in noise_results.items():
            json_result = {}
            for k, v in result.items():
                if isinstance(v, (np.integer, np.floating)):
                    json_result[k] = v.item()
                elif isinstance(v, np.ndarray):
                    json_result[k] = v.tolist()
                else:
                    json_result[k] = v
            json_results[key] = json_result
        

        json.dump({'sweep_results': json_results,}, f, indent=4)
    print(f"Complete results JSON saved to: {json_path}")

if __name__ == "__main__":
    #Test parallel parameter sweep

    # Run the noise sweep with default parameters
    noise_results = run_noise_parameter_sweep_parallel(
        n_runs_per_param=10,
        dead_time_levels=None,  # nanoseconds
        afterpulsing_levels= None,  # probability
        crosstalk_levels= [0, 0.0007, 0.0014, 0.0028, 0.0056],  # probability
        dark_count_rate_levels= None,  # counts per pixel per second
        verbose=True,
        
    )