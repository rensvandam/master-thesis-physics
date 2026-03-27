import os
import json
import datetime
import numpy as np
import h5py
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import time

from .run_scanning_experiment import run_scanning_experiment
from ..model.localization import localize
from .evaluate_localization import compute_distance_metrics, compute_distance_metrics_batch
from ..model.helper_functions import transform_coordinates


def run_single_simulation(run_idx, base_params, results_dir, save_individual=True, do_localize=True, verbose=False):
    """
    Run a single simulation - designed to be called by parallel workers.
    
    Parameters:
    - run_idx: Index of this simulation run
    - base_params: Base parameters for the simulation
    - results_dir: Directory to save results
    - save_individual: Whether to save individual run data
    - do_localize: Whether to run localization on the simulated data
    - verbose: Print progress (usually False for parallel workers)
    
    Returns:
    - dict with simulation results and metrics
    """
    
    # Create unique parameters for this run
    current_params = base_params.copy()
    current_params['seed'] = run_idx + 1000 + 2  # Ensure unique seeds

    # Run simulation
    setup, photon_count_map, G2_diff_map, nr_emitters_map, metadata = run_scanning_experiment(**current_params)
    
    # Extract ground truth positions
    ground_truth_positions = metadata['emitter_positions']
    
    # Run localization
    if do_localize:
        localization = localize(
            photon_count_map.T, 
            G2_diff_map.T, 
            nr_emitters_map.T, 
            metadata, 
            plot=current_params['show_plots'], 
            psf_file='project/data/psf.json', 
            reg_weight=1000, 
            verbose=False
        )
    
        # Extract localized positions
        emitters_pos = localization['emitters']
        localized_positions = []
        for pos in emitters_pos:
            localized_positions.append((pos['x'], pos['y']))
        fit_error = localization['RSS']
    
        localized_positions_um = [transform_coordinates(
            pos[0], pos[1], metadata['area_size'], metadata['positions'], metadata['pixel_size'], direction='to_physical') for pos in localized_positions]
        
        # Compute distance metrics
        distance_metrics = compute_distance_metrics(ground_truth_positions, localized_positions_um)
    
        # Prepare metrics
        run_metrics = {
            'localization_error': fit_error,
            'n_true': len(ground_truth_positions),
            'n_detected': len(localized_positions),
            'detection_rate': len(localized_positions) / len(ground_truth_positions) if ground_truth_positions else 0.0,
            'mean_distance': distance_metrics['mean_distance'] if distance_metrics else None,
            'std_distance': distance_metrics['std_distance'] if distance_metrics else None,
        }
    else:
        localized_positions_um = []
        run_metrics = {
            'localization_error': None,
            'n_true': len(ground_truth_positions),
            'n_detected': 0,
            'detection_rate': 0.0,
            'mean_distance': None,
            'std_distance': None
        }
    
    # Save individual run data if requested
    if save_individual:
        # Use process-safe filename to avoid conflicts
        pid = os.getpid()
        filename = results_dir / f"run_{run_idx:03d}_pid_{pid}.h5"
        
        save_individual_run(
            filename,
            photon_count_map, 
            G2_diff_map, 
            nr_emitters_map,
            ground_truth_positions,
            localized_positions_um,
            run_metrics,
            metadata,
            run_idx
        )
    
    # Return results for aggregation
    return {
        'run_idx': run_idx,
        'ground_truth_positions': ground_truth_positions,
        'localized_positions': localized_positions_um,
        'metrics': run_metrics,
        'success': True,
        'error': None
    }


def run_evaluation_for_density(density_params):
    """
    Run evaluation for a single density parameter - designed for density-level parallelization.
    
    Parameters:
    - density_params: Dict containing:
        - 'density': emitter density value
        - 'base_params': base parameters
        - 'n_runs': number of runs for this density
        - 'save_individual': whether to save individual runs
        - 'do_localize': whether to run localization
        - 'save': whether to save results
        - 'verbose': verbosity level
    
    Returns:
    - dict with results for this density
    """
    
    density = density_params['density']
    base_params = density_params['base_params'].copy()
    n_runs = density_params['n_runs']
    save_individual = density_params['save_individual']
    do_localize = density_params['do_localize']
    save = density_params['save']
    verbose = density_params['verbose']
    
    # Update parameters for this density
    base_params['emitter_density'] = density
    
    # Handle additional sweep parameters
    if 'laser_power' in density_params:
        base_params['laser_power'] = density_params['laser_power']
        param_key = f"density_{density}_power_{density_params['laser_power']/1000:.0f}k"
    elif 'dwell_time' in density_params:
        base_params['dwell_time'] = density_params['dwell_time']
        param_key = f"density_{density}_dwelltime_{density_params['dwell_time']}"
    else:
        param_key = f"density_{density}"
    
    if verbose:
        print(f"Processing density {density} (PID: {os.getpid()})")
    
    start_time = time.time()
    
    # Create results directory for this density
    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(f"./project/data/evaluation_{param_key}_{timestamp}")
        results_dir.mkdir(parents=True, exist_ok=True)
    else:
        results_dir = Path("")
    
    # Run simulations for this density (sequentially within this process)
    results = []
    for run_idx in range(n_runs):
        try:
            result = run_single_simulation(
                run_idx=run_idx,
                base_params=base_params,
                results_dir=results_dir,
                save_individual=save_individual,
                do_localize=do_localize,
                verbose=False
            )
            results.append(result)
        except Exception as e:
            results.append({
                'run_idx': run_idx,
                'ground_truth_positions': [],
                'localized_positions': [],
                'metrics': None,
                'success': False,
                'error': str(e)
            })
    
    # Process results
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    if not successful_results:
        return {
            'param_key': param_key,
            'density': density,
            'success': False,
            'error': f"All {n_runs} runs failed",
            'execution_time': time.time() - start_time
        }
    
    # Calculate summary statistics
    all_ground_truth = [r['ground_truth_positions'] for r in successful_results]
    all_localized = [r['localized_positions'] for r in successful_results]
    all_metrics = [r['metrics'] for r in successful_results]
    
    if all_localized[0] != []:
        summary_data = calculate_summary_statistics_parallel(all_metrics, all_localized, all_ground_truth)
        summary_data['n_runs'] = len(successful_results)
        summary_data['n_failed'] = len(failed_results)
        summary_data['execution_time'] = time.time() - start_time
        summary_data['base_params'] = base_params
    else:
        summary_data = {
            'n_runs': len(successful_results),
            'n_failed': len(failed_results),
            'execution_time': time.time() - start_time,
            'base_params': base_params,
            'mean_localization_error': 0,
            'std_localization_error': 0,
            'mean_detection_rate': 0,
            'std_detection_rate': 0,
            'mean_distance': 0,
            'std_distance': 0
        }
    
    # Save summary for this density
    if save:
        save_summary(results_dir, summary_data, all_ground_truth, all_localized, all_metrics)
    
    execution_time = time.time() - start_time
    
    if verbose:
        print(f"Density {density} completed in {execution_time:.1f}s (PID: {os.getpid()})")
    
    # Prepare return data
    result_data = {
        'param_key': param_key,
        'density': density,
        'results_dir': results_dir,
        'summary_data': summary_data,
        'execution_time': execution_time,
        'success': True,
        'mean_detection_rate': summary_data['mean_detection_rate'],
        'std_detection_rate': summary_data['std_detection_rate'],
        'mean_localization_error': summary_data['mean_localization_error'],
        'std_localization_error': summary_data['std_localization_error'],
        'mean_distance': summary_data['mean_distance'],
        'std_distance': summary_data['std_distance']
    }
    
    # Add sweep parameter to result
    if 'laser_power' in density_params:
        result_data['laser_power'] = density_params['laser_power']
    elif 'dwell_time' in density_params:
        result_data['dwell_time'] = density_params['dwell_time']
    
    return result_data


def run_parameter_sweep_density_parallel(
    emitter_densities=[1, 5, 10],
    laser_powers=[200E3],
    dwell_times=[0.1],
    n_runs_per_param=3,
    base_params=None,
    n_workers=None,
    do_localize=True,
    save=True,
    verbose=True
):
    """
    Run parameter sweep with parallelization over emitter densities.
    
    This approach is optimal when you want:
    - Few runs per parameter (1-3)
    - Many parameter combinations to test
    - Maximum utilization of available CPU cores
    
    Parameters:
    - emitter_densities: List of emitter densities to test
    - laser_powers: List of laser powers to test  
    - dwell_times: List of dwell times to test
    - n_runs_per_param: Number of Monte Carlo runs per parameter combination (keep low, e.g., 1-3)
    - base_params: Base parameters
    - n_workers: Number of parallel workers (None = auto-detect)
    - do_localize: Whether to run localization
    - save: Whether to save results
    - verbose: Print progress information
    
    Returns:
    - sweep_results: Dictionary with results for each parameter combination
    """
    
    if base_params is None:
        base_params = {
            'area_size': (0.5, 0.5),
            'positions': (10, 10),
            'detection_efficiency': 1.0,
            'enable_noise': False,
            'save_data': False,
            'show_plots': False
        }
    
    # Prepare all parameter combinations
    param_combinations = []
    
    if len(laser_powers) > 1:
        # Sweep over density and laser power
        for density in emitter_densities:
            for power in laser_powers:
                param_combinations.append({
                    'density': density,
                    'laser_power': power,
                    'base_params': base_params,
                    'n_runs': n_runs_per_param,
                    'save_individual': save,
                    'do_localize': do_localize,
                    'save': save,
                    'verbose': False  # Reduce verbosity for workers
                })
    elif len(dwell_times) > 1:
        # Sweep over density and dwell time
        for density in emitter_densities:
            for dtime in dwell_times:
                param_combinations.append({
                    'density': density,
                    'dwell_time': dtime,
                    'base_params': base_params,
                    'n_runs': n_runs_per_param,
                    'save_individual': save,
                    'do_localize': do_localize,
                    'save': save,
                    'verbose': False
                })
    else:
        # Sweep over density only
        for density in emitter_densities:
            param_combinations.append({
                'density': density,
                'base_params': base_params,
                'n_runs': n_runs_per_param,
                'save_individual': save,
                'do_localize': do_localize,
                'save': save,
                'verbose': False
            })
    
    total_combinations = len(param_combinations)
    
    # Determine number of workers
    if n_workers is None:
        n_workers = min(total_combinations, cpu_count())
    
    if verbose:
        print(f"Starting density-parallel parameter sweep with {total_combinations} combinations...")
        print(f"Emitter densities: {emitter_densities}")
        if len(laser_powers) > 1:
            print(f"Laser powers: {laser_powers}")
        if len(dwell_times) > 1:
            print(f"Dwell times: {dwell_times}")
        print(f"Runs per combination: {n_runs_per_param}")
        print(f"Workers: {n_workers}")
        print(f"Available CPU cores: {cpu_count()}")
        print(f"Parallelization strategy: Each worker processes one density combination")
    
    total_start_time = time.time()
    
    # Run parameter combinations in parallel
    if verbose:
        print("Starting parallel parameter sweep...")
    
    with Pool(processes=n_workers) as pool:
        if verbose:
            # Use map with progress updates
            results = []
            for i, result in enumerate(pool.imap(run_evaluation_for_density, param_combinations)):
                results.append(result)
                if result['success']:
                    print(f"  Completed {i + 1}/{total_combinations}: {result['param_key']} "
                          f"({result['execution_time']:.1f}s, detection_rate={result['mean_detection_rate']:.3f})")
                else:
                    print(f"  Failed {i + 1}/{total_combinations}: {result['param_key']} - {result['error']}")
        else:
            # Simple map without progress updates
            results = pool.map(run_evaluation_for_density, param_combinations)
    
    total_time = time.time() - total_start_time
    
    # Process results
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    if failed_results:
        print(f"Warning: {len(failed_results)} parameter combinations failed:")
        for failed in failed_results:
            print(f"  {failed['param_key']}: {failed['error']}")
    
    # Convert to sweep_results format
    sweep_results = {}
    for result in successful_results:
        sweep_results[result['param_key']] = result
    
    # Save sweep summary
    if save:
        save_sweep_summary_density_parallel(sweep_results, total_time, n_runs_per_param)
    
    if verbose:
        print(f"\nDensity-parallel parameter sweep completed!")
        print(f"Successful combinations: {len(successful_results)}/{total_combinations}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average time per combination: {total_time/len(successful_results):.1f}s")
        print(f"Estimated speedup vs sequential: ~{n_workers:.1f}x")
        
        # Print summary of results
        if successful_results:
            print("\nResults summary:")
            for result in successful_results:
                print(f"  {result['param_key']}: detection_rate={result['mean_detection_rate']:.3f}±{result['std_detection_rate']:.3f}, "
                      f"loc_error={result['mean_localization_error']:.3f}±{result['std_localization_error']:.3f}")
    
    return sweep_results


def save_sweep_summary_density_parallel(sweep_results, total_time, n_runs_per_param):
    """Save parameter sweep summary optimized for density parallelization."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./project/data/parameter_sweep_density_parallel_{timestamp}.json"
    
    # Create summary for JSON saving
    json_summary = {
        'total_execution_time': total_time,
        'timestamp': timestamp,
        'method': 'density_parallel',
        'n_runs_per_param': n_runs_per_param,
        'n_combinations': len(sweep_results),
        'results': {}
    }
    
    for key, result in sweep_results.items():
        result_summary = {
            'emitter_density': result['density'],
            'results_dir': str(result['results_dir']),
            'execution_time': result['execution_time'],
            'mean_detection_rate': result['mean_detection_rate'],
            'std_detection_rate': result['std_detection_rate'],
            'mean_localization_error': result['mean_localization_error'],
            'std_localization_error': result['std_localization_error'],
            'mean_distance': result['mean_distance'],
            'std_distance': result['std_distance']
        }
        
        # Add sweep parameter if present
        if 'laser_power' in result:
            result_summary['laser_power'] = result['laser_power']
        if 'dwell_time' in result:
            result_summary['dwell_time'] = result['dwell_time']
            
        json_summary['results'][key] = result_summary
    
    with open(filename, 'w') as f:
        json.dump(json_summary, f, indent=2)
    
    print(f"Density-parallel parameter sweep summary saved to: {filename}")


def calculate_summary_statistics_parallel(all_metrics, all_localized, all_ground_truth):
    """Calculate summary statistics across all runs (parallel version)."""
    all_errors = []
    detection_rates = []
    
    for metrics in all_metrics:
        all_errors.append(metrics['localization_error'])
        detection_rates.append(metrics['detection_rate'])
    
    # Compute distance metrics
    distance_metrics = compute_distance_metrics_batch(all_ground_truth, all_localized)
    
    summary = {
        'mean_localization_error': np.mean(all_errors),
        'std_localization_error': np.std(all_errors),
        'mean_detection_rate': np.mean(detection_rates),
        'std_detection_rate': np.std(detection_rates),
        'mean_distance': distance_metrics['batch_mean_distance'] if distance_metrics else None,
        'std_distance': distance_metrics['batch_std_distance'] if distance_metrics else None,
        'all_localization_errors': all_errors,
        'all_detection_rates': detection_rates
    }
    
    return summary


def save_individual_run(filename, photon_count_map, G2_diff_map, nr_emitters_map,
                       ground_truth, localized, metrics, metadata, run_idx):
    """Save data from individual simulation run (parallel-safe version)."""
    
    with h5py.File(filename, 'w') as f:
        # Simulation data
        f['photon_count_map'] = photon_count_map
        f['G2_diff_map'] = G2_diff_map
        f['nr_emitters_map'] = nr_emitters_map
        
        # Ground truth and results
        f['ground_truth_positions'] = np.array(ground_truth)
        f['localized_positions'] = np.array(localized)
        
        # Metrics
        f['localization_error'] = np.array(metrics['localization_error'])
        f.attrs['detection_rate'] = metrics['detection_rate']
        f.attrs['n_true'] = metrics['n_true']
        f.attrs['n_detected'] = metrics['n_detected']
        
        # Distance metrics
        if metrics['mean_distance'] is not None:
            f.attrs['mean_distance'] = metrics['mean_distance']
            f.attrs['std_distance'] = metrics['std_distance']
        
        # Metadata
        f.attrs['run_idx'] = run_idx
        f.attrs['process_id'] = os.getpid()
        for key, value in metadata.items():
            if key != 'emitter_positions':  # Already saved separately
                f.attrs[key] = value


def save_summary(results_dir, summary_data, all_ground_truth, all_localized, all_metrics):
    """Save summary statistics and aggregated data."""
    filename = results_dir / "summary.h5"
    
    with h5py.File(filename, 'w') as f:
        # Summary statistics
        for key, value in summary_data.items():
            if isinstance(value, (list, np.ndarray)):
                f[key] = np.array(value)
            elif not isinstance(value, dict):
                f.attrs[key] = value


if __name__ == "__main__":
    # Example usage for density-parallel parameter sweep
    base_params = {
        'area_size': (4, 4),
        'positions': (80, 80),
        'detection_efficiency': 1.0,
        'enable_noise': True,
        'dark_count_rate': 100,
        'crosstalk': 0.001,
        'afterpulsing': 0.0014,
        'dead_time': 50,
        'dwell_time': 2,
        'save_data': False,
        'show_plots': False,
    }

    # Run density-parallel parameter sweep
    sweep_results = run_parameter_sweep_density_parallel(
        emitter_densities=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Many densities
        laser_powers=[10E3],  # Single laser power
        n_runs_per_param=2,  # Few runs per parameter
        base_params=base_params,
        do_localize=False,
        save=True,
        n_workers=None,  # Auto-detect (use all available cores)
        verbose=True
    )
    
    print(f"\nCompleted parameter sweep with {len(sweep_results)} combinations")