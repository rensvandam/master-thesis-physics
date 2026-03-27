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
    
    #try:
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
        # print(localized_positions_um)
        # print(ground_truth_positions)
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
        
    # except Exception as e:
    #     # Return error information
    #     return {
    #         'run_idx': run_idx,
    #         'ground_truth_positions': [],
    #         'localized_positions': [],
    #         'metrics': None,
    #         'success': False,
    #         'error': str(e)
    #     }


def run_evaluation_parallel(
    n_runs=50,
    base_params=None,
    save_individual=True,
    verbose=True,
    n_workers=None,
    do_localize=True,
    save=True
):
    """
    Run multiple scanning simulations in parallel for localization algorithm evaluation.
    
    Parameters:
    - n_runs: Number of simulation runs to perform
    - base_params: Dict of parameters to pass to run_scanning_experiment()
    - save_individual: Whether to save individual simulation results
    - verbose: Print progress information
    - n_workers: Number of parallel workers (None = auto-detect)
    
    Returns:
    - results_dir: Path to directory containing all results
    - summary_data: Dictionary with aggregated results
    """
    
    # Default parameters for scanning experiment
    if base_params is None:
        base_params = {
            'area_size': (0.5, 0.5),
            'positions': (10, 10),
            'emitter_density': 5,
            'laser_power': 300E3,
            'dwell_time': 0.1,  # milliseconds
            'detection_efficiency': 1.0,
            'enable_noise': False,
            'save_data': False,  # We'll handle saving ourselves
            'show_plots': False  # Don't show plots during batch runs
        }
    
    # Determine number of workers
    if n_workers is None:
        n_workers = min(n_runs, max(1, cpu_count() - 1))
    
    if verbose:
        print(f"Starting parallel evaluation with {n_runs} runs using {n_workers} workers...")
        print(f"Available CPU cores: {cpu_count()}")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if save:
        # Create results directory
        results_dir = Path(f"./project/data/evaluation_{timestamp}")
        results_dir.mkdir(parents=True, exist_ok=True)
    
        if verbose:
            print(f"Results will be saved to: {results_dir}")
    else:
        results_dir = ""
    
    # Save run parameters
    run_config = {
        'n_runs': n_runs,
        'base_params': base_params,
        'n_workers': n_workers,
        'timestamp': timestamp,
        'description': 'Parallel Monte Carlo evaluation of localization algorithm'
    }
    
    if save:
        with open(results_dir / 'config.json', 'w') as f:
            json.dump(run_config, f, indent=2)
    
    # Prepare function for parallel execution
    worker_func = partial(
        run_single_simulation,
        base_params=base_params,
        results_dir=results_dir,
        save_individual=save_individual,
        do_localize=do_localize,
        verbose=True  # Don't print from workers
    )
    
    # Run simulations in parallel
    start_time = time.time()
    
    if verbose:
        print("Starting parallel simulations...")
    
    with Pool(processes=n_workers) as pool:
        # Submit all jobs
        run_indices = list(range(n_runs))
        
        if verbose:
            # Use map with progress updates
            results = []
            for i, result in enumerate(pool.imap(worker_func, run_indices)):
                results.append(result)
                if (i + 1) % max(1, n_runs // 10) == 0:
                    elapsed = time.time() - start_time
                    print(f"  Completed {i + 1}/{n_runs} runs ({elapsed:.1f}s elapsed)")
        else:
            # Simple map without progress updates
            results = pool.map(worker_func, run_indices)
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"Parallel execution completed in {elapsed_time:.1f}s")
    
    # Process results
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    if failed_results:
        print(f"Warning: {len(failed_results)} simulations failed:")
        for failed in failed_results[:5]:  # Show first 5 errors
            print(f"  Run {failed['run_idx']}: {failed['error']}")
        if len(failed_results) > 5:
            print(f"  ... and {len(failed_results) - 5} more")
    
    if not successful_results:
        raise RuntimeError("All simulations failed!")
    
    # Extract data for summary statistics
    all_ground_truth = [r['ground_truth_positions'] for r in successful_results]
    all_localized = [r['localized_positions'] for r in successful_results]
    all_metrics = [r['metrics'] for r in successful_results]
    
    # Calculate summary statistics
    if all_localized[0] != []:
        summary_data = calculate_summary_statistics_parallel(all_metrics, all_localized, all_ground_truth)
        summary_data['n_runs'] = len(successful_results)
        summary_data['n_failed'] = len(failed_results)
        summary_data['n_workers'] = n_workers
        summary_data['execution_time'] = elapsed_time
        summary_data['base_params'] = base_params
    else:
        summary_data = {
            'n_runs': 0,
            'n_failed': len(failed_results),
            'n_workers': n_workers,
            'execution_time': elapsed_time,
            'base_params': base_params,
            'mean_localization_error': 0,
            'std_localization_error': 0,
            'mean_detection_rate': 0,
            'std_detection_rate': 0,
            'mean_distance': 0,
            'std_distance': 0
        }

    print(summary_data)
    
    # Save summary
    if save:
        save_summary(results_dir, summary_data, all_ground_truth, all_localized, all_metrics)
    
    if verbose:
        print(f"\nEvaluation completed!")
        print(f"Successful runs: {len(successful_results)}/{n_runs}")
        print(f"Total time: {elapsed_time:.1f}s")
        print(f"Time per run: {elapsed_time/len(successful_results):.1f}s")
        print(f"Speedup estimate: ~{n_workers:.1f}x")
        print(f"Mean localization error: {summary_data['mean_localization_error']:.3f} ± {summary_data['std_localization_error']:.3f}")
        print(f"Mean detection rate: {summary_data['mean_detection_rate']:.3f} ± {summary_data['std_detection_rate']:.3f}")
        if summary_data['mean_distance'] is not None:
            print(f"Mean distance: {summary_data['mean_distance']:.3f} ± {summary_data['std_distance']:.3f}")
    
    return results_dir, summary_data


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
    """Save summary statistics and aggregated data (same as original)."""
    filename = results_dir / "summary.h5"
    
    with h5py.File(filename, 'w') as f:
        # Summary statistics
        for key, value in summary_data.items():
            if isinstance(value, (list, np.ndarray)):
                f[key] = np.array(value)
            elif not isinstance(value, dict):
                f.attrs[key] = value


def run_parameter_sweep_parallel(
    emitter_densities=[1, 5, 10],
    laser_powers=[200E3],
    dwell_times=[0.1],
    n_runs_per_param=10,
    base_params=None,
    n_workers=None,
    do_localize=True,
    save=True,
    verbose=True
):
    """
    Run parallel parameter sweep evaluation.
    
    Parameters:
    - emitter_densities: List of emitter densities to test
    - laser_powers: List of laser powers to test
    - n_runs_per_param: Number of Monte Carlo runs per parameter combination
    - base_params: Base parameters (will be updated with sweep parameters)
    - n_workers: Number of parallel workers (None = auto-detect)
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
    
    sweep_results = {}
    if len(laser_powers)>1:
        total_combinations = len(emitter_densities) * len(laser_powers)
    elif len(dwell_times)>1:
        total_combinations = len(emitter_densities) * len(dwell_times)
    else:
        total_combinations = len(emitter_densities)
    #total_combinations = len(emitter_densities) * len(laser_powers)
    current_combo = 0
    
    if verbose:
        print(f"Starting parallel parameter sweep with {total_combinations} combinations...")
        print(f"Emitter densities: {emitter_densities}")
        print(f"Laser powers: {laser_powers}")
        print(f"Dwell times: {dwell_times}")
        print(f"Runs per combination: {n_runs_per_param}")
        if n_workers is None:
            print(f"Workers: auto-detect (max {cpu_count()})")
        else:
            print(f"Workers: {n_workers}")
    
    total_start_time = time.time()
    
    for density in emitter_densities:
        if len(laser_powers) > 1:
            for power in laser_powers:
                current_combo += 1
                
                if verbose:
                    print(f"\nCombination {current_combo}/{total_combinations}:")
                    print(f"  Emitter density: {density}")
                    print(f"  Laser power: {power}")
                
                # Update parameters for this combination
                sweep_params = base_params.copy()
                sweep_params['emitter_density'] = density
                sweep_params['laser_power'] = power
                
                # Run parallel evaluation for this parameter combination
                combo_start = time.time()
                results_dir, summary_data = run_evaluation_parallel(
                    n_runs=n_runs_per_param,
                    base_params=sweep_params,
                    save_individual=save,
                    n_workers=n_workers,
                    do_localize=do_localize,
                    save=save,
                    verbose=verbose
                )
                combo_time = time.time() - combo_start
                
                # Store results with descriptive key
                key = f"density_{density}_power_{power/1000:.0f}k"
                sweep_results[key] = {
                    'emitter_density': density,
                    'laser_power': power,
                    'results_dir': results_dir,
                    'summary_data': summary_data,
                    'execution_time': combo_time,
                    'mean_detection_rate': summary_data['mean_detection_rate'],
                    'std_detection_rate': summary_data['std_detection_rate'],
                    'mean_localization_error': summary_data['mean_localization_error'],
                    'std_localization_error': summary_data['std_localization_error'],
                    'mean_distance': summary_data['mean_distance'],
                    'std_distance': summary_data['std_distance']
                }
                
                if verbose:
                    print(f"  Time: {combo_time:.1f}s")
                    print(f"  Detection rate: {summary_data['mean_detection_rate']:.3f} ± {summary_data['std_detection_rate']:.3f}")
                    print(f"  Localization error: {summary_data['mean_localization_error']:.3f} ± {summary_data['std_localization_error']:.3f}")
                    if summary_data['mean_distance'] is not None:
                        print(f"  Mean distance: {summary_data['mean_distance']:.3f} ± {summary_data['std_distance']:.3f}")
        elif len(dwell_times) > 1:
            for dtime in dwell_times:
                current_combo += 1
                
                if verbose:
                    print(f"\nCombination {current_combo}/{total_combinations}:")
                    print(f"  Emitter density: {density}")
                    print(f"  Dwell time: {dtime}")
                
                # Update parameters for this combination
                sweep_params = base_params.copy()
                sweep_params['emitter_density'] = density
                sweep_params['dwell_time'] = dtime
                
                # Run parallel evaluation for this parameter combination
                combo_start = time.time()
                results_dir, summary_data = run_evaluation_parallel(
                    n_runs=n_runs_per_param,
                    base_params=sweep_params,
                    save_individual=save,
                    n_workers=n_workers,
                    save=save,
                    verbose=verbose
                )
                combo_time = time.time() - combo_start
                
                # Store results with descriptive key
                key = f"density_{density}_dwelltime_{dtime}"
                sweep_results[key] = {
                    'emitter_density': density,
                    'dwell_time': dtime,
                    'results_dir': results_dir,
                    'summary_data': summary_data,
                    'execution_time': combo_time,
                    'mean_detection_rate': summary_data['mean_detection_rate'],
                    'std_detection_rate': summary_data['std_detection_rate'],
                    'mean_localization_error': summary_data['mean_localization_error'],
                    'std_localization_error': summary_data['std_localization_error'],
                    'mean_distance': summary_data['mean_distance'],
                    'std_distance': summary_data['std_distance']
                }
                
                if verbose:
                    print(f"  Time: {combo_time:.1f}s")
                    print(f"  Detection rate: {summary_data['mean_detection_rate']:.3f} ± {summary_data['std_detection_rate']:.3f}")
                    print(f"  Localization error: {summary_data['mean_localization_error']:.3f} ± {summary_data['std_localization_error']:.3f}")
                    if summary_data['mean_distance'] is not None:
                        print(f"  Mean distance: {summary_data['mean_distance']:.3f} ± {summary_data['std_distance']:.3f}")
        else:
            current_combo += 1
            
            if verbose:
                print(f"\nCombination {current_combo}/{total_combinations}:")
                print(f"  Emitter density: {density}")
                print(f"  Laser power: {laser_powers[0]/1000:.0f}kW/cm2")  # Use first power if only one
            
            # Update parameters for this combination
            sweep_params = base_params.copy()
            sweep_params['emitter_density'] = density
            sweep_params['laser_power'] = laser_powers[0] # Use first power if only one
            
            # Run parallel evaluation for this parameter combination
            combo_start = time.time()
            results_dir, summary_data = run_evaluation_parallel(
                n_runs=n_runs_per_param,
                base_params=sweep_params,
                save_individual=save,
                n_workers=n_workers,
                save=save,
                verbose=verbose
            )
            combo_time = time.time() - combo_start
            
            # Store results with descriptive key
            key = f"density_{density}"
            sweep_results[key] = {
                'emitter_density': density,
                'results_dir': results_dir,
                'summary_data': summary_data,
                'execution_time': combo_time,
                'mean_detection_rate': summary_data['mean_detection_rate'],
                'std_detection_rate': summary_data['std_detection_rate'],
                'mean_localization_error': summary_data['mean_localization_error'],
                'std_localization_error': summary_data['std_localization_error'],
                'mean_distance': summary_data['mean_distance'],
                'std_distance': summary_data['std_distance']
            }
            
            if verbose:
                print(f"  Time: {combo_time:.1f}s")
                print(f"  Detection rate: {summary_data['mean_detection_rate']:.3f} ± {summary_data['std_detection_rate']:.3f}")
                print(f"  Localization error: {summary_data['mean_localization_error']:.3f} ± {summary_data['std_localization_error']:.3f}")
                if summary_data['mean_distance'] is not None:
                    print(f"  Mean distance: {summary_data['mean_distance']:.3f} ± {summary_data['std_distance']:.3f}")
    total_time = time.time() - total_start_time
    
    # Save sweep summary
    if save:
        save_sweep_summary_parallel(sweep_results, total_time)
    
    if verbose:
        print(f"\nParameter sweep completed!")
        print(f"Total time: {total_time/3600:.1f} hours")
        print(f"Results saved for {len(sweep_results)} parameter combinations")
    
    return sweep_results


def save_sweep_summary_parallel(sweep_results, total_time):
    """Save parameter sweep summary to file (parallel version)."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./project/data/parameter_sweep_parallel_{timestamp}.json"
    
    # Create summary for JSON saving
    json_summary = {
        'total_execution_time': total_time,
        'timestamp': timestamp,
        'method': 'parallel',
        'results': {}
    }
    
    for key, result in sweep_results.items():
        if 'laser_power' in result.keys():
            json_summary['results'][key] = {
                'emitter_density': result['emitter_density'],
                'laser_power': result['laser_power'],
                'results_dir': str(result['results_dir']),
                'execution_time': result['execution_time'],
                'mean_detection_rate': result['mean_detection_rate'],
                'std_detection_rate': result['std_detection_rate'],
                'mean_localization_error': result['mean_localization_error'],
                'std_localization_error': result['std_localization_error'],
                'mean_distance': result['mean_distance'],
                'std_distance': result['std_distance']
            }
        elif 'dwell_time' in result.keys():
            json_summary['results'][key] = {
                'emitter_density': result['emitter_density'],
                'dwell_time': result['dwell_time'],
                'results_dir': str(result['results_dir']),
                'execution_time': result['execution_time'],
                'mean_detection_rate': result['mean_detection_rate'],
                'std_detection_rate': result['std_detection_rate'],
                'mean_localization_error': result['mean_localization_error'],
                'std_localization_error': result['std_localization_error'],
                'mean_distance': result['mean_distance'],
                'std_distance': result['std_distance']
            }
        else:
            json_summary['results'][key] = {
                'emitter_density': result['emitter_density'],
                'results_dir': str(result['results_dir']),
                'execution_time': result['execution_time'],
                'mean_detection_rate': result['mean_detection_rate'],
                'std_detection_rate': result['std_detection_rate'],
                'mean_localization_error': result['mean_localization_error'],
                'std_localization_error': result['std_localization_error'],
                'mean_distance': result['mean_distance'],
                'std_distance': result['std_distance']
            }
    
    with open(filename, 'w') as f:
        json.dump(json_summary, f, indent=2)
    
    print(f"Parallel parameter sweep summary saved to: {filename}")


if __name__ == "__main__":
    # Test parallel evaluation
    # results_dir, summary = run_evaluation_parallel(
    #     n_runs=20,
    #     n_workers=4,
    #     verbose=True
    # )
    # print(f"\nResults saved to: {results_dir}")
    
    #Test parallel parameter sweep
    # base_params = {
    #     'area_size': (20, 20),
    #     'positions': (400, 400),
    #     'detection_efficiency': 1.0,
    #     'enable_noise': True,
    #     'dark_count_rate': 100,
    #     'crosstalk': 0.001,
    #     'afterpulsing': 0.0014,
    #     'dead_time': 50,
    #     'dwell_time':0.1, # milliseconds (ms)
    #     'save_data': False,
    #     'show_plots': False
    # }

    # base_params = {
    #     'area_size': (5, 5),
    #     'positions': (100, 100),
    #     'detection_efficiency': 1.0,
    #     'enable_noise': True,
    #     'dark_count_rate': 100,
    #     'crosstalk': 0.001,
    #     'afterpulsing': 0.0014,
    #     'dead_time': 50,
    #     'dwell_time':0.1, # milliseconds (ms)
    #     'save_data': False,
    #     'show_plots': False
    # }

    # base_params = {
    #     'area_size': (0.5, 0.5),
    #     'positions': (10, 10),
    #     'emitter_density': 9,
    #     'detection_efficiency': 1.0,
    #     'enable_noise': False,
    #     'save_data': False,
    #     'show_plots': False
    # }

    base_params = {
        'area_size': (2, 2),
        'positions': (40, 40),
        'detection_efficiency': 1.0,
        'enable_noise': True,
        'dark_count_rate': 100,
        'crosstalk': 0.001,
        'afterpulsing': 0.0014,
        'dead_time': 50,
        'dwell_time': 2,#0.1,#0.3, # milliseconds (ms)
        'save_data': False,
        'show_plots': True,
        #'emitters_manual': [(0,0)]
    }

    sweep_results = run_parameter_sweep_parallel(
        emitter_densities=[1,2,3,4,5,6,7,8,9,10],#[1, 5, 10],
        laser_powers=[10E3],  # [200E3, 300E3],
        #dwell_times = [0.5],
        n_runs_per_param=100,
        base_params=base_params,
        do_localize=False,
        save=True,
        verbose=True
    )