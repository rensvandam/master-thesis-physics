import os
import json
import datetime
import numpy as np
import h5py
from pathlib import Path
from scipy.spatial.distance import cdist

from ..examples.run_scanning_experiment import run_scanning_experiment
from project.model.localization import localize
from project.model.helper_functions import transform_coordinates


def run_evaluation(
    n_runs=50,
    base_params=None,
    save_individual=True,
    verbose=True
):
    """
    Run multiple scanning simulations for localization algorithm evaluation.
    
    Parameters:
    - n_runs: Number of simulation runs to perform
    - base_params: Dict of parameters to pass to run_scanning_experiment()
    - save_individual: Whether to save individual simulation results
    - verbose: Print progress information
    
    Returns:
    - results_dir: Path to directory containing all results
    - summary_data: Dictionary with aggregated results
    """
    
    # Default parameters for scanning experiment
    if base_params is None:
        base_params = {
            'area_size': (0.5, 0.5), # Area size in micrometers
            'positions': (10, 10),
            'emitter_density': 5,
            'laser_power': 300E3,
            'detection_efficiency': 1.0,
            'enable_noise': False,
            'save_data': False,  # We'll handle saving ourselves
            'show_plots': False  # Don't show plots during batch runs
        }
    
    # Create results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"./project/data/evaluation_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Starting evaluation with {n_runs} runs...")
        print(f"Results will be saved to: {results_dir}")
    
    # Save run parameters
    run_config = {
        'n_runs': n_runs,
        'base_params': base_params,
        'timestamp': timestamp,
        'description': 'Monte Carlo evaluation of localization algorithm'
    }
    
    with open(results_dir / 'config.json', 'w') as f:
        json.dump(run_config, f, indent=2)
    
    # Storage for summary statistics
    all_ground_truth = []
    all_localized = []
    all_metrics = []
    
    # Run simulations
    for run_idx in range(n_runs):
        if verbose and (run_idx + 1) % 10 == 0:
            print(f"  Completed {run_idx + 1}/{n_runs} runs")
        
        # Use different seed for each run
        current_params = base_params.copy()
        current_params['seed'] = run_idx + 1000  # Offset to avoid seed=0 issues
        
        # Run simulation
        setup, photon_count_map, G2_diff_map, nr_emitters_map, metadata = run_scanning_experiment(**current_params)
        
        # Extract ground truth positions
        ground_truth_positions = metadata['emitter_positions']
        
        localization = localize(photon_count_map.T, G2_diff_map.T, nr_emitters_map.T, metadata, plot = False, psf_file = 'project/data/psf.json', reg_weight=1000, verbose=False)
        emitters_pos = localization['emitters']
        localized_positions = []
        for pos in emitters_pos: # Change format of array
            localized_positions.append((pos['x'], pos['y']))
        fit_error = localization['RSS']
    	
        
        print("Ground truth and localized positions for this run:")
        print(ground_truth_positions)
        print(localized_positions)

        # Change localized_positions to micrometers
        #localized_positions_um = [(pos[0] * metadata['pixel_size'] - metadata['area_size'][0]/2, pos[1] * metadata['pixel_size'] - metadata['area_size'][0]/2) for pos in localized_positions]
        localized_positions_um = [transform_coordinates(
            pos[0], pos[1], metadata['area_size'], metadata['positions'], metadata['pixel_size'], direction='to_physical') for pos in localized_positions]

        distance_metrics = compute_distance_metrics(ground_truth_positions, localized_positions_um)



        if verbose:
            print("The localized positions for this run are:", localized_positions)
            print("The ground truth positions for this run are:", ground_truth_positions)

        run_metrics = {
        'localization_error': fit_error,
        'n_true': len(ground_truth_positions),
        'n_detected': len(localized_positions),
        'detection_rate': len(localized_positions) / len(ground_truth_positions) if ground_truth_positions else 0.0,
        'mean_distance': distance_metrics['mean_distance'] if distance_metrics else None,
        'std_distance': distance_metrics['std_distance'] if distance_metrics else None,
        }
        
        # Store results
        all_ground_truth.append(ground_truth_positions)
        all_localized.append(localized_positions_um)
        all_metrics.append(run_metrics)
        
        # Save individual run data if requested
        if save_individual:
            save_individual_run(
                results_dir, 
                run_idx, 
                photon_count_map, 
                G2_diff_map, 
                nr_emitters_map,
                ground_truth_positions,
                localized_positions,
                run_metrics,
                metadata
            )
    
    # Calculate summary statistics
    summary_data = calculate_summary_statistics(all_metrics, all_localized, all_ground_truth)
    summary_data['n_runs'] = n_runs
    summary_data['base_params'] = base_params
    
    # Save summary
    save_summary(results_dir, summary_data, all_ground_truth, all_localized, all_metrics)
    
    if verbose:
        print(f"\nEvaluation completed!")
        print(f"Mean localization error: {summary_data['mean_localization_error']:.3f} ± {summary_data['std_localization_error']:.3f}")
        print(f"Mean detection rate: {summary_data['mean_detection_rate']:.3f} ± {summary_data['std_detection_rate']:.3f}")
        print(f"Mean distance: {summary_data['mean_distance']:.3f} ± {summary_data['std_distance']:.3f}")
    
    return results_dir, summary_data

def calculate_summary_statistics(all_metrics, all_localized, all_ground_truth):
    """Calculate summary statistics across all runs."""
    all_errors = []
    detection_rates = []
    
    for metrics in all_metrics:
        all_errors.append(metrics['localization_error'])
        detection_rates.append(metrics['detection_rate'])

    dintance_metrics = compute_distance_metrics_batch(all_ground_truth, all_localized)
    
    summary = {
        'mean_localization_error': np.mean(all_errors),
        'std_localization_error': np.std(all_errors),
        'mean_detection_rate': np.mean(detection_rates),
        'std_detection_rate': np.std(detection_rates),
        'mean_distance': dintance_metrics['overall_mean_distance'] if dintance_metrics else None,
        'std_distance': dintance_metrics['overall_std_distance'] if dintance_metrics else None,
        'all_localization_errors': all_errors,
        'all_detection_rates': detection_rates
    }
    
    return summary

def compute_distance_metrics(ground_truth_positions, fitted_positions):
    """
    Compute distance metrics between ground truth and fitted emitter positions.
    
    Uses Hungarian algorithm approach: each ground truth position is matched 
    to its nearest fitted position, and vice versa. The final metric uses
    whichever matching gives the smaller total distance.
    
    Parameters:
    - ground_truth_positions: List of (x, y) tuples or 2D array
    - fitted_positions: List of (x, y) tuples or 2D array
    
    Returns:
    - dict with keys: 'mean_distance', 'std_distance', 'matched_distances', 
      'n_matched', 'matching_efficiency'
    - Returns None if either input is empty
    """
    
    # Convert to numpy arrays and handle empty cases
    gt_pos = np.array(ground_truth_positions)
    fit_pos = np.array(fitted_positions)
    
    if len(gt_pos) == 0 or len(fit_pos) == 0:
        return None
    
    # Ensure 2D arrays
    if gt_pos.ndim == 1:
        gt_pos = gt_pos.reshape(1, -1)
    if fit_pos.ndim == 1:
        fit_pos = fit_pos.reshape(1, -1)
    
    # Compute distance matrix
    dist_matrix = cdist(gt_pos, fit_pos)
    
    # Strategy 1: Match each ground truth to nearest fitted position
    gt_to_fit_distances = np.min(dist_matrix, axis=1)
    gt_to_fit_total = np.sum(gt_to_fit_distances)
    
    # Strategy 2: Match each fitted position to nearest ground truth
    fit_to_gt_distances = np.min(dist_matrix, axis=0)
    fit_to_gt_total = np.sum(fit_to_gt_distances)
    
    # Use the strategy that gives smaller total distance
    if gt_to_fit_total <= fit_to_gt_total:
        matched_distances = gt_to_fit_distances
        n_matched = len(gt_pos)
    else:
        matched_distances = fit_to_gt_distances
        n_matched = len(fit_pos)
    
    # Compute statistics
    mean_distance = np.mean(matched_distances)
    std_distance = np.std(matched_distances)
    matching_efficiency = n_matched / max(len(gt_pos), len(fit_pos))
    
    return {
        'mean_distance': mean_distance,
        'std_distance': std_distance,
        'matched_distances': matched_distances,
        'n_matched': n_matched,
        'matching_efficiency': matching_efficiency
    }


def compute_distance_metrics_batch(ground_truth_list, fitted_list):
    """
    Compute distance metrics for multiple simulations.
    
    Parameters:
    - ground_truth_list: List of ground truth position lists/arrays
    - fitted_list: List of fitted position lists/arrays
    
    Returns:
    - dict with aggregated statistics across all simulations
    """
    
    all_mean_distances = []
    all_std_distances = []
    all_matched_distances = []
    all_matching_efficiencies = []
    n_valid_runs = 0
    
    for gt_pos, fit_pos in zip(ground_truth_list, fitted_list):
        metrics = compute_distance_metrics(gt_pos, fit_pos)
        
        if metrics is not None:
            all_mean_distances.append(metrics['mean_distance'])
            all_std_distances.append(metrics['std_distance'])
            all_matched_distances.extend(metrics['matched_distances'])
            all_matching_efficiencies.append(metrics['matching_efficiency'])
            n_valid_runs += 1
    
    if n_valid_runs == 0:
        return None
    
    return {
        'batch_mean_distance': np.mean(all_mean_distances),
        'batch_std_distance': np.std(all_mean_distances),
        'batch_mean_std': np.mean(all_std_distances),
        'overall_mean_distance': np.mean(all_matched_distances),
        'overall_std_distance': np.std(all_matched_distances),
        'mean_matching_efficiency': np.mean(all_matching_efficiencies),
        'std_matching_efficiency': np.std(all_matching_efficiencies),
        'n_valid_runs': n_valid_runs,
        'all_mean_distances': all_mean_distances,
        'all_std_distances': all_std_distances,
        'all_matching_efficiencies': all_matching_efficiencies
    }


def save_individual_run(results_dir, run_idx, photon_count_map, G2_diff_map, nr_emitters_map,
                       ground_truth, localized, metrics, metadata):
    """Save data from individual simulation run."""
    filename = results_dir / f"run_{run_idx:03d}.h5"
    
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
        
        # Metadata
        f.attrs['run_idx'] = run_idx
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



# Parameter sweep functionality
def run_parameter_sweep(
    emitter_densities=[1, 5, 10],
    laser_powers=[100E3, 200E3, 300E3],
    n_runs_per_param=10,
    base_params=None,
    verbose=True
):
    """
    Run evaluation for different emitter densities and laser powers.
    
    Parameters:
    - emitter_densities: List of emitter densities to test
    - laser_powers: List of laser powers to test
    - n_runs_per_param: Number of Monte Carlo runs per parameter combination
    - base_params: Base parameters (will be updated with sweep parameters)
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
    total_combinations = len(emitter_densities) * len(laser_powers)
    current_combo = 0
    
    if verbose:
        print(f"Starting parameter sweep with {total_combinations} combinations...")
        print(f"Emitter densities: {emitter_densities}")
        print(f"Laser powers: {laser_powers}")
        print(f"Runs per combination: {n_runs_per_param}")
    
    for density in emitter_densities:
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
            
            # Run evaluation for this parameter combination
            results_dir, summary_data = run_evaluation(
                n_runs=n_runs_per_param,
                base_params=sweep_params,
                save_individual=True,
                verbose=verbose  # Reduce verbosity for sweep
            )
            
            # Store results with descriptive key
            key = f"density_{density}_power_{power/1000:.0f}k"
            sweep_results[key] = {
                'emitter_density': density,
                'laser_power': power,
                'results_dir': results_dir,
                'summary_data': summary_data,
                'mean_detection_rate': summary_data['mean_detection_rate'],
                'std_detection_rate': summary_data['std_detection_rate'],
                'mean_localization_error': summary_data['mean_localization_error'],
                'std_localization_error': summary_data['std_localization_error'],
                'mean_distance': summary_data['mean_distance'],
                'std_distance': summary_data['std_distance'],
            }
            
            if verbose:
                print(f"  Detection rate: {summary_data['mean_detection_rate']:.3f} ± {summary_data['std_detection_rate']:.3f}")
                print(f"  Localization error: {summary_data['mean_localization_error']:.3f} ± {summary_data['std_localization_error']:.3f}")
                print(f"  Mean distance: {summary_data['mean_distance']:.3f} ± {summary_data['std_distance']:.3f}")
    
    # Save sweep summary
    save_sweep_summary(sweep_results)
    
    if verbose:
        print(f"\nParameter sweep completed!")
        print(f"Results saved for {len(sweep_results)} parameter combinations")
    
    return sweep_results


def save_sweep_summary(sweep_results):
    """Save parameter sweep summary to file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./project/data/parameter_sweep_{timestamp}.json"
    
    # Create summary for JSON saving (exclude non-serializable objects)
    json_summary = {}
    for key, result in sweep_results.items():
        json_summary[key] = {
            'emitter_density': result['emitter_density'],
            'laser_power': result['laser_power'],
            'results_dir': str(result['results_dir']),
            'mean_detection_rate': result['mean_detection_rate'],
            'std_detection_rate': result['std_detection_rate'],
            'mean_localization_error': result['mean_localization_error'],
            'std_localization_error': result['std_localization_error'],
            'mean_distance': result['mean_distance'],
            'std_distance': result['std_distance'],
        }
    
    with open(filename, 'w') as f:
        json.dump(json_summary, f, indent=2)
    
    print(f"Parameter sweep summary saved to: {filename}")



import json
import h5py
import numpy as np
from pathlib import Path
# compute_distance_metrics_batch is defined above in this file


def load_parameter_sweep(sweep_json_file):
    """
    Load parameter sweep results from JSON file.
    
    Parameters:
    - sweep_json_file: Path to the parameter sweep JSON summary file
    
    Returns:
    - sweep_data: Dictionary with loaded sweep results
    """
    with open(sweep_json_file, 'r') as f:
        sweep_data = json.load(f)
    
    return sweep_data


def reanalyze_parameter_sweep(sweep_json_file, compute_distances=True, verbose=True):
    """
    Reanalyze an existing parameter sweep dataset with updated metrics.
    
    Parameters:
    - sweep_json_file: Path to the parameter sweep JSON summary file
    - compute_distances: Whether to compute distance metrics (requires individual run files)
    - verbose: Print progress information
    
    Returns:
    - updated_results: Dictionary with reanalyzed results including new metrics
    """
    
    # Load existing sweep data
    sweep_data = load_parameter_sweep(sweep_json_file)
    updated_results = {}
    
    if verbose:
        print(f"Reanalyzing parameter sweep from: {sweep_json_file}")
        print(f"Found {len(sweep_data)} parameter combinations")
    
    for key, result in sweep_data.items():
        if verbose:
            print(f"\nProcessing {key}...")
        
        results_dir = Path(result['results_dir'])
        
        # Check if results directory exists
        if not results_dir.exists():
            if verbose:
                print(f"  Warning: Results directory not found: {results_dir}")
            updated_results[key] = result.copy()
            continue
        
        # Load summary file if it exists
        summary_file = results_dir / "summary.h5"
        if summary_file.exists():
            summary_data = load_summary_data(summary_file)
        else:
            summary_data = {}
        
        # Recompute distance metrics if requested and individual files exist
        if compute_distances:
            distance_metrics = recompute_distances_from_runs(results_dir, verbose)
            if distance_metrics:
                summary_data.update(distance_metrics)
        
        # Update the result
        updated_result = result.copy()
        updated_result['summary_data'] = summary_data
        
        # Add new metrics to top level for easy access
        if 'batch_mean_distance' in summary_data:
            updated_result['mean_distance'] = summary_data['batch_mean_distance']
            updated_result['std_distance'] = summary_data['batch_std_distance']
            updated_result['matching_efficiency'] = summary_data['mean_matching_efficiency']
        
        updated_results[key] = updated_result
        
        if verbose and 'batch_mean_distance' in summary_data:
            print(f"  Mean distance: {summary_data['batch_mean_distance']:.4f} ± {summary_data['batch_std_distance']:.4f}")
            print(f"  Matching efficiency: {summary_data['mean_matching_efficiency']:.3f}")
    
    # Save updated results
    save_updated_sweep(sweep_json_file, updated_results)
    
    if verbose:
        print(f"\nReanalysis completed! Updated results saved.")
    
    return updated_results


def load_summary_data(summary_file):
    """Load summary data from HDF5 file."""
    summary_data = {}
    
    with h5py.File(summary_file, 'r') as f:
        # Load datasets
        for key in f.keys():
            summary_data[key] = f[key][:]
        
        # Load attributes
        for key in f.attrs.keys():
            summary_data[key] = f.attrs[key]
    
    return summary_data


def recompute_distances_from_runs(results_dir, verbose=False):
    """
    Recompute distance metrics by loading individual run files.
    
    Parameters:
    - results_dir: Path to results directory containing individual run files
    - verbose: Print progress information
    
    Returns:
    - distance_metrics: Dictionary with distance metrics or None if failed
    """
    #load config file
    config_file = results_dir / 'config.json'
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    pixel_size = config_data['base_params']['area_size'][0]/ config_data['base_params']['positions'][0]
    area_size = config_data['base_params']['area_size']
    # Find all run files
    run_files = sorted(results_dir.glob("run_*.h5"))
    
    if not run_files:
        if verbose:
            print(f"  No individual run files found in {results_dir}")
        return None
    
    all_ground_truth = []
    all_localized = []
    
    for run_file in run_files:
        try:
            with h5py.File(run_file, 'r') as f:
                gt_pos = f['ground_truth_positions'][:]
                loc_pos = f['localized_positions'][:]
                
                # Convert to list of tuples
                if len(gt_pos) > 0:
                    gt_list = [(pos[0], pos[1]) for pos in gt_pos]
                else:
                    gt_list = []
                
                if len(loc_pos) > 0:
                    loc_list = [(pos[0], pos[1]) for pos in loc_pos]
                else:
                    loc_list = []
                
                all_ground_truth.append(gt_list)
                all_localized.append(loc_list)
        
        except Exception as e:
            if verbose:
                print(f"  Error reading {run_file}: {e}")
            continue
    
    if not all_ground_truth:
        return None
    
    # Compute batch distance metrics
    distance_metrics = compute_distance_metrics_batch(all_ground_truth, all_localized)
    
    if verbose and distance_metrics:
        print(f"  Loaded {len(all_ground_truth)} runs for distance analysis")
    
    return distance_metrics


def save_updated_sweep(original_file, updated_results):
    """Save updated sweep results to new file."""
    original_path = Path(original_file)
    updated_file = original_path.parent / f"{original_path.stem}_updated.json"
    
    # Create summary for JSON saving (exclude non-serializable objects)
    json_summary = {}
    for key, result in updated_results.items():
        json_result = {}
        for k, v in result.items():
            if k == 'summary_data':
                # Skip the full summary data for JSON (too large)
                continue
            elif isinstance(v, (np.ndarray, np.integer, np.floating)):
                json_result[k] = float(v) if np.isscalar(v) else v.tolist()
            else:
                json_result[k] = v
        json_summary[key] = json_result
    
    with open(updated_file, 'w') as f:
        json.dump(json_summary, f, indent=2)
    
    print(f"Updated sweep results saved to: {updated_file}")



###################################
##### Parameter Sweep Execution
###################################

if __name__ == "__main__":

    #Run parameter sweep
    sweep_results = run_parameter_sweep(
    emitter_densities=[5],
    laser_powers=[300E3],
    n_runs_per_param=2,
    verbose = True
    )
    #estimated time: 7 hours


###################################
##### Reanalysis of existing sweep
###################################

# if __name__ == "__main__":
#     # Example usage
#     sweep_file = "./project/data/parameter_sweep_20250612_103401.json"  # Replace with your file
#     # Simple reanalysis
#     updated_results = reanalyze_parameter_sweep(sweep_file, verbose=True)