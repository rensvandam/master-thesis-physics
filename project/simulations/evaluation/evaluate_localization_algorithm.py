import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

from project.model.localization import localize
from .evaluate_localization import compute_distance_metrics
from project.model.helper_functions import transform_coordinates


def compute_distance_metrics(estimated_positions, ground_truth_positions):
    """
    Compute distances between estimated and ground truth positions
    
    Args:
        estimated_positions: numpy array of shape (N, 2)
        ground_truth_positions: numpy array of shape (M, 2)
    
    Returns:
        distances: numpy array of distances (closest matches)
    """
    # Simple nearest neighbor matching
    distances = []
    
    for est_pos in estimated_positions:
        # Find closest ground truth position
        dists_to_gt = np.sqrt(np.sum((ground_truth_positions - est_pos)**2, axis=1))
        min_dist = np.min(dists_to_gt)
        distances.append(min_dist)
    
    return np.array(distances)

def analyze_single_run(h5_file_path, n_repetitions=25):
    """
    Analyze bias and precision for a single run
    
    Args:
        h5_file_path: path to the h5 file
        n_repetitions: number of localization repetitions
    
    Returns:
        bias: mean distance across all repetitions
        precision: standard deviation of distances across repetitions
    """
    with h5py.File(h5_file_path, 'r') as f:
        gt_pos = f['ground_truth_positions'][:]
        nr_em_map = f['nr_emitters_map'][:]
        G2_map = f['G2_diff_map'][:]
        count_map = f['photon_count_map'][:]
        metadata = dict(f.attrs)

        #metadata['dwell_time'] = 0.5
    
    all_distances = []
    all_detection_rates = []

    # Repeat localization n_repetitions times
    for rep in range(n_repetitions):

        # Localize emitters
        estimated_positions = localize(I_meas=count_map, 
                                       Gd_meas=G2_map, 
                                       est_emitters=nr_em_map, 
                                       metadata=metadata, 
                                        psf_file='project/data/psf.json', 
                                        verbose=False
                                    )

        estimated_positions = [transform_coordinates(x['y'], x['x'], metadata['area_size'], metadata['positions'], metadata['pixel_size'], direction='to_physical') for x in estimated_positions['emitters']]

        print(estimated_positions)
        print(gt_pos)
        # Compute distances to ground truth
        distances = compute_distance_metrics(estimated_positions, gt_pos)

        detection_rate = len(estimated_positions) / len(gt_pos) if len(gt_pos) > 0 else 0
        
        # Aggregate distances for this repetition (mean across all emitters)
        if len(distances) > 0:
            mean_distance_this_rep = np.mean(distances)
            all_distances.append(mean_distance_this_rep)

        all_detection_rates.append(detection_rate)
    
    # Calculate bias and precision across repetitions
    all_distances = np.array(all_distances)
    bias = np.mean(all_distances)  # Mean distance (systematic error)
    precision = np.std(all_distances)  # Standard deviation (random error)
    avg_detection_rate = np.mean(all_detection_rates)  # Average detection rate
    
    return bias, precision, avg_detection_rate

def analyze_multiple_runs(data_directory, file_pattern="run_*.h5", n_repetitions=25):
    """
    Analyze bias and precision across multiple runs
    
    Args:
        data_directory: directory containing h5 files
        file_pattern: pattern to match h5 files
        n_repetitions: number of repetitions per run
    
    Returns:
        run_numbers: list of run numbers
        biases: list of bias values
        precisions: list of precision values
    """
    # Find all h5 files
    h5_files = glob.glob(str(Path(data_directory) / file_pattern))
    h5_files.sort()  # Sort for consistent ordering
    
    run_numbers = []
    biases = []
    precisions = []
    avg_detection_rates = []
    for i, h5_file in enumerate(h5_files):
        print(f"Processing run {i+1}/{len(h5_files)}: {Path(h5_file).name}")
        
        try:
            bias, precision, avg_detection_rate = analyze_single_run(h5_file, n_repetitions)
            
            run_numbers.append(i + 1)
            biases.append(bias)
            precisions.append(precision)
            avg_detection_rates.append(avg_detection_rate)
            
        except Exception as e:
            print(f"Error processing {h5_file}: {e}")
            continue
    
    return run_numbers, biases, precisions, avg_detection_rates

def plot_bias_precision_results(run_numbers, biases, precisions, avg_detection_rates):
    """
    Create separate plots for bias and precision
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot bias
    ax1.plot(run_numbers, biases, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Run Number')
    ax1.set_ylabel('Bias (Mean Distance)')
    ax1.set_title('Localization Bias Across Different Ground Truth Scenarios')
    ax1.grid(True, alpha=0.3)
    
    # Add horizontal line at zero for reference
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero Bias')
    ax1.legend()
    
    # Plot precision
    ax2.plot(run_numbers, precisions, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Run Number')
    ax2.set_ylabel('Precision (Std Dev of Distances)')
    ax2.set_title('Localization Precision Across Different Ground Truth Scenarios')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Mean Bias: {np.mean(biases):.3f} ± {np.std(biases):.3f}")
    print(f"Mean Precision: {np.mean(precisions):.10f} ± {np.std(precisions):.10f}")
    print(f"Mean Average Detection Rate: {np.mean(avg_detection_rates):.2f} ± {np.std(avg_detection_rates):.2f}")

# Main execution
if __name__ == "__main__":
    # Set your data directory path

   # emitter densities 1,5,10, 0.1 dwell time, NO noise, 200kW laser power
    #data_directory = "project/data/evaluation_20250710_164147/"
    #data_directory = "project/data/evaluation_20250710_164315/"
    #data_directory = "project/data/evaluation_20250710_164908/"

   # emitter densities 1,5,10, 0.1 dwell time, 50ns deadtime, 200kW laser power
    #data_directory = "project/data/evaluation_20250708_150117/"
    #data_directory = "project/data/evaluation_20250708_150248/"
    #data_directory = "project/data/evaluation_20250708_150829/"
    
   # emitter densities 1,5,10, 0.1 dwell time, 50ns deadtime, 100kW laser power
    #data_directory = "project/data/evaluation_20250716_151400/"
    data_directory = "project/data/evaluation_20250716_151526/"
   # data_directory = "project/data/evaluation_20250716_152058/"


    
    # Run analysis
    print("Starting bias and precision analysis...")
    # run_numbers, biases, precisions, avg_detection_rates = analyze_multiple_runs(
    #     data_directory, 
    #     file_pattern="run_*.h5", 
    #     n_repetitions=25
    # )
    # Plot results
    # if len(run_numbers) > 0:
    #     plot_bias_precision_results(run_numbers, biases, precisions, avg_detection_rates)
    # else:
    #     print("No runs were successfully processed!")

    #################################
    # Just one run

    # Find all h5 files
    file_pattern = "run_000*.h5"  # Change this to match your specific run file
    h5_files = glob.glob(str(Path(data_directory) / file_pattern))
    h5_files.sort()
    bias, precision, avg_detection_rate = analyze_single_run(h5_files[0], n_repetitions=25)
    # Plot results
    print(f"Bias: {bias}, Precision: {precision}, Average Detection Rate: {avg_detection_rate}")