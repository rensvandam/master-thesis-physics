from tabnanny import verbose
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from scipy.optimize import linear_sum_assignment
import h5py
from pathlib import Path
import glob
from datetime import datetime
import json

from tqdm import tqdm

#from project.model.localization import localize


from project.model.localization import localize
from .evaluate_localization import compute_distance_metrics
from project.model.helper_functions import transform_coordinates

def match_positions_hungarian(ground_truth_positions: np.ndarray, 
                             estimated_positions: np.ndarray, 
                             max_distance: float = np.inf,
                             verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Match estimated positions to ground truth positions using Hungarian algorithm.
    Handles cases where the number of estimated positions != number of ground truth positions.
    
    Parameters:
    -----------
    ground_truth_positions : np.ndarray
        Array of shape (n_gt, n_dims) containing true emitter positions
    estimated_positions : np.ndarray
        Array of shape (n_est, n_dims) containing estimated positions
    max_distance : float, optional
        Maximum allowed distance for a valid match. Default is np.inf (no limit)
    
    Returns:
    --------
    Tuple containing:
        - matched_estimated_positions: estimated positions matched to ground truth (n_gt, n_dims)
                                     Contains NaN for unmatched ground truth positions
        - assignment_indices: indices showing which estimated position matches which ground truth
                            Contains -1 for unmatched ground truth positions
        - total_cost: sum of all matching distances
        - match_distances: individual distances for each match
    """
    n_gt = ground_truth_positions.shape[0]
    n_est = estimated_positions.shape[0]
    n_dims = ground_truth_positions.shape[1] 

    if verbose:
        print(f"n_gt: {n_gt}. n_est: {n_est}. n_dims: {n_dims}.")
    
    if n_est == 0:
        # No estimated positions
        matched_estimated_positions = np.full((n_gt, n_dims), np.nan)
        assignment_indices = np.full(n_gt, -1, dtype=int)
        return matched_estimated_positions, assignment_indices, np.inf, np.full(n_gt, np.inf)
    
    if n_gt == 0:
        # No ground truth positions
        return np.array([]).reshape(0, n_dims), np.array([], dtype=int), 0.0, np.array([])
    
    # Create cost matrix - handle different sizes
    max_size = max(n_gt, n_est)
    cost_matrix = np.full((max_size, max_size), max_distance * 100)  # High cost for dummy assignments
    
    # Fill in actual distances
    for i in range(n_gt):
        for j in range(n_est):
            distance = np.linalg.norm(ground_truth_positions[i] - estimated_positions[j])
            cost_matrix[i, j] = distance

    if verbose:
        print(f"Cost matrix:")
        print(cost_matrix)
    
    # Solve assignment problem using Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    if verbose:
        print(f"indices matches; {row_indices}, {col_indices}")
    
    # Extract valid assignments (not involving dummy nodes)
    matched_estimated_positions = np.full((n_gt, n_dims), np.nan) 
    assignment_indices = np.full(n_gt, -1, dtype=int)
    match_distances = np.full(n_gt, np.inf)
    total_cost = 0.0
    
    for row_idx, col_idx in zip(row_indices, col_indices):
        if row_idx < n_gt and col_idx < n_est:
            distance = np.linalg.norm(ground_truth_positions[row_idx] - estimated_positions[col_idx])
            if verbose:
                print(f"distance: {distance}; max_distance {max_distance}")
            if distance <= max_distance:
                matched_estimated_positions[row_idx] = estimated_positions[col_idx]
                assignment_indices[row_idx] = col_idx
                match_distances[row_idx] = distance
                total_cost += distance
    
    return matched_estimated_positions, assignment_indices, total_cost, match_distances


def calculate_localization_stats_relative_coordinates(ground_truth_positions: np.ndarray, 
                                                     estimated_positions_list: List[np.ndarray],
                                                     max_distance: float = np.inf) -> Dict:
    """
    Calculate localization statistics using relative coordinates centered on the mean position.
    This makes the theoretical mean zero for each emitter across multiple runs.
    
    Parameters:
    -----------
    ground_truth_positions : np.ndarray
        Array of shape (n_emitters, n_dims) containing true emitter positions
    estimated_positions_list : List[np.ndarray]
        List of arrays, each of varying shape (n_est_i, n_dims) containing estimated positions
        for each localization run (positions are not pre-matched to ground truth)
    max_distance : float, optional
        Maximum allowed distance for a valid match. Default is np.inf (no limit)
    
    Returns:
    --------
    Dict containing relative coordinate statistics:
        - 'relative_coords_per_emitter': relative coordinates for each emitter across all runs
        - 'mean_relative_coords_per_emitter': mean relative coordinate for each emitter (should be ~[0,0])
        - 'std_relative_coords_per_emitter': std deviation of relative coordinates for each emitter
        - 'precision_per_emitter': RMS of relative coordinates (localization precision)
        - 'match_rate_per_emitter': fraction of runs where each emitter was successfully matched
        - 'overall_precision': overall RMS across all matched pairs
        - 'mean_precision': mean of per-emitter precisions
        - 'std_precision': std of per-emitter precisions
        - 'assignments_per_run': assignment indices for each run
        - 'total_costs_per_run': total matching cost for each run
        - 'n_estimated_per_run': number of estimated positions in each run
        - 'n_matched_per_run': number of successful matches in each run
        - 'centroid_positions_per_emitter': mean position for each emitter (the reference point)
    """
    n_emitters = ground_truth_positions.shape[0]
    n_dims = ground_truth_positions.shape[1]
    n_runs = len(estimated_positions_list)
    
    # Store matched positions for each emitter across all runs
    matched_positions_per_emitter = [[] for _ in range(n_emitters)]
    assignments_per_run = []
    total_costs_per_run = []
    n_estimated_per_run = []
    n_matched_per_run = []

    FP_list = []  # False positives per run
    TP_list = []  # True positives per run
    FN_list = []  # False negatives per run
    
    # First pass: collect all matched positions
    for run_idx in range(n_runs):
        estimated_positions = estimated_positions_list[run_idx]
        n_estimated_per_run.append(len(estimated_positions))
        
        # Match positions using Hungarian algorithm.
        # Matched positions here is an array (length: n_gt) of arrays (each containing the matched estimated position for that gt)
        matched_positions, assignment_indices, total_cost, match_distances = match_positions_hungarian(
            ground_truth_positions, estimated_positions, max_distance
        )
        #print(f"Matched positions: {matched_positions}")
        
        assignments_per_run.append(assignment_indices)
        total_costs_per_run.append(total_cost)
        
        # Count successful matches
        n_matched = np.sum(assignment_indices >= 0)
        n_matched_per_run.append(n_matched)
        
        # Store matched positions for each emitter
        for emitter_idx in range(n_emitters):
            if assignment_indices[emitter_idx] >= 0:  # Valid match
                matched_positions_per_emitter[emitter_idx].append(matched_positions[emitter_idx])
            else:  # No match found - store NaN
                matched_positions_per_emitter[emitter_idx].append(np.full(n_dims, np.nan))

        TP = np.sum(assignment_indices >= 0)
        FN = np.sum(assignment_indices == -1)
        FP = len(estimated_positions) - TP
        print(f"Run {run_idx}: TP={TP}, FP={FP}, FN={FN}")
        FP_list.append(FP)
        TP_list.append(TP)
        FN_list.append(FN)
    
    # Convert to numpy arrays.
    matched_positions_per_emitter = [np.array(pos_list) for pos_list in matched_positions_per_emitter]
    # So the array above is an array with for each GT emitter, it contains the n_runs (e.g. 50/100) number of localized coordinates for each emitter (for the same specifications ofcourse)
    
    # Calculate centroids (mean positions) for each emitter
    centroid_positions_per_emitter = []
    for emitter_idx in range(n_emitters):
        positions = matched_positions_per_emitter[emitter_idx] 
        # Calculate centroid ignoring NaN values
        valid_positions = positions[~np.isnan(positions).any(axis=1)]
        if len(valid_positions) > 0:
            centroid = np.nanmean(valid_positions, axis=0) # centroid of n_runs number of localized positions
        else:
            centroid = np.full(n_dims, np.nan)
        centroid_positions_per_emitter.append(centroid)
    
    centroid_positions_per_emitter = np.array(centroid_positions_per_emitter)
    
    # Calculate relative coordinates (positions relative to centroid)
    relative_coords_per_emitter = []
    for emitter_idx in range(n_emitters):
        positions = matched_positions_per_emitter[emitter_idx] # all positions for this emitter across all runs (e.g. 50/100)
        centroid = centroid_positions_per_emitter[emitter_idx] # single centroid for an emitter
        
        # Calculate relative coordinates
        if not np.isnan(centroid).any():
            relative_coords = positions - centroid
        else:
            relative_coords = np.full_like(positions, np.nan)
        
        relative_coords_per_emitter.append(relative_coords)
    
    relative_coords_per_emitter = np.array(relative_coords_per_emitter) #These are for each emitter, n_runs number of relative coordinates of estimations w.r.t. centroid
    
    # Calculate statistics for relative coordinates
    mean_relative_coords_per_emitter = []
    std_relative_coords_per_emitter = []
    precision_per_emitter = []
    match_rate_per_emitter = []

    mean_bias_per_emitter = []
    mean_bias_direction_per_emitter = []
    
    for emitter_idx in range(n_emitters):
        rel_coords = relative_coords_per_emitter[emitter_idx]
        
        # Mean relative coordinates (should be close to zero)
        mean_rel_coord = np.nanmean(rel_coords, axis=0)
        mean_relative_coords_per_emitter.append(mean_rel_coord)
        
        # Standard deviation of relative coordinates
        std_rel_coord = np.nanstd(rel_coords, axis=0, ddof=1)
        std_relative_coords_per_emitter.append(std_rel_coord)
        
        # Precision (RMS of relative coordinates)
        # valid_coords = rel_coords[~np.isnan(rel_coords).any(axis=1)]
        # if len(valid_coords) > 0:
        #     rms_precision = np.sqrt(np.mean(np.sum(valid_coords**2, axis=1)))
        # else:
        #     rms_precision = np.nan
        # precision_per_emitter.append(rms_precision)

        # Mean bias is average distance from ground truth to estimated coordinate (valid_positions)
        positions = matched_positions_per_emitter[emitter_idx]
        valid_positions = positions[~np.isnan(positions).any(axis=1)]

        if len(valid_positions) > 0:
            mean_vector = np.mean(valid_positions - ground_truth_positions[emitter_idx], axis=0)
            bias = np.linalg.norm(mean_vector)
            bias_direction = np.arctan2(mean_vector[1], mean_vector[0])  # Angle in radians
        else:
            bias = np.nan
            bias_direction = np.nan
        mean_bias_per_emitter.append(bias)
        mean_bias_direction_per_emitter.append(bias_direction)

        # Precision (RMS of distance from ground truth to estimaed coordinate (valid_positions)
        if len(valid_positions) > 0:
            mean_position = np.mean(valid_positions, axis=0)
            rms_precision = np.sqrt(np.mean(np.sum((valid_positions - mean_position)**2, axis=1)))
            #rms_precision = np.sqrt(np.mean((valid_positions - ground_truth_positions[emitter_idx])**2))
        else:
            rms_precision = np.nan
        precision_per_emitter.append(rms_precision)
        
        # Match rate
        n_valid = np.sum(~np.isnan(rel_coords).any(axis=1))
        match_rate = n_valid / n_runs
        match_rate_per_emitter.append(match_rate)
    
    # Convert to numpy arrays
    mean_relative_coords_per_emitter = np.array(mean_relative_coords_per_emitter)
    std_relative_coords_per_emitter = np.array(std_relative_coords_per_emitter)
    precision_per_emitter = np.array(precision_per_emitter)
    match_rate_per_emitter = np.array(match_rate_per_emitter)
    
    # Calculate overall statistics
    all_valid_rel_coords = []
    for emitter_idx in range(n_emitters):
        rel_coords = relative_coords_per_emitter[emitter_idx]
        valid_coords = rel_coords[~np.isnan(rel_coords).any(axis=1)]
        if len(valid_coords) > 0:
            all_valid_rel_coords.extend(valid_coords)
    
    if len(all_valid_rel_coords) > 0:
        all_valid_rel_coords = np.array(all_valid_rel_coords)
        overall_precision = np.sqrt(np.mean(np.sum(all_valid_rel_coords**2, axis=1)))
    else:
        overall_precision = np.nan
    
    # Statistics of per-emitter precisions
    valid_precisions = precision_per_emitter[~np.isnan(precision_per_emitter)]
    if len(valid_precisions) > 0:
        mean_precision = np.mean(valid_precisions)
        std_precision = np.std(valid_precisions, ddof=1) if len(valid_precisions) > 1 else 0.0
    else:
        mean_precision = np.nan
        std_precision = np.nan
    
    return {
        'relative_coords_per_emitter': relative_coords_per_emitter,
        'mean_relative_coords_per_emitter': mean_relative_coords_per_emitter,
        'std_relative_coords_per_emitter': std_relative_coords_per_emitter,
        'precision_per_emitter': precision_per_emitter,
        'match_rate_per_emitter': match_rate_per_emitter,
        'overall_precision': overall_precision,
        'mean_precision': mean_precision,
        'std_precision': std_precision,
        'assignments_per_run': assignments_per_run,
        'total_costs_per_run': total_costs_per_run,
        'n_estimated_per_run': n_estimated_per_run,
        'n_matched_per_run': n_matched_per_run,
        'centroid_positions_per_emitter': centroid_positions_per_emitter, 
        'mean_bias_per_emitter': mean_bias_per_emitter,  # this one is the average bias
        'mean_bias_direction_per_emitter': mean_bias_direction_per_emitter,
        'TP_per_run': TP_list,
        'FP_per_run': FP_list,
        'FN_per_run': FN_list
    }


def calculate_localization_stats_with_matching(ground_truth_positions: np.ndarray, 
                                              estimated_positions_list: List[np.ndarray],
                                              max_distance: float = np.inf) -> Dict:
    """
    Calculate localization distance statistics across multiple runs with position matching.
    Handles cases where number of estimated positions != number of ground truth positions.
    
    Parameters:
    -----------
    ground_truth_positions : np.ndarray
        Array of shape (n_emitters, n_dims) containing true emitter positions
    estimated_positions_list : List[np.ndarray]
        List of arrays, each of varying shape (n_est_i, n_dims) containing estimated positions
        for each localization run (positions are not pre-matched to ground truth)
    max_distance : float, optional
        Maximum allowed distance for a valid match. Default is np.inf (no limit)
    
    Returns:
    --------
    Dict containing:
        - 'distances_per_emitter': distances for each emitter across all runs (NaN for unmatched)
        - 'mean_distance_per_emitter': mean distance for each emitter (ignoring NaN)
        - 'std_distance_per_emitter': std deviation for each emitter (ignoring NaN)
        - 'match_rate_per_emitter': fraction of runs where each emitter was successfully matched
        - 'overall_mean_distance': overall mean distance across all matched pairs
        - 'overall_std_distance': overall std deviation across all matched pairs
        - 'mean_of_means': mean of per-emitter means
        - 'std_of_means': std of per-emitter means
        - 'assignments_per_run': assignment indices for each run
        - 'total_costs_per_run': total matching cost for each run
        - 'n_estimated_per_run': number of estimated positions in each run
        - 'n_matched_per_run': number of successful matches in each run
    """
    n_emitters = ground_truth_positions.shape[0]
    n_runs = len(estimated_positions_list)
    
    # Store distances for each emitter across all runs
    distances_per_emitter = [[] for _ in range(n_emitters)]
    assignments_per_run = []
    total_costs_per_run = []
    n_estimated_per_run = []
    n_matched_per_run = []
    
    # Process each run
    for run_idx in range(n_runs):
        estimated_positions = estimated_positions_list[run_idx]
        n_estimated_per_run.append(len(estimated_positions))
        
        # Match positions using Hungarian algorithm
        matched_positions, assignment_indices, total_cost, match_distances = match_positions_hungarian(
            ground_truth_positions, estimated_positions, max_distance
        )
        
        assignments_per_run.append(assignment_indices)
        total_costs_per_run.append(total_cost)
        
        # Count successful matches
        n_matched = np.sum(assignment_indices >= 0)
        n_matched_per_run.append(n_matched)
        
        # Calculate distances for each emitter
        for emitter_idx in range(n_emitters):
            if assignment_indices[emitter_idx] >= 0:  # Valid match
                distance = match_distances[emitter_idx]
                distances_per_emitter[emitter_idx].append(distance)
            else:  # No match found
                distances_per_emitter[emitter_idx].append(np.nan)
    
    # Convert to numpy array and calculate statistics
    distances_per_emitter = np.array([np.array(dist_list) for dist_list in distances_per_emitter])
    
    # Calculate statistics per emitter (ignoring NaN values)
    mean_distance_per_emitter = np.array([np.nanmean(distances) for distances in distances_per_emitter])
    std_distance_per_emitter = np.array([np.nanstd(distances, ddof=1) for distances in distances_per_emitter])
    match_rate_per_emitter = np.array([np.sum(~np.isnan(distances)) / n_runs for distances in distances_per_emitter])
    
    # Calculate overall statistics (only for matched pairs)
    all_valid_distances = []
    for distances in distances_per_emitter:
        all_valid_distances.extend(distances[~np.isnan(distances)])
    
    if len(all_valid_distances) > 0:
        overall_mean_distance = np.mean(all_valid_distances)
        overall_std_distance = np.std(all_valid_distances, ddof=1) if len(all_valid_distances) > 1 else 0.0
    else:
        overall_mean_distance = np.nan
        overall_std_distance = np.nan
    
    # Calculate mean and std of the per-emitter means (ignoring NaN)
    valid_means = mean_distance_per_emitter[~np.isnan(mean_distance_per_emitter)]
    if len(valid_means) > 0:
        mean_of_means = np.mean(valid_means)
        std_of_means = np.std(valid_means, ddof=1) if len(valid_means) > 1 else 0.0
    else:
        mean_of_means = np.nan
        std_of_means = np.nan
    
    return {
        'distances_per_emitter': distances_per_emitter,
        'mean_distance_per_emitter': mean_distance_per_emitter,
        'std_distance_per_emitter': std_distance_per_emitter,
        'match_rate_per_emitter': match_rate_per_emitter,
        'overall_mean_distance': overall_mean_distance,
        'overall_std_distance': overall_std_distance,
        'mean_of_means': mean_of_means,
        'std_of_means': std_of_means,
        'assignments_per_run': assignments_per_run,
        'total_costs_per_run': total_costs_per_run,
        'n_estimated_per_run': n_estimated_per_run,
        'n_matched_per_run': n_matched_per_run
    }


def plot_localization_results_with_assignments(stats: Dict, save_path: str = None):
    """
    Plot localization distance results including assignment information.
    
    Parameters:
    -----------
    stats : Dict
        Results from calculate_localization_stats_with_matching
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Distance distribution for each emitter
    ax1 = axes[0, 0]
    distances = stats['distances_per_emitter']
    n_emitters = distances.shape[0]
    
    # Box plot
    ax1.boxplot([distances[i] for i in range(n_emitters)], 
                labels=[f'E{i+1}' for i in range(n_emitters)])
    ax1.set_title('Distance Distribution per Emitter')
    ax1.set_xlabel('Emitter')
    ax1.set_ylabel('Localization Error Distance')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean distance per emitter with error bars
    ax2 = axes[0, 1]
    emitter_indices = np.arange(n_emitters)
    ax2.errorbar(emitter_indices, stats['mean_distance_per_emitter'], 
                yerr=stats['std_distance_per_emitter'], 
                marker='o', capsize=5, capthick=2)
    ax2.set_title('Mean Distance per Emitter (±1σ)')
    ax2.set_xlabel('Emitter Index')
    ax2.set_ylabel('Mean Localization Error Distance')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(emitter_indices)
    
    # Plot 3: Overall distance distribution
    ax3 = axes[0, 2]
    all_distances = distances.flatten()
    ax3.hist(all_distances, bins=20, alpha=0.7, edgecolor='black')
    ax3.axvline(stats['overall_mean_distance'], color='red', linestyle='--', 
                label=f'Mean: {stats["overall_mean_distance"]:.3f}')
    ax3.set_title('Overall Distance Distribution')
    ax3.set_xlabel('Localization Error Distance')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Assignment consistency (how often each estimated position is matched to each ground truth)
    ax4 = axes[1, 0]
    assignments = np.array(stats['assignments_per_run'])
    assignment_matrix = np.zeros((n_emitters, n_emitters))
    
    for run_assignments in assignments:
        for gt_idx, est_idx in enumerate(run_assignments):
            assignment_matrix[gt_idx, est_idx] += 1
    
    im = ax4.imshow(assignment_matrix, cmap='Blues', aspect='auto')
    ax4.set_title('Assignment Frequency Matrix')
    ax4.set_xlabel('Estimated Position Index')
    ax4.set_ylabel('Ground Truth Position Index')
    
    # Add text annotations
    for i in range(n_emitters):
        for j in range(n_emitters):
            text = ax4.text(j, i, int(assignment_matrix[i, j]), 
                           ha="center", va="center", color="black")
    
    plt.colorbar(im, ax=ax4, label='Number of Times Matched')
    
    # Plot 5: Total matching cost per run
    ax5 = axes[1, 1]
    run_indices = np.arange(len(stats['total_costs_per_run']))
    ax5.plot(run_indices, stats['total_costs_per_run'], 'o-')
    ax5.set_title('Total Matching Cost per Run')
    ax5.set_xlabel('Run Index')
    ax5.set_ylabel('Total Matching Cost')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    mean_assignment_cost = np.mean(stats['total_costs_per_run'])
    std_assignment_cost = np.std(stats['total_costs_per_run'], ddof=1)
    
    summary_text = f"""
    Summary Statistics:
    
    Overall Mean Distance: {stats['overall_mean_distance']:.4f}
    Overall Std Distance: {stats['overall_std_distance']:.4f}
    
    Mean of Per-Emitter Means: {stats['mean_of_means']:.4f}
    Std of Per-Emitter Means: {stats['std_of_means']:.4f}
    
    Mean Assignment Cost: {mean_assignment_cost:.4f}
    Std Assignment Cost: {std_assignment_cost:.4f}
    
    Number of Emitters: {n_emitters}
    Number of Runs: {distances.shape[1]}
    
    Best Performing Emitter: {np.argmin(stats['mean_distance_per_emitter']) + 1}
    (Mean distance: {np.min(stats['mean_distance_per_emitter']):.4f})
    
    Worst Performing Emitter: {np.argmax(stats['mean_distance_per_emitter']) + 1}
    (Mean distance: {np.max(stats['mean_distance_per_emitter']):.4f})
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_detailed_results_with_matching(stats: Dict):
    """
    Print detailed localization statistics including matching information.
    
    Parameters:
    -----------
    stats : Dict
        Results from calculate_localization_stats_with_matching
    """
    print("=" * 70)
    print("LOCALIZATION DISTANCE ANALYSIS RESULTS (WITH POSITION MATCHING)")
    print("=" * 70)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"Overall Mean Distance: {stats['overall_mean_distance']:.6f}")
    print(f"Overall Std Distance:  {stats['overall_std_distance']:.6f}")
    print(f"Overall 95% CI: [{stats['overall_mean_distance'] - 1.96*stats['overall_std_distance']:.6f}, "
          f"{stats['overall_mean_distance'] + 1.96*stats['overall_std_distance']:.6f}]")
    
    print(f"\nPER-EMITTER MEAN STATISTICS:")
    print(f"Mean of Means: {stats['mean_of_means']:.6f}")
    print(f"Std of Means:  {stats['std_of_means']:.6f}")
    
    print(f"\nMATCHING STATISTICS:")
    mean_cost = np.mean(stats['total_costs_per_run'])
    std_cost = np.std(stats['total_costs_per_run'], ddof=1)
    print(f"Mean Assignment Cost: {mean_cost:.6f}")
    print(f"Std Assignment Cost:  {std_cost:.6f}")
    print(f"Min Assignment Cost:  {np.min(stats['total_costs_per_run']):.6f}")
    print(f"Max Assignment Cost:  {np.max(stats['total_costs_per_run']):.6f}")
    
    print(f"\nPER-EMITTER DETAILED RESULTS:")
    print(f"{'Emitter':<8} {'Mean Dist':<12} {'Std Dist':<12} {'Min Dist':<12} {'Max Dist':<12}")
    print("-" * 60)
    
    for i in range(len(stats['mean_distance_per_emitter'])):
        distances = stats['distances_per_emitter'][i]
        print(f"{i+1:<8} {stats['mean_distance_per_emitter'][i]:<12.6f} "
              f"{stats['std_distance_per_emitter'][i]:<12.6f} "
              f"{np.min(distances):<12.6f} {np.max(distances):<12.6f}")
    
    # Show assignment consistency
    print(f"\nASSIGNMENT CONSISTENCY:")
    assignments = np.array(stats['assignments_per_run'])
    n_emitters = len(stats['mean_distance_per_emitter'])
    assignment_matrix = np.zeros((n_emitters, n_emitters))
    
    for run_assignments in assignments:
        for gt_idx, est_idx in enumerate(run_assignments):
            assignment_matrix[gt_idx, est_idx] += 1
    
    print("Ground Truth -> Most Frequently Matched Estimated Position:")
    for i in range(n_emitters):
        most_frequent_match = np.argmax(assignment_matrix[i])
        frequency = assignment_matrix[i, most_frequent_match]
        percentage = (frequency / len(assignments)) * 100
        print(f"GT {i+1} -> EST {most_frequent_match+1} ({frequency:.0f}/{len(assignments)} runs, {percentage:.1f}%)")

def print_detailed_relative_coordinate_results(stats: Dict):
    """
    Print detailed relative coordinate localization statistics.
    
    Parameters:
    -----------
    stats : Dict
        Results from calculate_localization_stats_relative_coordinates
    """
    print("=" * 70)
    print("LOCALIZATION ANALYSIS RESULTS (RELATIVE COORDINATES)")
    print("=" * 70)
    
    print(f"\nOVERALL PRECISION STATISTICS:")
    print(f"Overall Precision (RMS): {stats['overall_precision']:.10f}")
    print(f"Mean of Per-Emitter Precisions: {stats['mean_precision']:.10f}")
    print(f"Std of Per-Emitter Precisions:  {stats['std_precision']:.10f}")

    print(f"Mean Bias per Emitter: {np.mean(stats['mean_bias_per_emitter']):.6f}")
    print(f"Per Emitter Mean Bias:")
    for i, bias in enumerate(stats['mean_bias_per_emitter']):
        print(f"Emitter {i+1}: {bias:.6f}")

    print(f"\nMATCHING STATISTICS:")
    mean_cost = np.mean(stats['total_costs_per_run'])
    std_cost = np.std(stats['total_costs_per_run'], ddof=1)
    print(f"Mean Assignment Cost: {mean_cost:.6f}")
    print(f"Std Assignment Cost:  {std_cost:.6f}")
    print(f"Min Assignment Cost:  {np.min(stats['total_costs_per_run']):.6f}")
    print(f"Max Assignment Cost:  {np.max(stats['total_costs_per_run']):.6f}")
    
    print(f"\nPER-EMITTER DETAILED RESULTS:")
    print(f"{'Emitter':<8} {'Precision':<12} {'Mean X':<12} {'Mean Y':<12} {'Std X':<12} {'Std Y':<12} {'Match Rate':<12}")
    print("-" * 88)
    
    for i in range(len(stats['precision_per_emitter'])):
        mean_coords = stats['mean_relative_coords_per_emitter'][i]
        std_coords = stats['std_relative_coords_per_emitter'][i]
        precision = stats['precision_per_emitter'][i]
        match_rate = stats['match_rate_per_emitter'][i]
        
        print(f"{i+1:<8} {precision:<12.6f} "
              f"{mean_coords[0]:<12.6f} {mean_coords[1]:<12.6f} "
              f"{std_coords[0]:<12.6f} {std_coords[1]:<12.6f} "
              f"{match_rate:<12.3f}")
    
    print(f"\nCENTROID POSITIONS (REFERENCE POINTS):")
    print(f"{'Emitter':<8} {'Centroid X':<12} {'Centroid Y':<12}")
    print("-" * 32)
    
    for i, centroid in enumerate(stats['centroid_positions_per_emitter']):
        if not np.isnan(centroid).any():
            print(f"{i+1:<8} {centroid[0]:<12.6f} {centroid[1]:<12.6f}")
        else:
            print(f"{i+1:<8} {'NaN':<12} {'NaN':<12}")

def plot_positions_overview(ground_truth_positions: np.ndarray, 
                           estimated_positions_list: List[np.ndarray],
                           image_map,
                           metadata,
                           stats: Dict = None,
                           max_runs_to_plot: int = 10,
                           save_path: str = None):
    """
    Plot overview of ground truth positions vs estimated positions to verify spread.
    
    Parameters:
    -----------
    ground_truth_positions : np.ndarray
        Array of shape (n_emitters, 2) containing true emitter positions
    estimated_positions_list : List[np.ndarray]
        List of estimated positions from multiple runs
    stats : Dict, optional
        Results from calculate_localization_stats_relative_coordinates (for centroids)
    max_runs_to_plot : int
        Maximum number of runs to plot (to avoid overcrowding)
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: All positions overview
    ax1 = axes[0]
    
    # Display image with reduced alpha for background effect
    area_size = metadata['area_size']
    ax1.imshow(image_map.T, extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2],
            origin='lower',
            interpolation='nearest',
            alpha=0.3, cmap='hot', zorder=0)
    # Plot ground truth positions
    ax1.scatter(ground_truth_positions[:, 0], ground_truth_positions[:, 1], 
               c='red', s=100, marker='x', label='Ground Truth', zorder=10, edgecolors='black', linewidth=1)
    
    # Plot estimated positions from multiple runs (sample to avoid overcrowding)
    colors = plt.cm.tab10(np.linspace(0, 1, min(max_runs_to_plot, len(estimated_positions_list))))

    n_runs_to_plot = min(max_runs_to_plot, len(estimated_positions_list))
    
    for run_idx in range(n_runs_to_plot):
        estimated_positions = estimated_positions_list[run_idx]
        if len(estimated_positions) > 0:
            ax1.scatter(estimated_positions[:, 0], estimated_positions[:, 1], 
                       c=[colors[run_idx]], alpha=0.6, s=30, 
                       label=f'Run {run_idx+1}' if run_idx < 5 else None)
    
    # If we have stats with centroids, plot them too
    if stats is not None and 'centroid_positions_per_emitter' in stats:
        centroids = stats['centroid_positions_per_emitter']
        valid_centroids = centroids[~np.isnan(centroids).any(axis=1)]
        if len(valid_centroids) > 0:
            ax1.scatter(valid_centroids[:, 0], valid_centroids[:, 1], 
                       c='blue', s=80, marker='o', label='Centroids', 
                       zorder=9, edgecolors='black', linewidth=1)
    
    ax1.set_title(f'Position Overview (First {n_runs_to_plot} runs)')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Matched positions with connections
    ax2 = axes[1]
    
    if stats is not None:
        # Plot ground truth positions
        ax2.scatter(ground_truth_positions[:, 0], ground_truth_positions[:, 1], 
                   c='red', s=100, marker='*', label='Ground Truth', zorder=10, 
                   edgecolors='black', linewidth=1)
        
        # Plot centroids
        centroids = stats['centroid_positions_per_emitter']
        for i, centroid in enumerate(centroids):
            if not np.isnan(centroid).any():
                ax2.scatter(centroid[0], centroid[1], 
                           c='blue', s=80, marker='o', zorder=9, 
                           edgecolors='black', linewidth=1)
                
                # Draw line from ground truth to centroid
                ax2.plot([ground_truth_positions[i, 0], centroid[0]], 
                        [ground_truth_positions[i, 1], centroid[1]], 
                        'k--', alpha=0.5, linewidth=1)
                
                # Add emitter labels
                ax2.annotate(f'E{i+1}', (centroid[0], centroid[1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot some individual estimated positions around centroids
        for emitter_idx in range(len(stats['relative_coords_per_emitter'])):
            rel_coords = stats['relative_coords_per_emitter'][emitter_idx]
            centroid = centroids[emitter_idx]
            
            if not np.isnan(centroid).any():
                valid_rel_coords = rel_coords[~np.isnan(rel_coords).any(axis=1)]
                if len(valid_rel_coords) > 0:
                    # Sample some points to avoid overcrowding
                    n_sample = min(20, len(valid_rel_coords))
                    sample_indices = np.random.choice(len(valid_rel_coords), n_sample, replace=False)
                    sample_rel_coords = valid_rel_coords[sample_indices]
                    
                    # Convert back to absolute coordinates
                    sample_abs_coords = sample_rel_coords + centroid
                    
                    ax2.scatter(sample_abs_coords[:, 0], sample_abs_coords[:, 1], 
                               alpha=0.4, s=10, c=f'C{emitter_idx}')
        
        ax2.scatter([], [], c='blue', s=80, marker='o', label='Centroids', edgecolors='black')
        ax2.scatter([], [], alpha=0.4, s=10, c='gray', label='Est. Positions (sample)')
        
    ax2.set_title('Ground Truth vs Centroids with Sample Estimates')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_evaluations(evaluation_directories: List[str], 
                       n_repetitions: int = 100,
                       max_distance: float = 0.5,
                       save_path_prefix: str = None):
    """
    Compare localization performance across multiple evaluation directories.
    Creates plots for mean per-emitter bias and precision across evaluations.
    
    Parameters:
    -----------
    evaluation_directories : List[str]
        List of evaluation directory paths to compare
    n_repetitions : int
        Number of localization runs to perform for each evaluation
    max_distance : float
        Maximum allowed distance for a valid match
    save_path_prefix : str, optional
        Prefix for saved plot files (will append "_bias.png" and "_precision.png")
    
    Returns:
    --------
    Dict containing results for each evaluation
    """
    
    results = {}
    evaluation_names = []
    mean_biases = []
    mean_biases_direction = []
    mean_precisions = []
    
    print(f"Processing {len(evaluation_directories)} evaluations...")
    
    for eval_idx, data_directory in enumerate(evaluation_directories):
        print(f"\nProcessing evaluation {eval_idx + 1}/{len(evaluation_directories)}: {data_directory}")
        
        # Extract evaluation name from directory path
        eval_name = Path(data_directory).name
        evaluation_names.append(eval_name)
        
        try:
            # Find all h5 files
            file_pattern = "run_000*.h5"
            h5_files = glob.glob(str(Path(data_directory) / file_pattern))
            h5_files.sort()
            
            if not h5_files:
                print(f"Warning: No h5 files found in {data_directory}")
                mean_biases.append(np.nan)
                mean_precisions.append(np.nan)
                continue
            
            # Load ground truth and maps from first file
            with h5py.File(h5_files[0], 'r') as f:
                ground_truth_positions = f['ground_truth_positions'][:]
                nr_em_map = f['nr_emitters_map'][:]
                G2_map = f['G2_diff_map'][:]
                count_map = f['photon_count_map'][:]
                metadata = dict(f.attrs)
            
            # Collect estimated positions from multiple runs
            estimated_positions_list = []
            
            reps = range(n_repetitions)
            for rep in reps:
                if rep < len(h5_files):
                    # Load from existing h5 file if available
                    with h5py.File(h5_files[rep], 'r') as f:
                        if 'estimated_positions' in f:
                            estimated_positions = f['estimated_positions'][:]
                        else:
                            # If no pre-computed positions, run localization
                            # estimated_positions = localize(I_meas=count_map, 
                            #                              Gd_meas=G2_map, 
                            #                              est_emitters=nr_em_map, 
                            #                              metadata=metadata, 
                            #                              psf_file='project/data/psf.json', 
                            #                              verbose=False)
                            estimated_positions_data = localize(I_meas=count_map, 
                                                                 Gd_meas=G2_map, 
                                                                 est_emitters=nr_em_map, 
                                                                 metadata=metadata, 
                                                                 psf_file='project/data/psf.json', 
                                                                 verbose=False)
                            estimated_positions_um = [transform_coordinates(x['y'], x['x'], metadata['area_size'], 
                                                                       metadata['positions'], metadata['pixel_size'], 
                                                                       direction='to_physical') 
                                                 for x in estimated_positions_data['emitters']]
                            estimated_positions_um = np.array(estimated_positions_um)
                else:
                    # Generate additional runs if needed
                    estimated_positions_data = localize(I_meas=count_map, 
                                                 Gd_meas=G2_map, 
                                                 est_emitters=nr_em_map, 
                                                 metadata=metadata, 
                                                 psf_file='project/data/psf.json', 
                                                 verbose=False)
                    estimated_positions_um = [transform_coordinates(x['x'], x['y'], metadata['area_size'], 
                                                               metadata['positions'], metadata['pixel_size'], 
                                                               direction='to_physical') 
                                         for x in estimated_positions_data['emitters']]
                    estimated_positions_um = np.array(estimated_positions_um)
                
                estimated_positions_list.append(estimated_positions_um)
            
            # Calculate statistics with relative coordinates
            stats = calculate_localization_stats_relative_coordinates(
                ground_truth_positions, estimated_positions_list, max_distance=max_distance
            )

            print(stats['TP_per_run'])
            print(stats['FP_per_run'])
            print(stats['FN_per_run'])
            # Compute Jaccard index
            Jaccard = []
            for i in range(len(stats['TP_per_run'])):
                TP = stats['TP_per_run'][i]
                FP = stats['FP_per_run'][i]
                FN = stats['FN_per_run'][i]
                Jaccard.append(TP / (TP + FP + FN) if (TP + FP + FN) > 0 else np.nan)
            print(f"Jaccard Index: {Jaccard}")
            
            # Store results
            results[eval_name] = stats
            
            # Extract mean bias and precision across all emitters
            valid_biases = [bias for bias in stats['mean_bias_per_emitter'] if not np.isnan(bias)]
            valid_precisions = [prec for prec in stats['precision_per_emitter'] if not np.isnan(prec)]

            valid_biases_directions = [bias_dir for bias_dir in stats['mean_bias_direction_per_emitter'] if not np.isnan(bias_dir)]
            
            mean_bias = np.mean(valid_biases) if valid_biases else np.nan
            mean_precision = np.mean(valid_precisions) if valid_precisions else np.nan

            mean_bias_direction = np.mean(valid_biases_directions) if valid_biases_directions else np.nan

            mean_biases.append(mean_bias)
            mean_precisions.append(mean_precision)
            
            print(f"  Mean bias: {mean_bias:.6f}")
            print(f"  Mean precision: {mean_precision:.6f}")
            print(f"  Number of emitters: {len(ground_truth_positions)}")
            print(f"  Number of runs: {len(estimated_positions_list)}")
            
        except Exception as e:
            print(f"Error processing {data_directory}: {str(e)}")
            mean_biases.append(np.nan)
            mean_precisions.append(np.nan)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Mean per-emitter bias
    mean_bias_nm = [bias*1000 for bias in mean_biases]
    ax1.plot(range(1, len(evaluation_names) + 1), mean_bias_nm, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_title('Mean per-Emitter Bias', fontsize=20, fontweight='bold')
    ax1.set_xlabel(r'Emitter density [$\mu m^{-2}$]', fontsize=20)
    ax1.set_ylabel('Mean Bias [nm]', fontsize=20)
    ax1.set_xticks(range(len(evaluation_names)))
    #ax1.set_xticklabels(evaluation_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on points
    for i, bias in enumerate(mean_bias_nm):
        if not np.isnan(bias):
            ax1.annotate(f'{bias:.4f}', (i, bias), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 2: Mean per-emitter precision
    mean_precisions_nm = [prec*1000 for prec in mean_precisions]  # Convert to nm
    ax2.plot(range(1, len(evaluation_names) + 1), mean_precisions_nm, 'o-', linewidth=2, markersize=8, color='orange')
    ax2.set_title('Mean per-Emitter Precision', fontsize=20, fontweight='bold')
    ax2.set_xlabel('Emitter Density [$\mu m^{-2}$]', fontsize=20)
    ax2.set_ylabel('Mean Precision (RMS) [nm]', fontsize=20)
    ax2.set_xticks(range(len(evaluation_names)))
    #ax2.set_xticklabels(evaluation_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    print("PRECISIONS")
    print(mean_precisions_nm)
    # Add value labels on points
    for i, precision in enumerate(mean_precisions_nm):
        if not np.isnan(precision):
            ax2.annotate(f'{precision:.10f}', (i, precision), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=12)
    
    plt.tight_layout()
    
    # Save plots if requested
    if save_path_prefix:
        bias_path = f"{save_path_prefix}_bias.png"
        precision_path = f"{save_path_prefix}_precision.png"
        
        # Save individual plots
        fig_bias, ax_bias = plt.subplots(1, 1, figsize=(10, 6))
        ax_bias.plot(range(len(evaluation_names)), mean_biases, 'o-', linewidth=2, markersize=8)
        ax_bias.set_title('Mean Per-Emitter Bias Across Evaluations', fontsize=14, fontweight='bold')
        #ax_bias.set_xlabel('Evaluation', fontsize=12)
        ax_bias.set_ylabel('Mean Bias', fontsize=12)
        ax_bias.set_xticks(range(len(evaluation_names)))
        #ax_bias.set_xticklabels(evaluation_names, rotation=45, ha='right')
        ax_bias.grid(True, alpha=0.3)
        # for i, bias in enumerate(mean_biases):
        #     if not np.isnan(bias):
        #         ax_bias.annotate(f'{bias:.4f}', (i, bias), textcoords="offset points", 
        #                        xytext=(0,10), ha='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(bias_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        fig_prec, ax_prec = plt.subplots(1, 1, figsize=(10, 6))
        ax_prec.plot(range(len(evaluation_names)), mean_precisions, 'o-', linewidth=2, markersize=8, color='orange')
        ax_prec.set_title('Mean Per-Emitter Precision Across Evaluations', fontsize=14, fontweight='bold')
        ax_prec.set_xlabel('Evaluation', fontsize=12)
        ax_prec.set_ylabel('Mean Precision (RMS)', fontsize=12)
        ax_prec.set_xticks(range(len(evaluation_names)))
        ax_prec.set_xticklabels(evaluation_names, rotation=45, ha='right')
        ax_prec.grid(True, alpha=0.3)
        # for i, precision in enumerate(mean_precisions):
        #     if not np.isnan(precision):
        #         ax_prec.annotate(f'{precision:.4f}', (i, precision), textcoords="offset points", 
        #                        xytext=(0,10), ha='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(precision_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save combined plot
        combined_path = f"{save_path_prefix}_combined.png"
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        
        print(f"\nPlots saved:")
        print(f"  Bias plot: {bias_path}")
        print(f"  Precision plot: {precision_path}")
        print(f"  Combined plot: {combined_path}")
    
    plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Evaluation':<30} {'Mean Bias':<12} {'Mean Precision':<15}")
    print("-" * 57)
    
    for i, eval_name in enumerate(evaluation_names):
        bias_str = f"{mean_biases[i]:.6f}" if not np.isnan(mean_biases[i]) else "NaN"
        prec_str = f"{mean_precisions[i]:.6f}" if not np.isnan(mean_precisions[i]) else "NaN"
        print(f"{eval_name:<30} {bias_str:<12} {prec_str:<15}")
    
    # Find best performing evaluation
    valid_indices = [i for i, (b, p) in enumerate(zip(mean_biases, mean_precisions)) 
                    if not (np.isnan(b) or np.isnan(p))]
    
    if valid_indices:
        best_bias_idx = min(valid_indices, key=lambda i: mean_biases[i])
        best_precision_idx = min(valid_indices, key=lambda i: mean_precisions[i])
        
        print(f"\nBest bias: {evaluation_names[best_bias_idx]} ({mean_biases[best_bias_idx]:.6f})")
        print(f"Best precision: {evaluation_names[best_precision_idx]} ({mean_precisions[best_precision_idx]:.6f})")
    
    return {
        'evaluation_names': evaluation_names,
        'mean_biases': mean_biases,
        'mean_precisions': mean_precisions,
        'detailed_results': results
    }


# Example usage:
if __name__ == "__main__":
    # List of evaluation directories to compare
    evaluation_directories = [
    "project/data/evaluation_20250804_125707/",
    "project/data/evaluation_20250804_125907/",
    "project/data/evaluation_20250804_130258/",
    "project/data/evaluation_20250804_130836/",
    "project/data/evaluation_20250804_131603/",
    "project/data/evaluation_20250804_132520/",
    "project/data/evaluation_20250804_133624/",
    "project/data/evaluation_20250804_134928/",
    "project/data/evaluation_20250804_140424/",
    "project/data/evaluation_20250804_142103/",
    ]
    
    # Run comparison
    comparison_results = compare_evaluations(
        evaluation_directories, 
        n_repetitions=100,  # Reduce for faster processing
        save_path_prefix="evaluation_comparison"
    )
    
    # Access results
    print(f"\nMean biases: {comparison_results['mean_biases']}")
    print(f"Mean precisions: {comparison_results['mean_precisions']}")


def compare_evaluations_multi_run(evaluation_directories: List[str], 
                                  run_patterns: List[str] = ['run_000*.h5', 'run_001*.h5', 'run_002*.h5'],
                                  n_repetitions: int = 100,
                                  max_distance: float = 0.5,
                                  x_positions_given: List[float] = None,
                                  metadata_given: Dict = None,
                                  plot: bool = False,
                                  verbose: bool = False,
                                  save_path_prefix: str = None):
    """
    Compare localization performance across multiple evaluation directories and multiple run patterns.
    Creates plots for mean per-emitter bias and precision with error bars across evaluations.
    
    Parameters:
    -----------
    evaluation_directories : List[str]
        List of evaluation directory paths to compare
    run_patterns : List[str]
        List of file patterns for different runs (e.g., ['run_000*.h5', 'run_001*.h5', 'run_002*.h5'])
    n_repetitions : int
        Number of localization runs to perform for each evaluation
    max_distance : float
        Maximum allowed distance for a valid match
    save_path_prefix : str, optional
        Prefix for saved plot files (will append "_bias.png" and "_precision.png")
    
    Returns:
    --------
    Dict containing results for each evaluation
    """
    
    results = {}
    evaluation_names = []
    all_biases = []  # Will store bias values for each evaluation across all runs
    all_biases_std = []

    all_biases_dir = []


    all_precisions = []  # Will store precision values for each evaluation across all runs
    all_precisions_std = []
    all_jaccards = []  # Will store Jaccard index values for each evaluation across all runs
    all_jaccards_std = []  # Will store Jaccard std values for each evaluation across all runs
    
    print(f"Processing {len(evaluation_directories)} evaluations with {len(run_patterns)} run patterns...")
    
    for eval_idx, data_directory in enumerate(evaluation_directories):
        print(f"\nProcessing evaluation {eval_idx + 1}/{len(evaluation_directories)}: {data_directory}")
        
        # Extract evaluation name from directory path
        eval_name = Path(data_directory).name
        evaluation_names.append(eval_name)
        
        # Store bias and precision values for this evaluation across all run patterns (so evaluation = specific emitter density); run for specific ground truth.
        eval_biases = []
        eval_biases_std = []
        eval_biases_dir = []
        eval_precisions = []
        eval_precisions_std = []
        eval_jaccard = []
        eval_jaccard_std = []
        
        #try:
        for run_pattern in run_patterns:
            print(f"  Processing pattern: {run_pattern}")
            
            # Find all h5 files for this pattern
            h5_files = glob.glob(str(Path(data_directory) / run_pattern))
            h5_files.sort()
            
            if not h5_files:
                print(f"    Warning: No h5 files found for pattern {run_pattern} in {data_directory}")
                continue
            
            # Load ground truth and maps from first file
            with h5py.File(h5_files[0], 'r') as f:
                ground_truth_positions = f['ground_truth_positions'][:]
                nr_em_map = f['nr_emitters_map'][:]
                G2_map = f['G2_diff_map'][:]
                count_map = f['photon_count_map'][:]
                if metadata_given:
                    metadata = metadata_given
                else:
                    metadata = dict(f.attrs)

            # Collect estimated positions from multiple runs
            estimated_positions_list = []
            
            reps = range(n_repetitions)

            #progressbar
            pbar = tqdm(total=n_repetitions, desc="Processing repetitions")
            for rep in reps:
                #progressbar
                pbar.update(1)

                # if rep < len(h5_files):
                #     # Load from existing h5 file if available [I skipped this]
                #     with h5py.File(h5_files[rep], 'r') as f:
                #         if 'estimated_positions' in f:
                #             estimated_positions = f['estimated_positions'][:]
                #         else:
                #             # If no pre-computed positions, run localization
                #             estimated_positions = localize(I_meas=count_map, 
                #                                          Gd_meas=G2_map, 
                #                                          est_emitters=nr_em_map, 
                #                                          metadata=metadata, 
                #                                          psf_file='project/data/psf.json', 
                #                                          verbose=False)
                #             estimated_positions = [transform_coordinates(x['y'], x['x'], metadata['area_size'], 
                #                                                        metadata['positions'], metadata['pixel_size'], 
                #                                                        direction='to_physical') 
                #                                  for x in estimated_positions['emitters']]
                #             estimated_positions = np.array(estimated_positions)
                #else:
                    # Generate additional runs if needed
                estimated_positions_data = localize(I_meas=count_map, 
                                                Gd_meas=G2_map, 
                                                est_emitters=nr_em_map, 
                                                metadata=metadata, 
                                                psf_file='project/data/psf.json',
                                                plot=plot,
                                                verbose=verbose)
                estimated_positions_um = [transform_coordinates(x['x'], x['y'], metadata['area_size'], 
                                                            metadata['positions'], metadata['pixel_size'], 
                                                            direction='to_physical') 
                                        for x in estimated_positions_data['emitters']]
                #estimated_positions = [(x['y'], x['x']) for x in estimated_positions['emitters']]
                estimated_positions_um = np.array(estimated_positions_um)
            
                estimated_positions_list.append(estimated_positions_um)
            
            pbar.close()  # Close the progress bar

            ##########################
            # # After finishing repetitions for one run_pattern:
            # estimated_positions_all = np.vstack(estimated_positions_list)  # stack all runs together

            # print(f"Estimated pos list {estimated_positions_list}")
            # #first estimated_position for each repetition in the list
            # estimated_positions_first = [pos[0] for pos in estimated_positions_list]

            # fig, ax = plt.subplots(figsize=(6, 6))
            # ax.scatter(ground_truth_positions[:, 0], ground_truth_positions[:, 1],
            #         c='red', marker='x', s=80, label='Ground truth')

            # ax.scatter(estimated_positions_first[:, 0], estimated_positions_first[:, 1],
            #         c='blue', alpha=0.3, s=15, label='Localizations')
            # print("estimated positions first:")
            # print(estimated_positions_first)
            # ax.set_title(f"Localization Spread - {eval_name}, {run_pattern}", fontsize=14, fontweight='bold')
            # ax.set_xlabel("X position [µm]", fontsize=12)
            # ax.set_ylabel("Y position [µm]", fontsize=12)
            # ax.legend()
            # ax.axis("equal")
            # ax.grid(alpha=0.3)

            all_estimated_positions = np.vstack(estimated_positions_list)

            print("ESTPOS")
            print(all_estimated_positions)
            print("GTPOS")
            print(ground_truth_positions)
            # --- Step 1: Pick a cluster near a reference point (e.g. the first ground truth position) ---
            reference_point = all_estimated_positions[0]  # change index if you want another reference
            radius = 0.1  #, which is 0.1 µm, adjust this to control zoom region

            # Filter positions within radius of the reference point
            distances = np.linalg.norm(all_estimated_positions - reference_point, axis=1)
            cluster_positions = all_estimated_positions[distances < radius]

                        #standard deviation of cluster:
            print(f"STD: {np.std(cluster_positions, axis=0)}")

            # --- Step 2: Plot full view ---
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))

            # show the background image g2

            # all_estimated_positions = [transform_coordinates(x['y'], x['x'], metadata['area_size'], 
            #                                 metadata['positions'], metadata['pixel_size'], 
            #                                 direction='to_pixel') for x in all_estimated_positions]
            # ground_truth_positions = [transform_coordinates(x[1], x[0], metadata['area_size'], 
            #                                                 metadata['positions'], metadata['pixel_size'], 
            #                                                 direction='to_pixel') for x in ground_truth_positions]
            ax[0].imshow(G2_map, cmap='viridis', extent = [-1, 1, -1, 1], origin='lower')
            #ax[0].set_title("Background Image", fontsize=14, fontweight='bold')
            import matplotlib.ticker as ticker
            # Full view
            for pos in ground_truth_positions:
                ax[0].scatter(pos[1], pos[0],
                        c='red', marker='x', s=80)
            for pos in all_estimated_positions:
                ax[0].scatter(pos[1], pos[0],
                        c='blue', alpha=0.3, s=15)
            # ax[0].scatter(all_estimated_positions[:, 0], all_estimated_positions[:, 1],
            #             c='blue', alpha=0.3, s=15, label='Localizations')
            ax[0].set_title(f"Full View", fontsize=20, fontweight='bold')
            ax[0].set_xlabel("X position [µm]", fontsize=20)
            ax[0].set_ylabel("Y position [µm]", fontsize=20)
            ax[0].tick_params(axis='x', labelsize=14)
            ax[0].tick_params(axis='y', labelsize=14)

            ax[0].axis("equal")
            #ax[0].grid(alpha=0.3)
            #ax[0].legend()

            # --- Step 3: Zoom-in view ---
            #ax[1].scatter(reference_point[0], reference_point[1],
            #            c='red', marker='x', s=80, label='Ground truth')
            ax[1].scatter(cluster_positions[:, 0], cluster_positions[:, 1],
                        c='blue', alpha=0.7, s=40, label='Clustered localizations')

            # Calculate appropriate zoom limits based on data
            x_min, x_max = cluster_positions[:, 0].min(), cluster_positions[:, 0].max()
            y_min, y_max = cluster_positions[:, 1].min(), cluster_positions[:, 1].max()
            x_range = x_max - x_min
            y_range = y_max - y_min
            padding = max(x_range, y_range) * 0.2  # 20% padding

            ax[1].set_xlim(x_min - padding, x_max + padding)
            ax[1].set_ylim(y_min - padding, y_max + padding)

            # Fix the scientific notation issue
            ax[1].ticklabel_format(style='plain', axis='both', )  # Disable scientific notation
            ax[1].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.7f'))  # 3 decimal places
            ax[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.7f'))
            ax[1].tick_params(axis='x', labelsize=11)
            ax[1].tick_params(axis='y', labelsize=15)

            ax[1].set_title("Zoom-in on cluster", fontsize=20, fontweight='bold')
            ax[1].set_xlabel("X position [µm]", fontsize=20)
            ax[1].set_ylabel("Y position [µm]", fontsize=20)
            ax[1].axis("equal")
            ax[1].grid(alpha=0.3)
            ax[1].legend()

            plt.tight_layout()
            plt.subplots_adjust(wspace=0.3)  # Add some space between subplots

            #savefig

            plt.savefig("test_full_view.png")

            plt.show()


            if save_path_prefix:
                spread_path = "test_spread.png"
                plt.savefig(spread_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  Spread plot saved: {spread_path}")
            else:
                plt.show()

            

            ## 
            if plot:
                # quick plot with ground truth and estimations of first one
                gt_pos = ground_truth_positions
                loc_pos = estimated_positions_list[0]
                #gt_pos = gt_pos/metadata['pixel_size'] + metadata['positions']/2.01 # divide by 2.01 instead of 2 just so the plot is nicer.
                #loc_pos = loc_pos/metadata['pixel_size'] + metadata['positions']/2.01 # divide by 2.01 instead of 2 just so the plot is nicer.



                # gt_pos = [transform_coordinates(x[1], x[0], metadata['area_size'], 
                #                                             metadata['positions'], metadata['pixel_size'], 
                #                                             direction='to_pixel') for x in gt_pos]
                # gt_pos = np.array(gt_pos)
                # loc_pos = [transform_coordinates(x[1], x[0], metadata['area_size'], 
                #                                             metadata['positions'], metadata['pixel_size'], 
                #                                             direction='to_pixel') for x in loc_pos]
                # loc_pos = np.array(loc_pos)

                print("GT POS AND THEN LOC POS")
                print(gt_pos)
                print(loc_pos)

                print(f"Number of detected emitters: {len(gt_pos)}; {len(loc_pos)} localized")
                fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
                #red map
                img = ax.imshow(G2_map, cmap='viridis', extent = [-metadata['area_size'][0]/2, metadata['area_size'][0]/2, -metadata['area_size'][1]/2, metadata['area_size'][1]/2])
                # Plot ground truth positions
                ax.scatter(gt_pos[:, 0], gt_pos[:, 1], s=65, c='red', marker = 'o', label='Ground Truth', alpha=1)
                # Plot localized positions
                ax.scatter(loc_pos[:, 0], loc_pos[:, 1], s=95, c='red', marker = 'x', label='Localized', alpha=1)

                #ax.set_xticks([])
                #ax.set_yticks([])

                plt.colorbar(img, label='Count', fraction=0.046, pad=0.04)
                plt.show()
            ##########################
            
            # Calculate statistics with relative coordinates
            stats = calculate_localization_stats_relative_coordinates(
                ground_truth_positions, estimated_positions_list, max_distance=max_distance
            )

            TP = stats['TP_per_run']
            FP = stats['FP_per_run']
            FN = stats['FN_per_run']
            # Compute Jaccard index
            Jaccard = [] # list of jaccard indices for each localization
            for i in range(len(TP)):
                TP_val = TP[i]
                FP_val = FP[i]
                FN_val = FN[i]
                Jaccard.append(TP_val / (TP_val + FP_val + FN_val) if (TP_val + FP_val + FN_val) > 0 else np.nan)
                
            print(f"Jaccard over all repetitions: {Jaccard}")
            Jaccard_mean = np.nanmean(Jaccard) # mean jaccard over n_repetitions of localizations. so this corresponds to the mean jaccard for a specific ground truth and emitter density
            Jaccard_std = np.nanstd(Jaccard) # this is the std dev for a specific datapoint

            # Extract mean bias and precision across all emitters for this run pattern
            valid_biases = [bias for bias in stats['mean_bias_per_emitter'] if not np.isnan(bias)]
            valid_biases_dir = [bias_dir for bias_dir in stats['mean_bias_direction_per_emitter'] if not np.isnan(bias_dir)]
            valid_precisions = [prec for prec in stats['precision_per_emitter'] if not np.isnan(prec)]

            if valid_biases and valid_precisions:
                mean_bias = np.mean(valid_biases)
                mean_bias_dir = np.mean(valid_biases_dir)
                std_bias = np.std(valid_biases, ddof=1)  # Use sample std
                mean_precision = np.mean(valid_precisions)
                std_precision = np.std(valid_precisions, ddof=1)  # Use sample std
                print(f"std precision: {std_precision}")
                eval_biases.append(mean_bias)
                eval_biases_std.append(std_bias)

                eval_precisions.append(mean_precision)
                eval_precisions_std.append(std_precision)

                eval_biases_dir.append(mean_bias_dir)

                eval_jaccard.append(Jaccard_mean) #mean Jaccard over all locaizations for each ground truth
                eval_jaccard_std.append(Jaccard_std) #standard deviations of all localizations for each ground truth
                
                print(f"    Mean bias: {mean_bias:.6f}")
                print(f"    Mean precision: {mean_precision:.6f}")
                print(f"    Jaccard Index: {Jaccard_mean:.6f}")
                print(f"    Jaccard Index Std: {Jaccard_std:.6f}")
            else:
                print(f"Warning: No valid biases found for run {run_pattern} and rep {rep}")
                eval_biases.append(np.nan)
                eval_biases_std.append(np.nan)

                eval_precisions.append(np.nan)
                eval_precisions_std.append(np.nan)

                eval_jaccard.append(np.nan) #mean Jaccard over all locaizations for each ground truth
                eval_jaccard_std.append(np.nan) #standard deviations of all localizations for each ground truth

                eval_biases_dir.append(np.nan)

            # Store detailed results for this run pattern
            pattern_key = f"{eval_name}_{run_pattern.replace('*.h5', '')}"
            results[pattern_key] = stats

        print(f"eval precisions std: {eval_precisions_std}")
        
        # Store all bias and precision values for this evaluation
        all_biases.append(eval_biases)
        all_biases_std.append(eval_biases_std)

        all_biases_dir.append(eval_biases_dir)

        all_precisions.append(eval_precisions)
        all_precisions_std.append(eval_precisions_std)

        all_jaccards.append(eval_jaccard) #all mean jaccard indices for each ground truth in each evaluation
        all_jaccards_std.append(eval_jaccard_std) #all std devs of jaccard indices for each ground truth in each evaluation

        if eval_biases and eval_precisions:
            print(f"  Overall - Mean bias: {np.mean(eval_biases):.6f} ± {np.std(eval_biases):.6f}")
            print(f"  Overall - Mean precision: {np.mean(eval_precisions):.6f} ± {np.std(eval_precisions):.6f}")
            print(f"  Overall - Mean Jaccard Index: {np.mean(eval_jaccard):.6f} ± {np.std(eval_jaccard):.6f}")
            print(f"  Number of run patterns processed: {len(eval_biases)}")
            
        #except Exception as e:
        # print(f"Error processing {data_directory}: {str(e)}")
        # all_biases.append([])
        # all_biases_std.append([])

        # all_precisions.append([])
        # all_precisions_std.append([])

        # all_jaccards.append([])
        # all_jaccards_std.append([])

    # Calculate mean and standard error for each evaluation
    mean_biases = []
    bias_errors = []
    mean_precisions = []
    precision_errors = []
    
    for i, (biases, precisions) in enumerate(zip(all_biases, all_precisions)):
        if biases and precisions:
            mean_biases.append(np.mean(biases))
            bias_errors.append(np.std(biases))  # Use standard deviation as error
            mean_precisions.append(np.mean(precisions))
            precision_errors.append(np.std(precisions))
        else:
            mean_biases.append(np.nan)
            bias_errors.append(0)
            mean_precisions.append(np.nan)
            precision_errors.append(0)
    
    #############################################
    # Create plots with error bars
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # # Plot 1: Mean per-emitter bias with error bars
    # mean_bias_nm = [bias*1000 for bias in mean_biases]
    # bias_errors_nm = [err*1000 for err in bias_errors]
    
    # x_positions = range(len(evaluation_names))
    # ax1.errorbar(x_positions, mean_bias_nm, yerr=bias_errors_nm, 
    #             fmt='o-', linewidth=2, markersize=8, color='blue', capsize=5, capthick=2)
    # ax1.set_title('Mean Centroid Bias', fontsize=20, fontweight='bold')
    # ax1.set_xlabel(r'Emitter density [$\mu m^{-2}$]', fontsize=20)
    # ax1.set_ylabel('Bias [nm]', fontsize=20)
    # ax1.set_xticks(range(len(evaluation_names)))
    # ax1.grid(True, alpha=0.3)

    # # Plot 2: Mean per-emitter precision with error bars
    # mean_precisions_nm = [prec*1000 for prec in mean_precisions]
    # precision_errors_nm = [err*1000 for err in precision_errors]
    
    # ax2.errorbar(x_positions, mean_precisions_nm, yerr=precision_errors_nm,
    #             fmt='o-', linewidth=2, markersize=8, color='orange', capsize=5, capthick=2)
    # ax2.set_title('Mean RMS Precision', fontsize=20, fontweight='bold')
    # ax2.set_xlabel('Emitter Density [$\mu m^{-2}$]', fontsize=20)
    # ax2.set_ylabel('Precision [nm]', fontsize=20)
    # ax2.set_xticks(range(len(evaluation_names)))
    # ax2.grid(True, alpha=0.3)

    #############################################################
    # New plot code
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # # Plot 1: Mean per-emitter bias with error bars for 3 ground truths
    # all_biases contains for each evaluation, the number of ground truths as entries with the mean bias
    # x positions starts at 1
    if not x_positions_given:
        x_positions = range(1, len(evaluation_names) + 1)
    else:
        x_positions = x_positions_given

    max_len = max(len(b) for b in all_biases)

    for run_idx in range(max_len):
        all_biases_run = [
            b[run_idx] if run_idx < len(b) else np.nan 
            for b in all_biases
        ]
        all_biases_std_run = [
            b[run_idx] if run_idx < len(b) else np.nan
            for b in all_biases_std
        ]

    #for run_idx in range(len(all_biases[0])):
        #all biases has 10 entries with each one
        #all_biases_run = [eval[run_idx] for eval in all_biases]
        #all_biases_std_run = [eval[run_idx] for eval in all_biases_std]

        mean_bias_nm = [bias*1000 for bias in all_biases_run]
        error_bias_nm = [err*1000 for err in all_biases_std_run]
        ax1.errorbar(x_positions, mean_bias_nm, yerr=error_bias_nm,
                    fmt='o-', linewidth=2, markersize=8, label=f'Run {run_idx+1}' if run_idx < 5 else None,
                    capsize=5, capthick=2)
    ax1.set_title('Mean Centroid Biases', fontsize=20, fontweight='bold')

    ax1.set_xlabel(r'Emitter density [$\mu m^{-2}$]', fontsize=20)

    #ax1.set_xlabel('Dead Time [ns]', fontsize=20)

    ax1.set_ylabel('Bias [nm]', fontsize=20)
    ax1.set_xticks(x_positions)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mean per-emitter precision with error bars for 3 ground truths
    
    for run_idx in range(max_len):
        all_precisions_run = [
            b[run_idx] if run_idx < len(b) else np.nan 
            for b in all_precisions
        ]
        all_precisions_std_run = [
            b[run_idx] if run_idx < len(b) else np.nan
            for b in all_precisions_std
        ]

    #for run_idx in range(len(all_precisions[0])):

        #all_precisions_run = [eval[run_idx] for eval in all_precisions]
        #all_precisions_std_run = [eval[run_idx] for eval in all_precisions_std]

        mean_precisions_nm = [prec*1000 for prec in all_precisions_run]
        error_precisions_nm = [err*1000 for err in all_precisions_std_run]
        ax2.errorbar(x_positions, mean_precisions_nm, yerr=error_precisions_nm,
                    fmt='o-', linewidth=2, markersize=8, label=f'Run {run_idx+1}' if run_idx < 5 else None,
                    capsize=5, capthick=2)
    ax2.set_title('Mean RMS Precisions', fontsize=20, fontweight='bold')

    ax2.set_xlabel('Emitter Density [$\mu m^{-2}$]', fontsize=20)

    #ax2.set_xlabel('Dead Time [ns]', fontsize=20)

    ax2.set_ylabel('Precision (RMS) [nm]', fontsize=20)
    ax2.set_xticks(x_positions)
    ax2.grid(True, alpha=0.3)
    
    print("PRECISIONS WITH ERRORS")
    for i, (prec, err) in enumerate(zip(mean_precisions_nm, error_precisions_nm)):
        print(f"{evaluation_names[i]}: {prec:.6f} ± {err:.6f}")


    print("JACCARD INDICES")
    for i, jaccard in enumerate(all_jaccards):
        if jaccard:
            print(f"{evaluation_names[i]}: {np.mean(jaccard):.6f} ± {np.std(jaccard):.6f}")
        else:
            print(f"{evaluation_names[i]}: No valid Jaccard indices")
    
    # Add value labels on points
    # for i, (precision, error) in enumerate(zip(mean_precisions_nm, precision_errors_nm)):
    #     if not np.isnan(precision):
    #         ax2.annotate(f'{precision:.6f}±{error:.6f}', (i, precision), textcoords="offset points", 
    #                     xytext=(0,15), ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save plots if requested
    if save_path_prefix:
        combined_path = f"{save_path_prefix}_multi_run_combined.png"
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        
        # Save individual plots
        fig_bias, ax_bias = plt.subplots(1, 1, figsize=(10, 6))
        ax_bias.errorbar(x_positions, mean_biases, yerr=bias_errors, 
                        fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2)
        ax_bias.set_title('Mean Per-Emitter Bias Across Evaluations (Multi-Run)', fontsize=14, fontweight='bold')
        ax_bias.set_ylabel('Mean Bias', fontsize=12)
        ax_bias.set_xticks(x_positions)
        ax_bias.grid(True, alpha=0.3)
        plt.tight_layout()
        bias_path = f"{save_path_prefix}_multi_run_bias.png"
        plt.savefig(bias_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        fig_prec, ax_prec = plt.subplots(1, 1, figsize=(10, 6))
        ax_prec.errorbar(x_positions, mean_precisions, yerr=precision_errors,
                        fmt='o-', linewidth=2, markersize=8, color='orange', capsize=5, capthick=2)
        ax_prec.set_title('Mean Per-Emitter Precision Across Evaluations (Multi-Run)', fontsize=14, fontweight='bold')
        ax_prec.set_xlabel('Evaluation', fontsize=12)
        ax_prec.set_ylabel('Mean Precision (RMS)', fontsize=12)
        ax_prec.set_xticks(x_positions)
        #ax_prec.set_xticklabels(evaluation_names, rotation=45, ha='right')
        ax_prec.grid(True, alpha=0.3)
        plt.tight_layout()
        precision_path = f"{save_path_prefix}_multi_run_precision.png"
        plt.savefig(precision_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nPlots saved:")
        print(f"  Combined plot: {combined_path}")
        print(f"  Bias plot: {bias_path}")
        print(f"  Precision plot: {precision_path}")
    
    #plt.show()


    #x_positions = range(1, len(all_jaccards) + 1)
    # Also create seperate plot for Jaccard indices and show plot as well

    x_positions_jaccard = range(1, len(all_jaccards) + 1)

    fig_jaccard, ax_jaccard = plt.subplots(1, 1, figsize=(10, 6))
    #for i, (jaccard, jaccard_std) in enumerate(zip(all_jaccards, all_jaccards_std)):
        #if not x_positions_given:
        #x_positions_jaccard = [i + 1] * len(jaccard)  # Use i+1 for x positions to match evaluation index
        #else:
        #    x_positions_jaccard = x_positions_given
        #if jaccard:
    # for run_idx in range(len(all_jaccards[0])):
    #     all_jaccards_run = [eval[run_idx] for eval in all_jaccards]
    #     all_jaccards_std_run = [eval[run_idx] for eval in all_jaccards_std]


    for run_idx in range(max_len):
        all_jaccards_run = [
            b[run_idx] if run_idx < len(b) else np.nan 
            for b in all_jaccards   
        ]
        all_jaccards_std_run = [
            b[run_idx] if run_idx < len(b) else np.nan
            for b in all_jaccards_std
        ]

        ax_jaccard.errorbar(x_positions_jaccard, all_jaccards_run, yerr=all_jaccards_std_run,
                                fmt='o-', linewidth=2, markersize=8, label=evaluation_names[i],
                                capsize=5, capthick=2)

        #if jaccard:
            # ax_jaccard.errorbar(x_positions_jaccard, jaccard, yerr=jaccard_std,
            #                     fmt='o-', linewidth=2, markersize=8, label=evaluation_names[i],
            #                     capsize=5, capthick=2)
        #else:
        #    ax_jaccard.plot(x_positions_jaccard[i], np.nan, 'o', markersize=8, label=evaluation_names[i])
    ax_jaccard.set_title('Mean Jaccard Index', fontsize=14, fontweight='bold')

    ax_jaccard.set_xlabel('Emitter Density [$\mu m^{-2}$]', fontsize=12)

    #ax_jaccard.set_xlabel('Dead Time [ns]', fontsize=12)

    ax_jaccard.set_ylabel('Mean Jaccard Index', fontsize=12)
    ax_jaccard.set_xticks(x_positions_jaccard)
    #ax_jaccard.set_xticklabels(evaluation_names, rotation=45, ha='right')
    ax_jaccard.grid(True, alpha=0.3)
    plt.tight_layout()
    jaccard_path = f"{save_path_prefix}_multi_run_jaccard.png"
    plt.savefig(jaccard_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary
    print(f"\n{'='*80}")
    print("MULTI-RUN EVALUATION COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Evaluation':<30} {'Mean Bias':<20} {'Mean Precision':<20}")
    print("-" * 80)
    
    for i, eval_name in enumerate(evaluation_names):
        if not np.isnan(mean_biases[i]) and not np.isnan(mean_precisions[i]):
            bias_str = f"{mean_biases[i]:.6f} ± {bias_errors[i]:.6f}"
            prec_str = f"{mean_precisions[i]:.6f} ± {precision_errors[i]:.6f}"
            print(f"{eval_name:<30} {bias_str:<20} {prec_str:<20}")
        else:
            print(f"{eval_name:<30} {'NaN':<20} {'NaN':<20}")
    
    # Find best performing evaluation (considering mean values)
    valid_indices = [i for i, (b, p) in enumerate(zip(mean_biases, mean_precisions)) 
                    if not (np.isnan(b) or np.isnan(p))]
    
    if valid_indices:
        best_bias_idx = min(valid_indices, key=lambda i: mean_biases[i])
        best_precision_idx = min(valid_indices, key=lambda i: mean_precisions[i])
        
        print(f"\nBest bias: {evaluation_names[best_bias_idx]} ({mean_biases[best_bias_idx]:.6f} ± {bias_errors[best_bias_idx]:.6f})")
        print(f"Best precision: {evaluation_names[best_precision_idx]} ({mean_precisions[best_precision_idx]:.6f} ± {precision_errors[best_precision_idx]:.6f})")
    
    return {
        'evaluation_names': evaluation_names,
        'mean_biases': mean_biases,
        'bias_errors': bias_errors,
        'mean_precisions': mean_precisions,
        'precision_errors': precision_errors,
        'all_biases': all_biases,
        'all_biases_std': all_biases_std,
        'all_biases_dir': all_biases_dir,
        'all_precisions': all_precisions,
        'all_precisions_std': all_precisions_std,
        'all_jaccards': all_jaccards,
        'all_jaccard_std': all_jaccards_std,
        #'detailed_results': results
    }


# # Example usage:
# if __name__ == "__main__":
#     # List of evaluation directories to compare
#     evaluation_directories = [
#         "project/data/evaluation_20250804_125707/",
#         "project/data/evaluation_20250804_125907/",
#         "project/data/evaluation_20250804_130258/",
#         "project/data/evaluation_20250804_130836/",
#         "project/data/evaluation_20250804_131603/",
#         "project/data/evaluation_20250804_132520/",
#         "project/data/evaluation_20250804_133624/",
#         "project/data/evaluation_20250804_134928/",
#         "project/data/evaluation_20250804_140424/",
#         "project/data/evaluation_20250804_142103/",
#     ]
    
#     # Run comparison with multiple run patterns
#     comparison_results = compare_evaluations_multi_run(
#         evaluation_directories, 
#         run_patterns=['run_000*.h5', 'run_001*.h5', 'run_002*.h5'],
#         n_repetitions=100,
#         save_path_prefix="multi_run_evaluation_comparison"
#     )
    
#     # Access results
#     print(f"\nMean biases: {comparison_results['mean_biases']}")
#     print(f"Bias errors: {comparison_results['bias_errors']}")
#     print(f"Mean precisions: {comparison_results['mean_precisions']}")
#     print(f"Precision errors: {comparison_results['precision_errors']}")

# Updated example usage:
# if __name__ == "__main__":
#     # Example data generation (replace with your actual data)
#     np.random.seed(42)
    
#     # Simulate 100 localization runs with some noise and random ordering
#     n_repetitions = 100
#     estimated_positions_list = []

#     data_directory = "project/data/evaluation_20250716_151526/"
#     data_directory =  "project/data/evaluation_20250731_131304/"
#     data_directory =  "project/data/evaluation_20250731_134316/"
#     data_directory = "project/data/evaluation_20250804_091259/"
#     data_directory = "project/data/evaluation_20250804_125707/"

#     # Find all h5 files
#     file_pattern = "run_000*.h5"
#     h5_files = glob.glob(str(Path(data_directory) / file_pattern))
#     h5_files.sort()
    
#     with h5py.File(h5_files[0], 'r') as f:
#         ground_truth_positions = f['ground_truth_positions'][:]
#         nr_em_map = f['nr_emitters_map'][:]
#         G2_map = f['G2_diff_map'][:]
#         count_map = f['photon_count_map'][:]
#         metadata = dict(f.attrs)
    
#     # Repeat localization n_repetitions times
#     for rep in range(n_repetitions):
#         # Localize emitters
#         estimated_positions = localize(I_meas=count_map, 
#                                        Gd_meas=G2_map, 
#                                        est_emitters=nr_em_map, 
#                                        metadata=metadata, 
#                                         psf_file='project/data/psf.json', 
#                                         verbose=False
#                                     )
#         estimated_positions = [transform_coordinates(x['y'], x['x'], metadata['area_size'], metadata['positions'], metadata['pixel_size'], direction='to_physical') for x in estimated_positions['emitters']]
#         estimated_positions = np.array(estimated_positions)
#         estimated_positions_list.append(estimated_positions)
    
#     # Calculate statistics with relative coordinates
#     print("Calculating localization statistics with relative coordinates...")
#     stats = calculate_localization_stats_relative_coordinates(ground_truth_positions, estimated_positions_list, max_distance=0.5)
    
#     # Print results
#     print_detailed_relative_coordinate_results(stats)
    
#     # Plot results
#     plot_positions_overview(ground_truth_positions, estimated_positions_list, count_map, metadata, stats=stats, max_runs_to_plot=30) #save_path="localization_overview.png")
    
#     # Access specific statistics
#     print(f"\nQuick access examples:")
#     print(f"Overall localization precision: {stats['overall_precision']:.10f}")
#     print(f"Average per-emitter precision: {stats['mean_precision']:.10f} ± {stats['std_precision']:.4f}")
#     print(f"Average assignment cost: {np.mean(stats['total_costs_per_run']):.4f}")
    
#     # Verify that mean relative coordinates are close to zero
#     print(f"\nVerification that means are close to zero:")
#     for i, mean_coord in enumerate(stats['mean_relative_coords_per_emitter']):
#         if not np.isnan(mean_coord).any():
#             distance_from_zero = np.linalg.norm(mean_coord)
#             print(f"Emitter {i+1}: mean relative position magnitude = {distance_from_zero:.6f}")

def compare_evaluations_multi_run_revised(evaluation_directories: List[str], 
                                          run_patterns: List[str] = ['run_000*.h5', 'run_001*.h5', 'run_002*.h5'],
                                          max_distance: float = 0.5,
                                          x_positions_given: List[float] = None,
                                          metadata_given: Dict = None,
                                          plot: bool = False,
                                          plot_localization: bool = False,
                                          verbose: bool = False,
                                          save_path_prefix: str = None):
    """
    Compare localization performance with simplified bias and precision calculation.
    
    For each run: perform 1 localization, calculate mean distance to matched ground truth
    For each density: report mean and std of these mean distances across runs
    
    Metrics computed:
    - Mean of mean distances (overall accuracy across runs)
    - Standard deviation of mean distances (run-to-run variability)
    - Mean of within-run standard deviations (typical spread within localizations)
    
    Parameters:
    -----------
    evaluation_directories : List[str]
        List of evaluation directory paths to compare
    run_patterns : List[str]
        List of file patterns for different runs
    max_distance : float
        Maximum allowed distance for a valid match
    x_positions_given : List[float], optional
        X-axis positions for plotting
    metadata_given : Dict, optional
        Metadata to use instead of loading from file
    plot : bool
        Whether to show detailed plots during localization
    verbose : bool
        Whether to print detailed information
    save_path_prefix : str, optional
        Prefix for saved JSON file
    
    Returns:
    --------
    Dict containing computed metrics and saves to JSON file
    """
    
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'max_distance': max_distance,
            'run_patterns': run_patterns,
            'evaluation_directories': evaluation_directories,
            'x_positions_given': x_positions_given
        },
        'data': {}
    }
    
    evaluation_names = []
    
    print(f"Processing {len(evaluation_directories)} evaluations with {len(run_patterns)} run patterns...")
    
    for eval_idx, data_directory in enumerate(evaluation_directories):
        print(f"\nProcessing evaluation {eval_idx + 1}/{len(evaluation_directories)}: {data_directory}")
        
        # Extract evaluation name from directory path
        eval_name = Path(data_directory).name
        evaluation_names.append(eval_name)
        
        # Initialize results for this evaluation
        results['data'][eval_name] = {
            'evaluation_index': eval_idx,
            'directory': data_directory,
            'run_results': {},  # Results for each run pattern
            'summary_metrics': {}  # Aggregated metrics across all runs
        }
        
        # Store results for each run pattern
        all_mean_distances = []  # Mean distance for each run
        all_std_distances = []   # Standard deviation within each run
        all_RMSE = []
        all_jaccard_indices = [] # Jaccard index for each run
        all_tp_counts = []       # True positives for each run
        all_fp_counts = []       # False positives for each run
        all_fn_counts = []       # False negatives for each run

        all_estimated_positions_runs = []  # collect all estimated positions per run
        all_ground_truth_positions = None
        G2_map_for_plot = None
        
        #try:
        for run_idx, run_pattern in enumerate(run_patterns):
            if verbose:
                print(f"  Processing pattern: {run_pattern}")
            
            # Find all h5 files for this pattern
            h5_files = glob.glob(str(Path(data_directory) / run_pattern))
            h5_files.sort()
            
            if not h5_files:
                print(f"    Warning: No h5 files found for pattern {run_pattern} in {data_directory}")
                continue
            
            # Load ground truth and maps from first file
            with h5py.File(h5_files[0], 'r') as f:
                ground_truth_positions = f['ground_truth_positions'][:]
                nr_em_map = f['nr_emitters_map'][:]
                G2_map = f['G2_diff_map'][:]
                count_map = f['photon_count_map'][:]
                if metadata_given:
                    metadata = metadata_given
                else:
                    metadata = dict(f.attrs)

                if all_ground_truth_positions is None:
                    all_ground_truth_positions = ground_truth_positions
                    G2_map_for_plot = G2_map
            
            # Perform ONE localization for this run
            estimated_positions = localize(I_meas=count_map, 
                                            Gd_meas=G2_map, 
                                            est_emitters=nr_em_map, 
                                            metadata=metadata, 
                                            psf_file='project/data/psf.json',
                                            plot=plot,
                                            verbose=verbose)

            #print("Localization good")
            
            estimated_positions = [transform_coordinates(x['y'], x['x'], metadata['area_size'], 
                                                        metadata['positions'], metadata['pixel_size'], 
                                                        direction='to_physical') 
                                    for x in estimated_positions['emitters']]
            estimated_positions = np.array(estimated_positions)

            all_estimated_positions_runs.append(estimated_positions)
            
            # Match positions using Hungarian algorithm
            matched_positions, assignment_indices, total_cost, match_distances = match_positions_hungarian(
                ground_truth_positions, estimated_positions, max_distance
            )
            
            # Calculate metrics for this run
            valid_distances = match_distances[match_distances < np.inf]

            ######## Computing precision ############################################
            # Initialize container outside run loop
            if eval_idx == 0 and run_idx == 0:
                all_matched_coords = {i: [] for i in range(len(ground_truth_positions))}


            # After matching for this run:
            for gt_idx, est_idx in enumerate(assignment_indices):
                if est_idx >= 0:  # valid match
                    all_matched_coords[gt_idx].append(estimated_positions[est_idx])
            ###########################################################################

            if len(valid_distances) > 0:
                mean_distance = np.mean(valid_distances)
                std_distance = np.std(valid_distances, ddof=1) if len(valid_distances) > 1 else 0.0
                RMSE = np.sqrt(np.mean(np.square(valid_distances)))
            else:
                mean_distance = np.nan
                std_distance = np.nan
                RMSE = np.nan

            # Calculate TP, FP, FN
            TP = np.sum(assignment_indices >= 0)
            FN = np.sum(assignment_indices == -1)
            FP = len(estimated_positions) - TP
            
            # Calculate Jaccard index
            if TP + FP + FN > 0:
                jaccard = TP / (TP + FP + FN)
            else:
                jaccard = 0.0
            
            # Store results for this run
            run_key = f"run_{run_idx:03d}"
            results['data'][eval_name]['run_results'][run_key] = {
                'run_pattern': run_pattern,
                'n_ground_truth': int(len(ground_truth_positions)),
                'n_estimated': int(len(estimated_positions)),
                'n_matched': int(TP),
                'mean_distance': float(mean_distance) if not np.isnan(mean_distance) else None,
                'RMSE': float(RMSE) if not np.isnan(RMSE) else None,
                'std_distance': float(std_distance) if not np.isnan(std_distance) else None,
                'TP': int(TP),
                'FP': int(FP),
                'FN': int(FN),
                'jaccard_index': float(jaccard),
                'total_matching_cost': float(total_cost),
                'individual_distances': [float(d) for d in valid_distances] if len(valid_distances) > 0 else []
            }
            #print(f"Run {run_key}: {results['data'][eval_name]['run_results'][run_key]}")
            # Collect for summary statistics
            if not np.isnan(mean_distance):
                all_mean_distances.append(mean_distance)
            if not np.isnan(std_distance):
                all_std_distances.append(std_distance)
            if not np.isnan(RMSE):
                all_RMSE.append(RMSE)
            all_jaccard_indices.append(jaccard)
            all_tp_counts.append(TP)
            all_fp_counts.append(FP)
            all_fn_counts.append(FN)
            
            if verbose:
                print(f"    Run {run_idx}: Mean distance = {mean_distance:.6f}, "
                        f"Std distance = {std_distance:.6f}, Jaccard = {jaccard:.6f}")


        ############# calculating precisions ################################################
        per_emitter_precisions = {}
        for gt_idx, coords in all_matched_coords.items():
            coords = np.array(coords)
            if len(coords) > 1:
                centroid = coords.mean(axis=0)
                residuals = coords - centroid
                sigma_x = residuals[:,0].std(ddof=1)
                sigma_y = residuals[:,1].std(ddof=1)
                per_emitter_precisions[gt_idx] = {
                    'sigma_x': float(sigma_x),
                    'sigma_y': float(sigma_y)
                }
            else:
                per_emitter_precisions[gt_idx] = None

        # Aggregate
        all_sigmas_x = [v['sigma_x'] for v in per_emitter_precisions.values() if v is not None]
        all_sigmas_y = [v['sigma_y'] for v in per_emitter_precisions.values() if v is not None]
        mean_sigma_x = np.mean(all_sigmas_x) if all_sigmas_x else None
        mean_sigma_y = np.mean(all_sigmas_y) if all_sigmas_y else None
        #######################################################################################
        
        #print(all_mean_distances)
        # Calculate summary metrics across all runs for this evaluation
        if all_mean_distances:
            mean_of_means = np.mean(all_mean_distances)  # Overall accuracy
            std_of_means = np.std(all_mean_distances, ddof=1) if len(all_mean_distances) > 1 else 0.0  # Run-to-run variability
        else:
            mean_of_means = np.nan
            std_of_means = np.nan
        
        if all_std_distances:
            mean_of_stds = np.mean(all_std_distances)  # Typical within-run spread
            std_of_stds = np.std(all_std_distances, ddof=1) if len(all_std_distances) > 1 else 0.0
        else:
            mean_of_stds = np.nan
            std_of_stds = np.nan

        if all_RMSE:
            mean_RMSE = np.mean(all_RMSE)
            std_RMSE = np.std(all_RMSE, ddof=1) if len(all_RMSE) > 1 else 0.0
        else:
            mean_RMSE = np.nan
            std_RMSE = np.nan

        # Jaccard statistics
        mean_jaccard = np.mean(all_jaccard_indices) if all_jaccard_indices else np.nan
        std_jaccard = np.std(all_jaccard_indices, ddof=1) if len(all_jaccard_indices) > 1 else 0.0
        
        # Detection statistics
        mean_tp = np.mean(all_tp_counts) if all_tp_counts else np.nan
        mean_fp = np.mean(all_fp_counts) if all_fp_counts else np.nan
        mean_fn = np.mean(all_fn_counts) if all_fn_counts else np.nan
        
        # Store summary metrics
        results['data'][eval_name]['summary_metrics'] = {
            'n_runs_processed': len(all_mean_distances),
            'mean_of_mean_distances': float(mean_of_means) if not np.isnan(mean_of_means) else None,
            'std_of_mean_distances': float(std_of_means) if not np.isnan(std_of_means) else None,
            'mean_of_within_run_stds': float(mean_of_stds) if not np.isnan(mean_of_stds) else None,
            'std_of_within_run_stds': float(std_of_stds) if not np.isnan(std_of_stds) else None,
            'mean_RMSE': float(mean_RMSE) if not np.isnan(mean_RMSE) else None,
            'std_RMSE': float(std_RMSE) if not np.isnan(std_RMSE) else None,
            'mean_jaccard_index': float(mean_jaccard) if not np.isnan(mean_jaccard) else None,
            'std_jaccard_index': float(std_jaccard) if not np.isnan(std_jaccard) else None,
            'mean_tp': float(mean_tp) if not np.isnan(mean_tp) else None,
            'mean_fp': float(mean_fp) if not np.isnan(mean_fp) else None,
            'mean_fn': float(mean_fn) if not np.isnan(mean_fn) else None,
            'all_mean_distances': [float(d) for d in all_mean_distances],
            'all_within_run_stds': [float(d) for d in all_std_distances],
            'all_jaccard_indices': [float(j) for j in all_jaccard_indices],
            'per_emitter_precisions': per_emitter_precisions,
            'mean_sigma_x': mean_sigma_x,
            'mean_sigma_y': mean_sigma_y
        }
        
        print(f"  Summary for {eval_name}:")
        print(f"    Mean of mean distances: {mean_of_means:.6f} ± {std_of_means:.6f}")
        print(f"    Mean of within-run stds: {mean_of_stds:.6f} ± {std_of_stds:.6f}")
        print(f"    Mean Jaccard index: {mean_jaccard:.6f} ± {std_jaccard:.6f}")

        # === Optional plotting ===
        if plot_localization and all_ground_truth_positions is not None:
            zoom_window = 0.01
            zoom_emitter_index = 0
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Left: overview
            area_size = metadata['area_size']
            img = ax1.imshow(count_map, cmap='viridis', origin='lower', extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2])
            ax1.scatter(all_ground_truth_positions[:,1], all_ground_truth_positions[:,0],
                        c='red', marker='o', label='Ground Truth', s=20)
            for run_est in all_estimated_positions_runs:
                ax1.scatter(run_est[:,1], run_est[:,0], c='white', marker='x', s=10, alpha=0.3)
            ax1.set_title(f"Full view")

            ax1.set_xlabel("X Position [µm]")
            ax1.set_ylabel("Y Position [µm]")
            ax1.legend()
            
            # Right: zoom on one emitter
            zoom_gt = all_ground_truth_positions[zoom_emitter_index]
            img2 = ax2.imshow(G2_map_for_plot, cmap='viridis', origin='lower')
            ax2.scatter([zoom_gt[0]], [zoom_gt[1]], c='red', marker='o', s=100, label='Ground Truth')
            for j, run_est in enumerate(all_estimated_positions_runs):
                if j == 0:  # first run only
                    ax2.scatter(run_est[:,0], run_est[:,1], 
                                c='white', marker='x', s=30, alpha=0.5,
                                label='Estimated Positions')
                else:
                    ax2.scatter(run_est[:,0], run_est[:,1], 
                                c='white', marker='x', s=30, alpha=0.5)


            ax2.set_xlim(zoom_gt[0]-zoom_window, zoom_gt[0]+zoom_window)
            ax2.set_ylim(zoom_gt[1]-zoom_window, zoom_gt[1]+zoom_window)

            ax2.set_xticks(np.linspace(zoom_gt[0]-zoom_window, zoom_gt[0]+zoom_window, 5))
            ax2.set_yticks(np.linspace(zoom_gt[1]-zoom_window, zoom_gt[1]+zoom_window, 5))

            ax2.set_xlabel("X Position [µm]")
            ax2.set_ylabel("Y Position [µm]")
            # CRITICAL: Force equal aspect ratio and matching tick spacing
            ax2.set_aspect('equal', adjustable='box')
            ax2.set_title(f"Zoom-in on emitter")
            
            ax2.legend()

            plt.colorbar(img, label=r'$\Delta G^{(2)}$', fraction=0.046, pad=0.04)
            plt.colorbar(img2, label=r'$\Delta G^{(2)}$', fraction=0.046, pad=0.04)

            plt.tight_layout()

            plt.savefig(f"localization_plot_{eval_name}.png", dpi=300)
            plt.show()
        if plot_localization and all_ground_truth_positions is not None:
            zoom_window = 0.01
            zoom_emitter_index = 0
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
            # Left: overview
            area_size = metadata['area_size']
            img = ax1.imshow(G2_map_for_plot, cmap='viridis', origin='lower', 
                            extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2])
            ax1.scatter(all_ground_truth_positions[:,1], all_ground_truth_positions[:,0],
                        c='red', marker='o', label='Ground Truth', s=20)
            for run_est in all_estimated_positions_runs:
                ax1.scatter(run_est[:,1], run_est[:,0], c='white', marker='x', s=10, alpha=0.3)
            ax1.set_title(f"Full view")
            ax1.set_xlabel("X Position [µm]")
            ax1.set_ylabel("Y Position [µm]")
            ax1.legend()
            # Ensure equal aspect ratio for overview
            ax1.set_aspect('equal', adjustable='box')
        
            # Right: zoom on one emitter
            zoom_gt = all_ground_truth_positions[zoom_emitter_index]
            
            # Calculate zoom extent in physical coordinates
            zoom_x_min = zoom_gt[1] - zoom_window  # Note: using [1] for x-coordinate
            zoom_x_max = zoom_gt[1] + zoom_window
            zoom_y_min = zoom_gt[0] - zoom_window  # Note: using [0] for y-coordinate  
            zoom_y_max = zoom_gt[0] + zoom_window
            
            # Display the zoomed image with correct extent
            img2 = ax2.imshow(G2_map_for_plot, cmap='viridis', origin='lower',
                            extent=[-area_size[0]/2, area_size[0]/2, -area_size[1]/2, area_size[1]/2])
            
            # Plot ground truth and estimated positions in physical coordinates
            ax2.scatter([zoom_gt[1]], [zoom_gt[0]], c='red', marker='o', s=100, label='Ground Truth')
            
            for j, run_est in enumerate(all_estimated_positions_runs):
                if j == 0:  # first run only
                    ax2.scatter(run_est[:,1], run_est[:,0],
                                c='white', marker='x', s=30, alpha=0.5,
                                label='Estimated Positions')
                else:
                    ax2.scatter(run_est[:,1], run_est[:,0],
                                c='white', marker='x', s=30, alpha=0.5)
            
            # Set zoom limits in physical coordinates
            ax2.set_xlim(zoom_x_min, zoom_x_max)
            ax2.set_ylim(zoom_y_min, zoom_y_max)
            
            # CRITICAL: Force equal aspect ratio and matching tick spacing
            ax2.set_aspect('equal', adjustable='box')
            
            # Optional: Set explicit tick spacing for better readability
            tick_spacing = zoom_window / 4  # 4 ticks across the zoom window
            x_ticks = np.arange(zoom_x_min, zoom_x_max + tick_spacing, tick_spacing)
            y_ticks = np.arange(zoom_y_min, zoom_y_max + tick_spacing, tick_spacing)
            ax2.set_xticks(x_ticks)
            ax2.set_yticks(y_ticks)
            
            # Format tick labels to show more precision in zoom
            ax2.ticklabel_format(style='plain', useOffset=False)
            
            ax2.set_xlabel("X Position [µm]")
            ax2.set_ylabel("Y Position [µm]")
            ax2.set_title(f"Zoom-in on emitter (±{zoom_window*1000:.0f} nm)")
            ax2.legend()
            
            # Add colorbars
            plt.colorbar(img, ax=ax1, label=r'$\Delta G^{(2)}$', fraction=0.046, pad=0.04)
            plt.colorbar(img2, ax=ax2, label=r'$\Delta G^{(2)}$', fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            #plt.savefig(f"localization_plot_{eval_name}.png", dpi=300, bbox_inches='tight')
            plt.show()
                
        #except Exception as e:
        #    print(f"Error processing {data_directory}: {str(e)}")
        #    results['data'][eval_name]['error'] = str(e)
    
    # Add evaluation names to metadata
    results['metadata']['evaluation_names'] = evaluation_names
    
    # Save to JSON file
    if save_path_prefix:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"{save_path_prefix}_localization_metrics_{timestamp}.json"
        
        with open(json_filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {json_filename}")
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("LOCALIZATION PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Evaluation':<30} {'Mean±Std [µm]':<20} {'Within-Run±Std [µm]':<20} {'Jaccard±Std':<15}")
    print("-" * 85)
    
    for eval_name in evaluation_names:
        if eval_name in results['data'] and 'summary_metrics' in results['data'][eval_name]:
            metrics = results['data'][eval_name]['summary_metrics']
            
            mean_of_means = metrics['mean_of_mean_distances']
            std_of_means = metrics['std_of_mean_distances']
            mean_of_stds = metrics['mean_of_within_run_stds']
            std_of_stds = metrics['std_of_within_run_stds']
            mean_jaccard = metrics['mean_jaccard_index']
            std_jaccard = metrics['std_jaccard_index']
            
            # Format with appropriate precision
            if mean_of_means is not None:
                mean_str = f"{mean_of_means*1000:.1f}±{std_of_means*1000:.1f}"
            else:
                mean_str = "N/A"
            
            if mean_of_stds is not None:
                std_str = f"{mean_of_stds*1000:.1f}±{std_of_stds*1000:.1f}"
            else:
                std_str = "N/A"
            
            if mean_jaccard is not None:
                jaccard_str = f"{mean_jaccard:.3f}±{std_jaccard:.3f}"
            else:
                jaccard_str = "N/A"
            
            print(f"{eval_name[:29]:<30} {mean_str:<20} {std_str:<20} {jaccard_str:<15}")
        else:
            print(f"{eval_name[:29]:<30} {'Error':<20} {'Error':<20} {'Error':<15}")
    
    return results


###############################
##### Plotting functions ######
###############################

def plot_example_localizations(evaluation_directories: List[str],
                               example_indices: List[int] = [0, 2, 4, 8],
                               run_pattern: str = 'run_000*.h5',
                               map_type: str = 'G2_map',
                               metadata_given: Dict = None,
                               save_path: str = None,
                               verbose: bool = False,
                               crop_image: bool = True,
                               plot_title: str = ""):
    """
    Display example localization results in a 2x2 grid for selected evaluations.
    
    Parameters:
    -----------
    evaluation_directories : List[str]
        List of evaluation directory paths
    example_indices : List[int]
        Indices of evaluations to display (default: [0, 2, 4, 8] for 1st, 3rd, 5th, 9th)
    run_pattern : str
        File pattern for h5 files (default: 'run_000*.h5')
    map_type : str
        Type of map to display as background ('G2_map', 'nr_emitters_map', or 'photon_count_map')
    metadata_given : Dict, optional
        Metadata to use instead of loading from file
    save_path : str, optional
        Path to save the plot
    verbose : bool
        Whether to print detailed information
    plot_title : str
        Title for the overall plot
    
    Returns:
    --------
    Dict containing localization results for each example
    """
    
    # Validate inputs
    if len(example_indices) != 4:
        raise ValueError("example_indices must contain exactly 4 indices for 2x2 grid")
    
    # Check that all indices are valid
    max_idx = max(example_indices)
    if max_idx >= len(evaluation_directories):
        raise ValueError(f"Index {max_idx} is out of range for {len(evaluation_directories)} evaluations")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    results = {}
    
    dead_times = [5,20,35,50]

    for plot_idx, eval_idx in enumerate(example_indices):
        ax = axes[plot_idx]
        data_directory = evaluation_directories[eval_idx]
        eval_name = Path(data_directory).name
        
        if verbose:
            print(f"Processing evaluation {eval_idx + 1}: {eval_name}")
        
        try:
            # Find h5 files for this pattern
            h5_files = glob.glob(str(Path(data_directory) / run_pattern))
            h5_files.sort()
            
            if not h5_files:
                ax.text(0.5, 0.5, f'No files found\n{eval_name}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=48)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Load data from first file
            with h5py.File(h5_files[0], 'r') as f:
                ground_truth_positions = f['ground_truth_positions'][:]
                nr_em_map = f['nr_emitters_map'][:]
                G2_map = f['G2_diff_map'][:]
                count_map = f['photon_count_map'][:]
                if metadata_given:
                    metadata = metadata_given
                else:
                    metadata = dict(f.attrs)
            
            # Perform one localization
            estimated_positions_data = localize(I_meas=count_map, 
                                         Gd_meas=G2_map, 
                                         est_emitters=nr_em_map, 
                                         metadata=metadata, 
                                         psf_file='project/data/psf.json', 
                                         verbose=verbose)
            
            # Convert to physical coordinates
            print(estimated_positions_data['emitters'])
            print(f"GROUND TRUTH: {ground_truth_positions}")
            estimated_positions_um = [transform_coordinates(x['y'], x['x'], metadata['area_size'], 
                                                       metadata['positions'], metadata['pixel_size'], 
                                                       direction='to_physical') 
                                 for x in estimated_positions_data['emitters']]
            estimated_positions_um = np.array(estimated_positions_um)
            print(f"LOCALIZED: {estimated_positions_um}")
            # Convert positions to pixel coordinates for plotting
            # gt_pos_pixel = [transform_coordinates(x[1], x[0], metadata['area_size'], 
            #                                     metadata['positions'], metadata['pixel_size'], 
            #                                     direction='to_pixel') for x in ground_truth_positions]
            # gt_pos_pixel = np.array(gt_pos_pixel)
            # print(f"GT POS PIXEL: {gt_pos_pixel}")
            
            # if len(estimated_positions_um) > 0:
            #     loc_pos_pixel = [transform_coordinates(x[1], x[0], metadata['area_size'], 
            #                                         metadata['positions'], metadata['pixel_size'], 
            #                                         direction='to_pixel') for x in estimated_positions]
            #     loc_pos_pixel = np.array(loc_pos_pixel)
            # else:
            #     loc_pos_pixel = np.array([]).reshape(0, 2)

            # print(f"Positions estimated: {loc_pos_pixel}")

            gt_pos_pixel = ground_truth_positions
            loc_pos_pixel = estimated_positions_um
            
            # Select and display the chosen map
            if map_type == 'G2_map':
                display_map = G2_map
                cmap = 'viridis'
                map_label = r'$\Delta G^{(2)}$'
            elif map_type == 'nr_emitters_map':
                display_map = nr_em_map
                #clip
                display_map = np.clip(display_map, 0, 10)
                cmap = 'hot'
                map_label = 'Estimated emitters'
            elif map_type == 'photon_count_map':
                display_map = count_map
                cmap = 'hot'
                map_label = 'Photon counts'
            else:
                raise ValueError(f"Unknown map_type: {map_type}")
            
            # Display the map with cropping (10 pixels on each side)
            if crop_image:
                crop_pixels = 10
                h, w = display_map.shape
                if h > 2*crop_pixels and w > 2*crop_pixels:
                    cropped_map = display_map[crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]
                    im = ax.imshow(cropped_map, cmap=cmap, origin='lower', 
                                extent=[crop_pixels, w-crop_pixels, crop_pixels, h-crop_pixels])
                    
                    # Adjust positions for cropping
                    gt_pos_cropped = gt_pos_pixel.copy()
                    if len(loc_pos_pixel) > 0:
                        loc_pos_cropped = loc_pos_pixel.copy()
                    else:
                        loc_pos_cropped = np.array([]).reshape(0, 2)
                else:
                    # If image too small to crop, use original
                    display_map = np.rot90(display_map, k=1)
                    im = ax.imshow(display_map, cmap=cmap, origin='lower')
                    gt_pos_cropped = gt_pos_pixel
                    loc_pos_cropped = loc_pos_pixel
            else:
                # If not cropping, use original
                display_map = np.rot90(display_map, k=1)
                im = ax.imshow(display_map, cmap=cmap, extent = [-metadata['area_size'][0]/2, metadata['area_size'][0]/2, -metadata['area_size'][1]/2, metadata['area_size'][1]/2])
                gt_pos_cropped = gt_pos_pixel
                loc_pos_cropped = loc_pos_pixel

            
            # Plot ground truth positions (red circles)
            if len(gt_pos_cropped) > 0:
                ax.scatter(gt_pos_cropped[:, 0], gt_pos_cropped[:, 1], 
                          s=100, c='red', marker='o', label='Ground Truth', 
                          alpha=0.8, edgecolors='white', linewidths=1.5)
            
            # Plot localized positions (white X marks)
            if len(loc_pos_cropped) > 0:
                ax.scatter(loc_pos_cropped[:, 0], loc_pos_cropped[:, 1], 
                          s=120, c='white', marker='x', label='Localized', 
                          alpha=1.0, linewidths=2)
            
            # Calculate basic statistics
            n_gt = len(ground_truth_positions)
            n_loc = len(estimated_positions_um)
            
            # Quick matching for TP/FP/FN calculation
            if len(estimated_positions_um) > 0 and len(ground_truth_positions) > 0:
                _, assignment_indices, _, _ = match_positions_hungarian(
                    ground_truth_positions, estimated_positions_um, max_distance=0.020
                )
                TP = np.sum(assignment_indices >= 0)
                FN = np.sum(assignment_indices == -1)
                FP = len(estimated_positions_um) - TP
                
                if TP + FP + FN > 0:
                    jaccard = TP / (TP + FP + FN)
                else:
                    jaccard = 0.0
            else:
                TP, FP, FN = 0, n_loc, n_gt
                jaccard = 0.0
            
            # Set title with statistics
            #title = f'Eval {eval_idx+1}: {n_gt} GT, {n_loc} Loc\nTP:{TP}, FP:{FP}, FN:{FN}, J:{jaccard:.3f}'
            # per micrometer squared (so like mu m^-2)
            title = fr"Emitter Density: {eval_idx+1} $μm^{-2}$"
            #title = f"Dead Time: {dead_times[eval_idx]}"
            ax.set_title(title, fontsize=26, fontweight='bold')
            
            # Add colorbar with larger text
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(map_label, fontsize=24)
            cbar.ax.tick_params(labelsize=24)
            
            # Add legend only to first subplot
            if plot_idx == 0:
                ax.legend(loc='upper left', fontsize=24, framealpha=0.8)
            
            # Remove ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Store results
            results[eval_name] = {
                'ground_truth_positions': ground_truth_positions,
                'estimated_positions': estimated_positions_um,
                'n_ground_truth': n_gt,
                'n_localized': n_loc,
                'TP': TP,
                'FP': FP,
                'FN': FN,
                'jaccard_index': jaccard,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Error processing evaluation {eval_idx + 1} ({eval_name}): {str(e)}")
            ax.text(0.5, 0.5, f'Error loading\n{eval_name}\n{str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=40)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Overall title
    fig.suptitle(plot_title, fontsize=24, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("EXAMPLE LOCALIZATION SUMMARY")
        print(f"{'='*60}")
        print(f"{'Evaluation':<15} {'GT':<4} {'Loc':<4} {'TP':<3} {'FP':<3} {'FN':<3} {'Jaccard':<8}")
        print("-" * 60)
        
        for i, eval_idx in enumerate(example_indices):
            if eval_idx < len(evaluation_directories):
                eval_name = Path(evaluation_directories[eval_idx]).name
                if eval_name in results:
                    r = results[eval_name]
                    print(f"{eval_name[:14]:<15} {r['n_ground_truth']:<4} {r['n_localized']:<4} "
                          f"{r['TP']:<3} {r['FP']:<3} {r['FN']:<3} {r['jaccard_index']:<8.3f}")
                else:
                    print(f"{eval_name[:14]:<15} {'--':<4} {'--':<4} {'--':<3} {'--':<3} {'--':<3} {'--':<8}")
    
    return results

def plot_example_localizations(evaluation_directories: List[str],
                               example_indices: List[int] = [0, 2, 4, 8],
                               run_pattern: str = 'run_000*.h5',
                               map_type: str = 'photon_count',
                               metadata_given: Dict = None,
                               save_path: str = None,
                               verbose: bool = False,
                               crop_image: bool = True,
                               plot_title: str = ""):
    """
    Display example localization results in a 4x2 grid:
    - Left column: photon count map
    - Right column: G² map
    Each row corresponds to one evaluation (emitter density).
    """
    
    if len(example_indices) != 4:
        raise ValueError("example_indices must contain exactly 4 indices for 4 rows")
    
    max_idx = max(example_indices)
    if max_idx >= len(evaluation_directories):
        raise ValueError(f"Index {max_idx} is out of range for {len(evaluation_directories)} evaluations")
    
    fig, axes = plt.subplots(4, 2, figsize=(18, 24))
    
    results = {}
    dead_times = [5, 20, 35, 50]  # if needed

    for row_idx, eval_idx in enumerate(example_indices):
        data_directory = evaluation_directories[eval_idx]
        eval_name = Path(data_directory).name

        if verbose:
            print(f"Processing evaluation {eval_idx + 1}: {eval_name}")
        
        try:
            h5_files = glob.glob(str(Path(data_directory) / run_pattern))
            h5_files.sort()
            
            if not h5_files:
                for col in range(2):
                    ax = axes[row_idx, col]
                    ax.text(0.5, 0.5, f'No files found\n{eval_name}', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=32)
                    ax.set_xticks([])
                    ax.set_yticks([])
                continue
            
            with h5py.File(h5_files[0], 'r') as f:
                ground_truth_positions = f['ground_truth_positions'][:]
                nr_em_map = f['nr_emitters_map'][:]
                G2_map = f['G2_diff_map'][:]
                count_map = f['photon_count_map'][:]
                metadata = metadata_given if metadata_given else dict(f.attrs)
            
            estimated_positions_data = localize(I_meas=count_map, 
                                                Gd_meas=G2_map, 
                                                est_emitters=nr_em_map, 
                                                metadata=metadata, 
                                                psf_file='project/data/psf.json', 
                                                verbose=verbose)
            
            estimated_positions_um = [transform_coordinates(x['y'], x['x'], metadata['area_size'], 
                                                            metadata['positions'], metadata['pixel_size'], 
                                                            direction='to_physical') 
                                      for x in estimated_positions_data['emitters']]
            estimated_positions_um = np.array(estimated_positions_um)
            
            gt_pos_pixel = ground_truth_positions
            loc_pos_pixel = estimated_positions_um
            
            # Loop over two columns: left = intensity, right = G²
            for col_idx, (display_map, cmap, map_label) in enumerate([
                (count_map, "hot", "Photon counts"),
                (G2_map, "viridis", r'$\Delta G^{(2)}$')
            ]):
                ax = axes[row_idx, col_idx]
                
                if crop_image:
                    crop_pixels = 10
                    h, w = display_map.shape
                    if h > 2*crop_pixels and w > 2*crop_pixels:
                        cropped_map = display_map[crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]
                        im = ax.imshow(cropped_map, cmap=cmap, origin='lower', 
                                       extent=[crop_pixels, w-crop_pixels, crop_pixels, h-crop_pixels])
                        gt_pos_cropped = gt_pos_pixel.copy()
                        loc_pos_cropped = loc_pos_pixel.copy()
                    else:
                        display_map = np.rot90(display_map, k=1)
                        im = ax.imshow(display_map, cmap=cmap, origin='lower')
                        gt_pos_cropped = gt_pos_pixel
                        loc_pos_cropped = loc_pos_pixel
                else:
                    display_map = np.rot90(display_map, k=1)
                    im = ax.imshow(display_map, cmap=cmap,
                                   extent=[-metadata['area_size'][0]/2, metadata['area_size'][0]/2,
                                           -metadata['area_size'][1]/2, metadata['area_size'][1]/2])
                    gt_pos_cropped = gt_pos_pixel
                    loc_pos_cropped = loc_pos_pixel
                
                if len(gt_pos_cropped) > 0:
                    ax.scatter(gt_pos_cropped[:, 0], gt_pos_cropped[:, 1], 
                               s=100, c='red', marker='o', label='Ground Truth', 
                               alpha=0.8, edgecolors='white', linewidths=1.5)
                
                if len(loc_pos_cropped) > 0:
                    ax.scatter(loc_pos_cropped[:, 0], loc_pos_cropped[:, 1], 
                               s=120, c='white', marker='x', label='Localized', 
                               alpha=1.0, linewidths=2)
                
                # Add colorbar
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax)
                cbar.set_label(map_label, fontsize=20)
                cbar.ax.tick_params(labelsize=18)
                
                # Add subplot titles
                if col_idx == 0:
                    ax.set_title("Intensity", fontsize=20, fontweight='bold')
                else:
                    ax.set_title(r"$\Delta G^{(2)}$", fontsize=20, fontweight='bold')
                
                if row_idx == 0 and col_idx == 0:
                    ax.legend(loc='upper left', fontsize=18, framealpha=0.8)
                
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Add row label (emitter density)
            fig.text(0.16, 
                     (4 - row_idx - 0.5) / 4,  # normalized y position for row center
                     fr"Emitter Density: {eval_idx+1} $μm^{{-2}}$", 
                     #fr"Dead Time: {dead_times[eval_idx]} ns",
                     va='center', ha='right', fontsize=22, fontweight='bold', rotation=90)
            
            results[eval_name] = {
                'ground_truth_positions': ground_truth_positions,
                'estimated_positions': estimated_positions_um,
                'metadata': metadata
            }
            
        except Exception as e:
            for col in range(2):
                ax = axes[row_idx, col]
                ax.text(0.5, 0.5, f'Error loading\n{eval_name}\n{str(e)[:50]}...', 
                        ha='center', va='center', transform=ax.transAxes, fontsize=28)
                ax.set_xticks([])
                ax.set_yticks([])
    
    fig.suptitle(plot_title, fontsize=26, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0.08, 0, 1, 0.97])  # leave space for row labels
    plt.subplots_adjust(top=0.97)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Plot saved to: {save_path}")
    
    plt.show()
    return results



# Convenience function for your specific use case
def plot_evaluation_examples(evaluation_directories: List[str],
                           save_path_prefix: str = None,
                           map_type: str = 'G2_map',
                           crop_image: bool = False):
    """
    Plot examples from 1st, 3rd, 5th, and 9th evaluations as requested.
    
    Parameters:
    -----------
    evaluation_directories : List[str]
        List of evaluation directory paths
    save_path_prefix : str, optional
        Prefix for saved plot files
    map_type : str
        Type of map to display ('G2_map', 'nr_emitters_map', or 'photon_count_map')
    
    Returns:
    --------
    Dict containing localization results
    """
    
    # Use indices [0, 2, 4, 8] for 1st, 3rd, 5th, 9th evaluations
    example_indices = [0, 2, 4, 8]
    #example_indices = [0,1,2,3]
    
    # Make sure we don't exceed the available evaluations
    max_available = len(evaluation_directories) - 1
    example_indices = [idx for idx in example_indices if idx <= max_available]
    
    if len(example_indices) < 4:
        print(f"Warning: Only {len(evaluation_directories)} evaluations available. "
              f"Showing examples for indices: {example_indices}")
        # Pad with available indices if needed
        while len(example_indices) < 4 and len(example_indices) < len(evaluation_directories):
            next_idx = max(example_indices) + 1
            if next_idx < len(evaluation_directories):
                example_indices.append(next_idx)
            else:
                break
    
    save_path = f"{save_path_prefix}_examples.png" if save_path_prefix else None
    
    return plot_example_localizations(
        evaluation_directories=evaluation_directories,
        example_indices=example_indices[:4],  # Ensure we only take 4
        run_pattern='run_000*.h5',
        map_type=map_type,
        save_path=save_path,
        verbose=False,
        crop_image=crop_image,
        plot_title=f""
    )