import os
import sys

# If you know the exact path to your project root:
#project_root = 'home/rensvandam/mep/'
# os.chdir(project_root)

# If you know the exact path to your project root:
project_root = 'C:/Users/rensv/Onedrive - Delft University of Technology/MEP-RensPad/SPAD SMLM/spad-smlm/'
os.chdir(project_root)

# Add the project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from project.model.coherence_from_data import auto_coherence, show_coherence, coherence
from project.model.detection import show_photons, Spad23, Spad512, merge_photons
from project.model.sample import Alexa647
from project.model.setup import Setup, ScanningSetup, WidefieldSetup

from project.simulations.compute_localization_bias_precision import compare_evaluations, compare_evaluations_multi_run

import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from abc import ABC, abstractmethod


np.random.seed(42)

np.random.seed(42)

# Set the global font to be DejaVu Sans, size 10 (or any other font you prefer)
plt.rcParams['font.family'] = 'serif'  # Options: 'serif', 'sans-serif', 'monospace'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman']
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']

# Font sizes for different elements
plt.rcParams['font.size'] = 20          # Base font size
plt.rcParams['axes.titlesize'] = 14     # Title font size
plt.rcParams['axes.labelsize'] = 15     # X and Y label font size
plt.rcParams['xtick.labelsize'] = 15    # X tick label font size
plt.rcParams['ytick.labelsize'] = 15    # Y tick label font size
plt.rcParams['legend.fontsize'] = 18    # Legend font size
plt.rcParams['figure.titlesize'] = 16   # Figure title font size

# Font weight
plt.rcParams['axes.labelweight'] = 'normal'  # Options: 'normal', 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# Figure and axes settings
plt.rcParams['figure.figsize'] = (8, 6)     # Default figure size (width, height)
plt.rcParams['figure.dpi'] = 100            # Figure resolution
plt.rcParams['savefig.dpi'] = 300           # Saved figure resolution (high quality for thesis)
plt.rcParams['savefig.format'] = 'pdf'      # Default save format (PDF is vector format, good for thesis)
plt.rcParams['savefig.bbox'] = 'tight'      # Remove extra whitespace when saving

# Line and marker settings
plt.rcParams['lines.linewidth'] = 2.0       # Default line width
plt.rcParams['lines.markersize'] = 6        # Default marker size
plt.rcParams['lines.markeredgewidth'] = 1.0 # Marker edge width

# Axes settings
plt.rcParams['axes.linewidth'] = 1.2        # Axes border line width
plt.rcParams['axes.spines.top'] = False     # Remove top spine
plt.rcParams['axes.spines.right'] = False   # Remove right spine
plt.rcParams['axes.grid'] = False            # Enable grid by default
plt.rcParams['grid.alpha'] = 0.3            # Grid transparency
plt.rcParams['grid.linewidth'] = 0.8        # Grid line width

# Tick settings
plt.rcParams['xtick.major.size'] = 5        # X major tick size
plt.rcParams['xtick.minor.size'] = 3        # X minor tick size
plt.rcParams['ytick.major.size'] = 5        # Y major tick size
plt.rcParams['ytick.minor.size'] = 3        # Y minor tick size
plt.rcParams['xtick.major.width'] = 1.2     # X major tick width
plt.rcParams['xtick.minor.width'] = 0.8     # X minor tick width
plt.rcParams['ytick.major.width'] = 1.2     # Y major tick width
plt.rcParams['ytick.minor.width'] = 0.8     # Y minor tick width
plt.rcParams['xtick.direction'] = 'in'      # Tick direction: 'in', 'out', 'inout'
plt.rcParams['ytick.direction'] = 'in'

# Legend settings
plt.rcParams['legend.frameon'] = True       # Legend frame
plt.rcParams['legend.framealpha'] = 0.9     # Legend frame transparency
plt.rcParams['legend.fancybox'] = True      # Rounded corners for legend
plt.rcParams['legend.numpoints'] = 1        # Number of points in legend for line plots

# LaTeX settings (optional - for high-quality mathematical expressions)
plt.rcParams['text.usetex'] = False         # Set to True if you have LaTeX installed
plt.rcParams['mathtext.default'] = 'regular'  # Math font style

# Color settings - you can define a custom color palette
thesis_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=thesis_colors)



# #directories for evaluations without noise (TO BE RUN)
# evaluation_directories = [
#     "project/data/evaluation_20250811_215207/",
#     "project/data/evaluation_20250811_215416/",
#     "project/data/evaluation_20250811_215840/",
#     "project/data/evaluation_20250811_220400/",
#     "project/data/evaluation_20250811_221134/",
#     "project/data/evaluation_20250811_222149/",
#     "project/data/evaluation_20250811_223310/",
#     "project/data/evaluation_20250811_224624/",
#     "project/data/evaluation_20250811_230053/",
#     "project/data/evaluation_20250811_231712/",
# ]

#Test with large FOV (10x10 um)
evaluation_directories = [
    "project/data/evaluation_20250822_222411/"
]

evaluation_directories = [
    "project/data/evaluation_density_1_20250823_171354/",
    # "project/data/evaluation_density_2_20250823_171354/",
    # "project/data/evaluation_density_3_20250823_171354/",
    # "project/data/evaluation_density_4_20250823_171354/",
    # "project/data/evaluation_density_5_20250823_171354/",
]
metadata = {
    "afterpulsing": 0.0014,
    "area_size": [4, 4],
    "crosstalk": 0.001,
    "dark_count_rate": 100,
    "dead_time": 50,
    "jitter": 0.12,
    "detection_efficiency": 1.0,
    "laser_power": 10000.0,
    "magnification": 150,
    "dwell_time": 2,
    "positions": [80, 80],
    "pixel_size": 0.050,
    "enable_noise": True,
}

    # Run comparison with multiple run patterns
comparison_results = compare_evaluations_multi_run(
    evaluation_directories, 
    run_patterns=['run_000*.h5', 'run_001*.h5', 'run_002*.h5'],
    n_repetitions=1,
    save_path_prefix="multi_run_evaluation_comparison",
    max_distance=0.020,  #micrometer., or 20 nm. Should be close to expected resolution. This is the maximum distance to consider a localization as true positive (TP) or false positive (FP
    metadata_given=metadata,
    plot=True,
    verbose=True
)

# Access results
print(f"\nMean biases: {comparison_results['mean_biases']}")
print(f"Bias errors: {comparison_results['bias_errors']}")
print(f"Mean precisions: {comparison_results['mean_precisions']}")
print(f"Precision errors: {comparison_results['precision_errors']}")

# Store comparison results to a .json file
import json
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f'comparison_results_{timestamp}.json', 'w') as f:
    json.dump(comparison_results, f, indent=4)