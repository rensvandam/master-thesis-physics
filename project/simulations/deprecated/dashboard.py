import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import io
import base64
import time

import project.model.plot_functions as plotting
import project.model.coherence_from_data as coherence
import project.simulations.deprecated.emitter_map as emitter_map

from project.model.sample import Alexa647
from project.model.detection import Sensor, Spad512, Spad23, merge_photons, show_photons
from project.model.emitter_density_map import get_map
from project.model.coherence_analytical import expected_number_of_emitters

# Set page config
st.set_page_config(
    page_title="SPAD Sensor Simulation",
    layout="wide"
)

def visualize_data():
    # Create columns for layout
    col1, col2 = st.columns(2)
    # Use a selectbox for pixel selection
    with col1:
        # General simulation information
        st.subheader("Simulation Information")
        st.write(f"Number of emitters: {nr_emitters}")
        st.write(f"Number of detected photons: {st.session_state.measured_photons}")


        st.subheader("Select Pixel")
        pixel_ids = sorted(range(len(ssr.pixel_coordinates)))
        selected_pixel = st.selectbox("Choose a pixel", [i for i in pixel_ids], index=pixel_ids.index(10) if 10 in pixel_ids else 0)
        selected_pixel = selected_pixel
        st.session_state[selected_pixel] = selected_pixel

        highlight_neighbors = st.checkbox("Highlight Nearest Neighbors", value=False)

        # Show the sensor layout
        st.subheader("Sensor Layout")
        fig, ax = plt.subplots(figsize=(8, 8))
        ssr.show(ax=ax)
        plotting.show_pixel_numbers(ssr, ax=ax)

        st.pyplot(fig)

    # Display correlation visualization
    with col2:
        st.subheader(f"Correlation with Pixel {selected_pixel}")
        
        selected_pixel_correlations = coherence_matrix[selected_pixel]
        # Plot sensor with correlation overlay
        fig1, ax = plt.subplots(figsize=(8, 8))
        ssr.show(data_to_show=selected_pixel_correlations, title="Sensor with Correlation Overlay", ax=ax)

        # Highlight neighbors if checkbox is checked
        if highlight_neighbors:
                # For hexagonal SPAD23 sensor
            if hasattr(ssr, '_Spad23__neighbors'):
                neighbors_dict = getattr(ssr, '_Spad23__neighbors')
            neighbors = neighbors_dict[selected_pixel]
            # Highlight the selected pixel
            pixel_x, pixel_y = ssr.pixel_coordinates[selected_pixel]
            circle = plt.Circle((pixel_x, pixel_y), radius=0.4, fill=False, edgecolor='green', linewidth=2, linestyle='-')
            ax.add_patch(circle)
            
            # Highlight the neighbors
            for neighbor in neighbors:
                n_x, n_y = ssr.pixel_coordinates[neighbor]
                neighbor_circle = plt.Circle((n_x, n_y), radius=0.4, fill=False, edgecolor='red', linewidth=3, linestyle='--')
                ax.add_patch(neighbor_circle)
                
                # Draw line connecting selected pixel to neighbor
                ax.plot([pixel_x, n_x], [pixel_y, n_y], 'r--', linewidth=3, alpha=0.7)
            
            # Add a legend for the neighbors
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markeredgecolor='black', markersize=10, label='Selected Pixel'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markeredgecolor='green', markersize=10, label='Neighbors')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

        st.pyplot(fig1)

        st.dataframe(pd.DataFrame(selected_pixel_correlations, columns=["Correlation"], index=[f"Pixel {i}" for i in range(len(selected_pixel_correlations))]))
        
        # Plot the correlation matrix 
        st.subheader("Correlation Matrix")
        fig2,ax = plt.subplots(figsize=(8, 8))
        #st.session_state.coherence_matrix = st.session_state.coherence_matrix/np.max(st.session_state.coherence_matrix)
        cax = ax.matshow(st.session_state.coherence_matrix, cmap='coolwarm', vmin=-1, vmax=np.max(st.session_state.coherence_matrix))
        # Add colorbar
        fig2.colorbar(cax)
        ax.set_xticks(range(23))
        ax.set_yticks(range(23))
        ax.set_xticklabels(range(0, 23))
        ax.set_yticklabels(range(0, 23))
        # Create a mask to highlight the selected pixel's row and column
        if selected_pixel is not None:
            highlight_mask = np.zeros_like(st.session_state.coherence_matrix, dtype=bool)
            highlight_mask[selected_pixel, :] = True
            highlight_mask[:, selected_pixel] = True
            
            # Highlight selected pixel's row and column
            for i in range(st.session_state.coherence_matrix.shape[0]):
                for j in range(st.session_state.coherence_matrix.shape[1]):
                    if highlight_mask[i, j]:
                        rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='black', lw=2)
                        ax.add_patch(rect)

                        # Highlight neighbors in the correlation matrix if checkbox is checked
            if highlight_neighbors:
                # For hexagonal SPAD23 sensor
                if hasattr(ssr, '_Spad23__neighbors'):
                    neighbors_dict = getattr(ssr, '_Spad23__neighbors')
                neighbors = neighbors_dict[selected_pixel]
                for neighbor in neighbors:
                    # Highlight the neighbor's intersection with the selected pixel
                    rect1 = plt.Rectangle((neighbor-0.5, selected_pixel-0.5), 1, 1, fill=False, edgecolor='green', lw=2)
                    rect2 = plt.Rectangle((selected_pixel-0.5, neighbor-0.5), 1, 1, fill=False, edgecolor='green', lw=2)
                    ax.add_patch(rect1)
                    ax.add_patch(rect2)
            st.pyplot(fig2)

        # Plot the coherence to distance plot
        st.subheader("Coherence to Distance Plot")
        fig3, ax = plt.subplots(figsize=(8, 5))
        plotting.show_coherence_to_distance(ssr, lag, coherence_matrix, ax=ax)
        st.pyplot(fig3)

with st.sidebar:
    # Simulation parameters
    st.title("Simulation Parameters")

    # Select sensor type
    sensor_type = st.selectbox("Select Sensor Type", ["SPAD23", "SPAD512"])

    # Number of emitters
    nr_emitters = st.slider("Number of Emitters", min_value=1, max_value=10, value=1)

    # Laser power
    laser_power = st.slider("Laser Power [kW/cm²]", min_value=1, max_value=500, value=330)
    laser_power = laser_power * 10**3  # Convert to W/cm²

    # Measurement Time interval
    time_interval = st.slider("Measuring Time Interval [ns]", min_value=10**3, max_value=10**6, value=10**5)

    # Detection efficiency
    detection_efficiency = st.slider("Detection Efficiency", min_value=0.1, max_value=1.0, value=1.0, step=0.1)

    # # Duration of measurement
    # duration = st.slider("Duration of Measurement [ns]", min_value=10**3, max_value=10**6, value=10**4)

    # Seed for random number generator
    initial_seed = 65
    seed = st.slider("Set seed", min_value=0, max_value=100, value=initial_seed, key="seed")

    # Bin size for coherence calculation
    bin_size = st.slider("Bin Size", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

    # Coherence lag
    lag = st.slider("Coherence lag", min_value=0, max_value=10000, value=0, step=10)

# Main app
st.title("SPAD Sensor Correlation Dashboard")
st.markdown("""This dashboard allows you to explore SPAD sensor data.""")

# Initialize session state variables
if "nr_sims" not in st.session_state:
    st.session_state.nr_sims = 1

if "simulation_run" not in st.session_state:
    st.session_state.simulation_run = False

if "seed" not in st.session_state:
    st.session_state.seed = initial_seed
    
if "ssr" not in st.session_state:
    st.session_state.ssr = None
    
if "coherence_matrix" not in st.session_state:
    st.session_state.coherence_matrix = None

# Add number of simulations input before the button
nr_sims = st.sidebar.number_input("Number of simulations", min_value=1, max_value=10000, value=st.session_state.nr_sims)
st.session_state.nr_sims = nr_sims

# # Button to run a single simulation
# if st.button("Run Simulation"):
#     if sensor_type == "SPAD512":
#         st.session_state.ssr, st.session_state.emitters, st.session_state.photons, st.session_state.measurement, st.session_state.is_detected, st.session_state.estimated_emitters_array, st.session_state.coherence_matrix = \
#             emitter_map.SPAD512_simulated_measurement(nr_emitters, laser_power, time_interval, bin_size, detection_efficiency, seed)
#         st.session_state.simulation_run = True
#     elif sensor_type == "SPAD23":
#         st.info(f"Running simulation with parameters:"
#         f"\nNumber of emitters: {nr_emitters}"
#         f"\nLaser power: {laser_power} W/cm²"
#         f"\nTime interval: {time_interval} ns"
#         f"\nDetection efficiency: {detection_efficiency}"
#         f"\nSeed: {seed}"
#         f"\nBin size: {bin_size} ns"
#         f"\nCoherence lag: {lag}")

#         st.session_state.ssr, st.session_state.emitters, st.session_state.photons, st.session_state.measurement, st.session_state.is_detected, st.session_state.estimated_emitters_array, st.session_state.coherence_matrix = \
#             emitter_map.SPAD23_simulated_measurement(nr_emitters, laser_power, time_interval, bin_size, detection_efficiency, seed, lag=lag, dashboard=True, debug=True)
#         st.session_state.simulation_run = True

if st.button("Reset simulation"):
    st.session_state.simulation_run = False
# Button to run many simulations
# Inside your "Run simulation(s)" button handler:
if st.button("Run simulation(s)"):
    if sensor_type == "SPAD23":
        progress_bar = st.progress(0)
        coherence_matrix_avg = np.zeros((23, 23))
        measured_photons_avg = 0
        
        # Create a container to show debug information
        debug_container = st.container()
        with debug_container:
            st.subheader("Simulation Debug Information")
            debug_info = st.empty()
        
        # Set initial seed from user input
        current_seed = st.session_state.seed
        
        # Track simulation specific information
        all_simulation_info = []
        
        for i in range(nr_sims):
            # Reset the random number generator
            np.random.seed(current_seed)
            
            status_text = st.empty()
            status_text.write(f"Running simulation {i+1} of {nr_sims} with seed {current_seed}")
            

            
            # Run your simulation with the current seed
            temp_ssr, temp_emitters, temp_photons, temp_measurement, temp_detected, temp_estimated, temp_coherence = \
                emitter_map.SPAD23_simulated_measurement(
                    nr_emitters, 
                    laser_power,
                    time_interval, 
                    bin_size, 
                    detection_efficiency, 
                    current_seed, 
                    lag=lag, 
                    dashboard=True,
                    debug=True
                )
            
            coherence_matrix_avg += temp_coherence
            if not i == max(range(nr_sims)):
                temp_ssr.clear()
            progress_bar.progress((i+1)/nr_sims)
            
            # Generate next seed
            current_seed = np.random.randint(0, 10000)
        
        coherence_matrix_avg /= nr_sims
        st.session_state.coherence_matrix = coherence_matrix_avg
        st.session_state.simulation_run = True
        
        # Store the last simulation's other values in session state
        st.session_state.ssr = temp_ssr  # Save the last simulation's SSR for visualization

        measured_photons_avg += int(len(temp_measurement))
        measured_photons_avg /= nr_sims
        st.session_state.measured_photons = measured_photons_avg 

        st.success(f"Completed {nr_sims} simulations!")

# Always call visualize_data if simulation has been run
if st.session_state.simulation_run:
    # Access simulation results from session state here
    ssr = st.session_state.ssr
    st.session_state.coherence_matrix = coherence_matrix_avg
    coherence_matrix = st.session_state.coherence_matrix
    visualize_data() 