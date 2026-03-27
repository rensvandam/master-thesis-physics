import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import io
import base64
import time

import project.model.plot_functions as plotting
import project.model.coherence_from_data as coherence
import project.simulations.deprecated.emitter_map as emitter_map
import project.simulations.test_coherence_calc as test_coherence

from project.model.sample import Alexa647
from project.model.detection import Sensor, Spad512, Spad23, merge_photons, show_photons
from project.model.emitter_density_map import get_map
from project.model.coherence_analytical import expected_number_of_emitters

import project.tool.dash_plot_functions as dash_plot

# Set page config
st.set_page_config(
    page_title="SPAD Sensor Simulation",
    layout="wide"
)


def compare_coherence_calculations():
    st.write("Comparing coherence calculations")
    st.sidebar.write("Choose parameters for the simulation")
    with st.sidebar:
        # Simulation parameters
        st.title("Simulation Parameters")

        # Number of emitters
        nr_emitters = st.slider("Number of Emitters", min_value=1, max_value=10, value=1)

        # Laser power
        laser_power = st.slider("Laser Power [kW/cm²]", min_value=1, max_value=500, value=330)
        laser_power = laser_power * 10**3  # Convert to W/cm²

        # Measurement Time interval
        time_interval = st.slider("Measuring Time Interval [ns]", min_value=10**3, max_value=10**6, value=10**5)

        # Detection efficiency
        detection_efficiency = st.slider("Detection Efficiency", min_value=0.1, max_value=1.0, value=1.0, step=0.1)

        # Select detection inefficiencies
        st.title("Detection inefficiencies")
        INEFFICIENCIES = {
            "Dark counts": "measure_enable_dark_counts",
            "Timestamp jitter": "measure_enable_timestamp_jitter",
            "Dead time": "measure_enable_deadtime",
            "Cross Talk": "measure_enable_crosstalk",
            "After Pulsing": "measure_enable_afterpulsing"
        }
        NONE_OPTION = "None"
        ALL_OPTION = "All"
        detection_inefficiencies = st.multiselect(
            "Select detection inefficiencies",
            [NONE_OPTION] + list(INEFFICIENCIES.keys()) + [ALL_OPTION],
            default=[NONE_OPTION]
        )
        enabled_detection_inefficiencies_flags = dash_plot.get_enabled_flags_for_detection_inefficiencies(INEFFICIENCIES, detection_inefficiencies)

        # Set dead time if dead time is in the enabled detection inefficiencies
        if "measure_enable_deadtime" in enabled_detection_inefficiencies_flags and enabled_detection_inefficiencies_flags["measure_enable_deadtime"]:
            dead_time = st.number_input(
                "Dead Time [ns]", min_value=0.0, max_value=100.0, value=50.0, step=0.001)
            enabled_detection_inefficiencies_flags["spad_dead_time"] = dead_time

        # Seed for random number generator
        initial_seed = 65
        seed = st.slider("Set seed", min_value=0, max_value=100, value=initial_seed, key="seed")

        # Bin size for coherence calculation
        bin_size = st.slider("Bin Size", min_value=0.1, max_value=5.0, value=0.1, step=0.1)

        # Coherence nr steps
        nr_steps = st.slider("Coherence number of steps", min_value=0, max_value=10000, value=200, step=10)

        # Fit method
        fit_method = st.selectbox("Fit Method", ["with_k", "without_k"])

        # Visualise emitters
        if "visualise_emitters" not in st.session_state:
            st.session_state.visualise_emitters = False
        visualise_emitters = st.checkbox("Visualise Emitters", value=False, key="visualise_emitters")

        # Visualise photons
        if "visualise_photons" not in st.session_state:
            st.session_state.visualise_photons = False
        visualise_photons = st.checkbox("Visualise Photons", value=False, key="visualise_photons")

    if st.button("Run Simulation"):
        # Run the simulation
        pixel_coherence, bins1, s, emitters, photons, measurement, is_detected, estimated_emitters_array =\
            test_coherence.sim_whole_detector(nr_emitters, laser_power, time_interval, bin_size, detection_efficiency, seed, nr_steps=nr_steps, dashboard=True, debug=False, **enabled_detection_inefficiencies_flags)
        pixel_autocoherence_array, bins2, s, emitters, photons, measurement, is_detected, estimated_emitters_array =\
            test_coherence.sim_pixels_autocoherence(nr_emitters, laser_power, time_interval, bin_size, detection_efficiency, seed, nr_steps=nr_steps, dashboard=True, debug=False, **enabled_detection_inefficiencies_flags)
        pixel_neighborhood_coherence_array, bins3, s, emitters, photons, measurement, is_detected, estimated_emitters_array =\
            test_coherence.sim_neighbourhood_autocoherence(nr_emitters, laser_power, time_interval, bin_size, detection_efficiency, seed, nr_steps=nr_steps, dashboard=True, debug=False, **enabled_detection_inefficiencies_flags)
        #pixel_neighborhood_coherence_weighed_array, bins4, s, emitters, photons, measurement, is_detected, estimated_emitters_array =\
        #    test_coherence.sim_neighbourhood_autocoherence_weighed(nr_emitters, laser_power, time_interval, bin_size, detection_efficiency, seed, nr_steps=nr_steps, dashboard=True, debug=False, **enabled_detection_inefficiencies_flags)
        pixel_neighborhood_coherence_pixelautocoherence_substracted, bins4, s, emitters, photons, measurement, is_detected, estimated_emitters_array =\
            test_coherence.sim_neighborhood_coherence_pixelautocoherence_substracted(nr_emitters, laser_power, time_interval, bin_size, detection_efficiency, seed, nr_steps=nr_steps, dashboard=True, debug=False, **enabled_detection_inefficiencies_flags)

        # Plotting
        col1,col2,col3,col4 = st.columns([1,1,1,1])
        with col1:
            st.info("**Coherence for all pixels combined**")
            with st.expander("Show"):
                dash_plot.show_coherence(bins1, pixel_coherence, title = "Coherence for all pixels combined", show_fit=True, show_sigma=True, fit_method = fit_method)
        with col2:
            st.info("**Autocoherence for each pixel**")
            with st.expander("Show"):
                for i in range(len(pixel_autocoherence_array)):
                    dash_plot.show_coherence(bins2, pixel_autocoherence_array[i], title = f"Coherence for pixel {i}", show_fit=True, fit_method = fit_method)
                #dash_plot.show_coherence(bins, pixel_autocoherence, title = "Coherence for each pixel", show_fit=True)
        with col3:
            st.info("**Coherence for each pixel with its neighbours**")
            with st.expander("Show"):
                for i in range(len(pixel_neighborhood_coherence_array)):
                    dash_plot.show_coherence(bins3, pixel_neighborhood_coherence_array[i], title = f"Coherence for pixel {i} and its neighbors", show_fit=True, fit_method = fit_method)
        with col4:
            st.info("**Coherence for each pixel with its pixel autocoherence subtracted from sensor-wide autocoherence**")
            with st.expander("Show"):
                dash_plot.show_coherence(bins4, pixel_neighborhood_coherence_pixelautocoherence_substracted, title = "Coherence with pixel autocoherence subtracted from sensor-wide autocoherence", show_fit=True, fit_method = fit_method)

        if not st.session_state.visualise_photons or not st.session_state.visualise_emitters:
            s.clear()
        # Visualise emitters
        if st.session_state.visualise_emitters:
            # Show the sensor layout
            st.subheader("Sensor Layout")
            fig, ax = plt.subplots(figsize=(8, 8))
            s.show(ax=ax)
            plotting.show_pixel_numbers(s, ax=ax)
            if visualise_photons:
                show_photons(photons, is_detected, ax=ax)
            st.pyplot(fig)


    if st.button("Reset Simulation"):
        st.rerun()
def main():
    compare_coherence_calculations()
    
if __name__ == "__main__":
    main()