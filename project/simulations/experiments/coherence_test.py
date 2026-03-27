import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def sparse_convolution(indices1, indices2, nr_steps, offset):
    """
    Perform sparse convolution between two sets of indices.
    
    Parameters:
    -----------
    indices1 : array-like
        First set of indices
    indices2 : array-like
        Second set of indices
    nr_steps : int
        Number of steps to calculate
    offset : int
        Starting offset
    
    Returns:
    --------
    numpy array of photon pairs at each step
    """
    photon_pairs = np.zeros(nr_steps, dtype=int)
    
    for i in indices1:
        for j in indices2:
            diff = i - j
            if offset <= diff < offset + nr_steps:
                photon_pairs[diff - offset] += 1
    
    return photon_pairs

def coherence(signal1, signal2, interval, bin_size, nr_steps, 
              offset=0, normalize=True, auto_correlation=False):
    """
    Calculates the second-order quantum coherence of the measured photons.
    
    Parameters:
    -----------
    signal1, signal2 : array-like
        Signals consisting of hit times
    interval : float
        Total time interval of measurement
    bin_size : float
        Size of time bins
    nr_steps : int
        Number of steps for correlation calculation
    offset : int, optional
        Starting step offset (default 0)
    normalize : bool, optional
        Whether to normalize the correlation (default True)
    auto_correlation : bool, optional
        Whether this is an auto-correlation calculation (default False)
    
    Returns:
    --------
    correlation : numpy array
        Correlation values
    bins : numpy array
        Lag times corresponding to correlation values
    """
    # Calculate bin indices
    photon_indices1 = np.floor(signal1 / bin_size).astype(np.int64)
    photon_indices2 = np.floor(signal2 / bin_size).astype(np.int64)
    
    # Compute photon pairs
    photon_pairs = sparse_convolution(photon_indices1, photon_indices2, nr_steps, offset)
    
    # Adjust for auto-correlation if needed
    if auto_correlation and offset == 0:
        photon_pairs[0] -= len(signal1)
    
    # Compute number of bins in interval
    m = int(interval / bin_size)
    
    # Compute bias correction
    bias = np.arange(m - offset, m - offset - nr_steps, -1)
    
    # Correct correlation for bias
    correlation = photon_pairs * (m / bias)
    
    # Normalize if requested
    if normalize and len(signal1) > 0 and len(signal2) > 0:
        correlation = correlation * m / (len(signal1) * len(signal2))
    
    # Compute lag times
    bins = np.arange(offset, nr_steps + offset) * bin_size
    
    return correlation, bins

def generate_poisson_signal(interval, rate):
    """
    Generate a Poisson process signal.
    
    Parameters:
    -----------
    interval : float
        Total time interval
    rate : float
        Average number of events per unit time
    
    Returns:
    --------
    numpy array of event times
    """
    # Generate number of events from Poisson distribution
    num_events = np.random.poisson(rate * interval)
    
    # Generate uniformly distributed event times
    signal = np.sort(np.random.uniform(0, interval, num_events))
    
    return signal

def main():
    st.title('Photon Coherence Analysis')
    
    # Sidebar for input parameters
    st.sidebar.header('Coherence Parameters')
    
    # Measurement parameters
    interval = st.sidebar.number_input('Total Interval (s)', value=1.0, min_value=0.01)
    bin_size = st.sidebar.number_input('Bin Size (s)', value=0.01, min_value=0.001)
    nr_steps = st.sidebar.number_input('Number of Steps', value=20, min_value=1)
    offset = st.sidebar.number_input('Offset', value=0, min_value=0)
    
    # Signal generation parameters
    signal1_rate = st.sidebar.number_input('Signal 1 Rate (events/s)', value=10.0, min_value=0.1)
    signal2_rate = st.sidebar.number_input('Signal 2 Rate (events/s)', value=10.0, min_value=0.1)
    
    # Calculation options
    normalize = st.sidebar.checkbox('Normalize Correlation', value=True)
    auto_correlation = st.sidebar.checkbox('Auto-Correlation', value=False)
    
    # Generate signals
    signal1 = generate_poisson_signal(interval, signal1_rate)
    signal2 = generate_poisson_signal(interval, signal2_rate)
    
    # Compute coherence
    correlation, bins = coherence(
        signal1, signal2, 
        interval=interval, 
        bin_size=bin_size, 
        nr_steps=nr_steps,
        offset=offset, 
        normalize=normalize,
        auto_correlation=auto_correlation
    )
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bins, correlation, marker='o')
    ax.set_xlabel('Lag Time (s)')
    ax.set_ylabel('Coherence')
    ax.set_title('Second-Order Quantum Coherence')
    
    # Display plot
    st.pyplot(fig)
    
    # Additional information
    st.subheader('Signal Details')
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Signal 1 Events: {len(signal1)}")
    with col2:
        st.write(f"Signal 2 Events: {len(signal2)}")
    
    # Raw data display
    if st.checkbox('Show Raw Correlation Data'):
        data_df = pd.DataFrame({
            'Lag Time': bins,
            'Correlation': correlation
        })
        st.dataframe(data_df)

if __name__ == '__main__':
    main()