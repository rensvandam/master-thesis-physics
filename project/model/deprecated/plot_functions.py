import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def show_emitter_positions(positions, x_lim, y_lim, include_off_grid=False, ax=None):
    """
    Plots the positions of molecules as a scatterplot.
    """
    if ax:
        plt.sca(ax)
    if positions is None:
        return
    
    plt.scatter(positions[:, 0], positions[:, 1], color='cyan', marker='x', s=10)
    if include_off_grid:
        show_off_grid_emitters(positions, x_lim, y_lim)

def show_pixel_numbers(sensor, ax=None):
    """
    Show the pixel numbers on the sensor.
    """
    if ax:
        plt.sca(ax)

    # Add pixel numbers inside the circles
    for i, (x, y) in enumerate(sensor.pixel_coordinates):
        plt.text(x, y, str(i), color='black', fontsize=8, ha='center', va='center')


def show_off_grid_emitters(positions, x_lim, y_lim):
    """
    Molecules with a position outside the limits are indicated with an arrow at the respective boundary.
    """
    dx = 0.02 * x_lim
    dy = 0.02 * y_lim
    left = positions[:, 0] < 0
    plt.scatter(np.zeros(sum(left)) + dx, positions[left, 1], s=100, color='red', marker=r'$\leftarrow$')
    right = positions[:, 0] > x_lim
    plt.scatter(np.ones(sum(right)) * x_lim - dx, positions[right, 1], s=100, color='red', marker=r'$\rightarrow$')
    down = positions[:, 1] < 0
    plt.scatter(positions[down, 0] + dy, np.zeros(sum(down)), s=100, color='red', marker=r'$\downarrow$')
    up = positions[:, 1] > y_lim
    plt.scatter(positions[up, 0], np.ones(sum(up)) * y_lim - dy, s=100, color='red', marker=r'$\uparrow$')


def show_coherence(coherence_array, sensor, positions=None, title=None, save_as=None):
    """
    Show the quantum coherence on the sensor. Emitter positions are shown optionally.
    """
    sensor.show(coherence_array, vmin=0, vmax=1.0, title=title)
    show_emitter_positions(positions, sensor.x_edges[-1], sensor.y_edges[-1])
    if save_as:
        plt.savefig(fname=save_as)
    plt.show()


def show_emitter_density(emitter_density, sensor, max_nr_emitters=None, positions=None, title=None, save_as=None,
                         intensity_map=None, ax=None):
    """
    Show the emitter density on the sensor. If the intensity is included, the transparency is determined by the relative
    brightness. Emitter positions are shown optionally.
    """
    # Values smaller than 1 are invalid.
    copy_emitter_density = np.copy(emitter_density)
    copy_emitter_density[copy_emitter_density < 1.0] = 0

    # Determine the transparency
    alpha = np.ones(np.shape(copy_emitter_density)) if intensity_map is None else intensity_map / np.amax(intensity_map)
    alpha[copy_emitter_density < 0.5] = 1

    # Create the colormap
    cmap = plt.cm.jet
    cmap_list = [cmap(i) for i in range(cmap.N)]
    cmap_list[0] = (0, 0, 0, 1)  # First entry is black.
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmap_list, cmap.N)
    max_cmap = max_nr_emitters or np.amax(np.round(copy_emitter_density)).astype(int)
    bounds = np.linspace(-0.5, max_cmap + 0.5, max_cmap + 2)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Plot
    if ax:
        plt.sca(ax)
    x_coords = np.linspace(sensor.x_limits[0], sensor.x_limits[-1], copy_emitter_density.shape[0] + 1)
    y_coords = np.linspace(sensor.y_limits[0], sensor.y_limits[-1], copy_emitter_density.shape[0] + 1)
    x, y = np.meshgrid(x_coords, y_coords)
    img = plt.pcolormesh(x, y, copy_emitter_density.T, cmap=cmap, norm=norm, alpha=alpha.T)
    plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds[1:] - 0.5)
    #show_emitter_positions(positions, sensor.x_limits[-1], sensor.y_limits[-1])
    plt.xlabel("x [nm]")
    plt.ylabel("y [nm]")
    plt.title(title)
    plt.tight_layout()
    if save_as:
        plt.savefig(fname=save_as)
    #plt.show()

def show_coherence_to_distance(s, lag, coherence_matrix, ax=None):
    """
    Plots the coherence between pixels as a function of distance between pixels in a line plot.
    
    Parameters
    ----------
    s : Sensor
        Sensor object.
    coherence_matrix : np.ndarray
        Coherence matrix.
    dashboard : bool
        If True, returns a streamlit figure. Otherwise a matplotlib figure.
    
    Returns
    -------
    None
    """
    if ax:
        plt.sca(ax)
    # Calculating distance between pixels
    distances = np.zeros((s.nr_pixels, s.nr_pixels))
    for i in range(s.nr_pixels):
        for j in range(s.nr_pixels):
            distances[i, j] = np.sqrt((s.pixel_coordinates[i, 0] - s.pixel_coordinates[j, 0])**2 + (s.pixel_coordinates[i, 1] - s.pixel_coordinates[j, 1])**2)
    # Flatten the matrices to 1D arrays
    distances = distances.flatten()
    correlations = coherence_matrix.flatten()

    # Define distance bins
    unique_bins = np.unique(distances)
    unique_bins = unique_bins[np.insert(np.diff(unique_bins) > 1e-2, 0, True)]
    bin_indices = np.digitize(distances, unique_bins)  # Assign distances to bins

    # Compute mean correlation per distance bin
    bin_means = [correlations[bin_indices == i].mean() for i in range(1, len(unique_bins))]

    # Plot the results
    plt.plot(unique_bins[:-1], bin_means, marker='o', linestyle='-')
    plt.xlabel('Distance Between Pixels [µm]')
    plt.ylabel(f"g2({lag}) coherence correlation")
    plt.title('Spatial Correlation of Coherence')
    plt.grid()
    plt.show()