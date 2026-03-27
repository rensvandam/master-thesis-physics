import numpy as np
import matplotlib.pyplot as plt
import project.model.emitter_density_map as ne
import project.model.coherence_from_data as coherence
from project.model.sample import Alexa647, Sensor, Emitter, expected_excitation_time
from project.model.helper_functions import mean_squared_error
import project.model.plot_functions as plotting


def pixels_per_emitter_analysis(nr_pixels, nr_emitters, intensity_threshold):
    s = Sensor(nr_pixels, pixel_size=665 / 4)
    laser = 300 * 10 ** 3  # W / cm2
    interval = 10 ** 5  # ns
    eta = 1

    N = 10
    pixels_per_emitter = np.zeros((4, len(intensity_threshold), N))
    for j in range(N):
        print(f"rep {j}")
        positions = np.random.default_rng(seed=j + 51684).uniform(0, nr_pixels * s.pixel_size, size=(nr_emitters, 2))
        sigmas = []
        emitters = []
        for k in range(nr_emitters):
            e = Alexa647(x=positions[k, 0], y=positions[k, 1])
            e.generate_photons(laser_power=laser, time_interval=interval, seed=j * 54 + k, detection_efficiency=eta)
            emitters.append(e)
            sigmas.append(e.sigma)
        s.measure(emitters)

        for i in range(4):
            if i < 3:
                coherence_array = coherence.denoised_auto_coherence(s, interval, 0.01, i)
            else:
                coherence_array = coherence.auto_coherence_per_neighbourhood(s, interval, 0.01)

            for l in range(len(intensity_threshold)):
                emitter_density = ne.get_emitter_density(coherence_array, round_to_int=False, intensity=s.histogram,
                                                         intensity_threshold=intensity_threshold[l],
                                                         outliers_threshold=5)
                pixels_per_emitter[i, l, j] = np.sum(emitter_density) / nr_emitters
        s.clear_sensor()
    return pixels_per_emitter


def calculate_pixels_per_emitter(sigma, pixel_size, cut_off):
    return 2 * np.pi * np.log(1 / cut_off) * (sigma / pixel_size) ** 2


def plot_pixels_per_emitter():
    nr_pixels = 15
    nr_emitters = 15
    c = np.linspace(0.01, 0.99, 99)
    pixels_per_emitter = pixels_per_emitter_analysis(nr_pixels, nr_emitters, c)
    # np.save(f'../data/meeting_24_01_18/pixels_per_emitter_{nr_pixels}pixels_{nr_emitters}', pixels_per_emitter)

    # pixels_per_emitter = np.load('../data/meeting_24_01_18/pixels_per_emitter_15pixels_16_emitters.npy')
    analytical = calculate_pixels_per_emitter(665 / 4, 665 / 4, c)
    labels = ['Per pixel', 'Average 4 NN', 'Average 8 NN', '3x3 kernel']
    for i in range(4):
        av = np.mean(pixels_per_emitter[i, :, :], axis=1)
        std = np.std(pixels_per_emitter[i, :, :], axis=1)
        plt.plot(c, av, label=labels[i])
        plt.fill_between(c, av - std, av + std, alpha=0.2)
    plt.plot(c, analytical, label="Analytical", color='black')
    plt.legend()
    plt.xlabel("Intensity threshold")
    plt.ylabel("Number of pixels per emitter")
    plt.show()


def mse(nr_emitters, nr_pixels, intensity_threshold):
    s = Sensor(nr_pixels, pixel_size=665 / 4)
    laser = 300 * 10 ** 3  # W / cm2
    interval = 10 ** 5  # ns
    eta = 1
    f = f'../data/meeting_24_01_18/binsize_0.01_laser_{laser / 10 ** 3}kwcm2_interval' \
        f'{interval / 10 ** 6}ms_eta_{eta}_emitters_{nr_emitters}_pixels_{nr_pixels}_cutoff_0.1_1000reps_'

    N = 10
    mse = np.zeros((4, 4, N))
    for k in range(N):
        print(f"seed {k}")
        positions = np.random.default_rng(seed=k + 500).uniform(0, nr_pixels * s.pixel_size, size=(nr_emitters, 2))
        sigmas = []
        emitters = []
        for j in range(nr_emitters):
            e = Alexa647(x=positions[j, 0], y=positions[j, 1])
            e.generate_photons(laser_power=laser, time_interval=interval, seed=k * 54 + j, detection_efficiency=eta)
            emitters.append(e)
            sigmas.append(e.sigma)
        s.measure(emitters)
        s.show()
        plotting.show_emitter_positions(positions, s.x_edges[-1], s.y_edges[-1])
        plt.show()

        ground_truth = s.ground_truth(positions, sigmas, cutoff_fraction=0.1)
        ground_truth_zeros_corrected = ne.zeros_correction(ground_truth, s.histogram, intensity_threshold)
        for i in range(4):
            if i < 3:
                coherence_array = coherence.denoised_auto_coherence(s, interval, 0.01, i)
            else:
                coherence_array = coherence.auto_coherence_per_neighbourhood(s, interval, 0.01)
            plotting.show_emitter_density(
                ne.get_emitter_density(coherence_array, round_to_int=False, outliers_threshold=-1), sensor=s,
                max_nr_emitters=5)
            plt.show()
            plotting.show_emitter_density(
                ne.get_emitter_density(coherence_array, round_to_int=False, outliers_threshold=5, intensity=s.histogram,
                                       intensity_threshold=0.2), s, max_nr_emitters=5)
            plt.show()
            mse[0, i, k] = mean_squared_error(
                ne.get_emitter_density(coherence_array, round_to_int=False, outliers_threshold=5),
                ground_truth
            )
            mse[1, i, k] = mean_squared_error(
                ne.get_emitter_density(coherence_array, round_to_int=True, outliers_threshold=5),
                ground_truth
            )
            mse[2, i, k] = mean_squared_error(
                ne.get_emitter_density(coherence_array, round_to_int=False, intensity=s.histogram,
                                       intensity_threshold=intensity_threshold, outliers_threshold=5),
                ground_truth_zeros_corrected
            )
            mse[3, i, k] = mean_squared_error(
                ne.get_emitter_density(coherence_array, round_to_int=True, intensity=s.histogram,
                                       intensity_threshold=intensity_threshold, outliers_threshold=5),
                ground_truth_zeros_corrected
            )
        s.clear_sensor()

    np.save(f + "mse_float.npy", mse[0, :, :])
    np.save(f + "mse_int.npy", mse[1, :, :])
    np.save(f + "mse_float_zeros_added.npy", mse[2, :, :])
    np.save(f + "mse_int_zeros_added.npy", mse[3, :, :])


def print_mse():
    nr_pixels = 16
    nr_emitters = 15
    laser = 300 * 10 ** 3  # W / cm2
    interval = 10 ** 5  # ns
    eta = 1
    f = f'../data/meeting_24_01_18/binsize_0.01_laser_{laser / 10 ** 3}kwcm2_interval' \
        f'{interval / 10 ** 6}ms_eta_{eta}_emitters_{nr_emitters}_pixels_{nr_pixels}_cutoff_0.1_1000reps_'

    categories = ['float', 'int']
    zeros = ['', '_zeros_added']

    print("Mean Squared Error\n")
    for category in categories:
        for zero in zeros:
            array = np.load(f + "mse_" + category + zero + ".npy")
            print(category)
            print(zero)
            print(f"Average:\t\t{np.round(np.mean(array, axis=1), 2)}")
            print(f"Standard dev:\t{np.round(np.std(array, axis=1), 2)}\n")


def estimate_density(eta, bin_sizes, nr_pixels, iterations, max_nr_emitters, filepath):
    # # Situation 1: E[T] = 2 ns = const
    # # Actually is alpha, but since alpha / laser_0 = 1 for the case lifetime = 1, alpha = laser_0
    # laser_0 = expected_excitation_time(240000, 650, 1)
    # lifetime_0 = 1
    #
    # ratio = np.array([0.1])
    # d_lifetime = (ratio - lifetime_0) / (1 + ratio)
    #
    # lifetime = lifetime_0 + d_lifetime  # W / cm2
    # laser = laser_0 / (1 - d_lifetime)

    # # Situation 2: k is constant: 2 ns-1
    # laser_0 = expected_excitation_time(240000, 650, 1)  # Equals 1/\alpha
    # alpha = 1 / laser_0
    # lifetime_0 = 1
    #
    # ratio = np.array([0.1])
    # d_laser = laser_0 * (ratio - lifetime_0) / (lifetime_0 * (ratio + 1))
    #
    # lifetime = lifetime_0 / (1 - alpha * d_laser * lifetime_0)
    # laser = laser_0 + d_laser

    laser = np.array([330 * 10**3], dtype=float)
    lifetime = np.array([1], dtype=float)
    ratio = np.array([1])

    excitation = expected_excitation_time(240000, 650, laser)
    expected_cycle_time = lifetime + excitation
    print(f"Lifetime: {lifetime},\n excitation time: {excitation},\n ratio: {lifetime/excitation}")
    print(f"Expected cycle time: {expected_cycle_time}")

    s = Sensor(nr_pixels, pixel_size=665 / 4)
    interval = 0.2 * 10 ** 5  # ns
    print(f"Interval: {interval}")
    position = nr_pixels * s.pixel_size / 2
    coh_array = np.zeros((len(ratio), max_nr_emitters + 1, iterations, len(bin_sizes)))
    # coh_array = np.load(filepath + "_coherence.npy")

    np.save(filepath + "_bin_size.npy", bin_sizes)
    for m in range(len(ratio)):
        print(f"Ratio: {m}/{len(ratio)}")
        for i in range(1, max_nr_emitters + 1):
            print(f"Emitter nr: {i}/{max_nr_emitters}")
            for j in range(iterations):
                print(f"Iteration: {j}/{iterations}")
                emitters = []
                seed = i * 7079 + j * (max_nr_emitters + 1)
                for k in range(i):
                    e = Emitter(position, position, absorption_wavelength=650, emission_wavelength=665,
                                lifetime=lifetime[m], extinction_coefficient=240000)
                    e.generate_photons(laser_power=laser[m], time_interval=interval, seed=seed + k, detection_efficiency=eta)
                    emitters.append(e)
                s.measure(emitters)

                for l, bin_size in enumerate(bin_sizes):
                    coh_array[m, i, j, l] = coherence.auto_coherence(s.data[:, s.T], interval, bin_size=bin_size, nr_steps=1)[0][0]
                s.clear_sensor()

            if i % 10 == 0:
                save_results(filepath, coh_array, lifetime, laser)
    save_results(filepath, coh_array, lifetime, laser)
    return coh_array, lifetime, laser


def run_emitter_density_few_emitters():
    pixel = 32
    eta = 1
    iterations = 100
    nr_emitters = 100
    # bin_sizes = np.linspace(0.05, 5, endpoint=True, num=100)
    bin_sizes = np.array([1])
    f = f'../data/report/bin1/{pixel}pixels_{iterations}iter_{nr_emitters}nr_emitters_{eta}eta_meastime0_2'
    estimate_density(eta, bin_sizes, pixel, iterations, nr_emitters, f)


def save_results(f, coherence_array, lifetime_array, laser_array):
    np.save(f + "_coherence.npy", coherence_array)
    np.save(f + "_lifetime.npy", lifetime_array)
    np.save(f + "_laser.npy", laser_array)


def plot_nr_emitters_estimate():
    it = 25
    thres = 0.1
    # nr_emitters = estimate_nr_of_emitters(thres, it)
    # np.save(f'../data/meeting_24_01_18/emitter_estimations_{it}iter_{thres}threshold.npy', nr_emitters)
    nr_emitters = np.load(f'../data/meeting_24_01_18/emitter_estimations_{it}iter_{thres}threshold.npy')
    labels = ['0 NN', '4 NN', '8 NN', '3x3 kernel', 'True']
    plt.plot(np.arange(0, 21), np.arange(0, 21), label=labels[4], linestyle='-', marker='', color='black')
    for i in range(4):
        plt.plot(nr_emitters[4, :], nr_emitters[i, :], label=labels[i], linestyle='', marker='.')
    plt.legend()
    plt.title('Number of emitters')
    plt.xlabel('Actual')
    plt.ylabel('Estimated')
    plt.xticks(np.arange(0, 22, step=2))
    plt.yticks(np.arange(0, 22, step=2))
    plt.show()


def run_emitter_density_estimation():
    pixel_list = [32]
    eta_list = [1]
    bin_sizes = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    iterations = 100
    max_density = 2
    for pixel in pixel_list:
        for eta in eta_list:
            nr_emitters, coh_array = estimate_density(eta, bin_sizes, pixel, iterations, max_density)
            f = f'../data/meeting_24_02_15/nr_emitters_{pixel}pixels_{iterations}iter_2max_density' \
                f'_{eta}eta_autocoherencesensor'
            np.save(f + ".npy", nr_emitters)
            np.save(f + "_coherence.npy", coh_array)


def plot_emitter_densities():
    pixel_list = [32]
    eta_list = [1]
    bin_sizes = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    iterations = 10
    max_density = 2
    pixel_size = 665 / 4
    for pixel in pixel_list:
        for eta in eta_list:
            f = f'/meeting_24_02_15/nr_emitters_{pixel}pixels_{iterations}iter_{max_density}max_density' \
                f'_{eta}eta_autocoherencesensor_all_bins'
            measured = np.load("../data" + f + ".npy")
            nr_emitters = np.shape(measured)[0]
            true_density = np.arange(nr_emitters) * 10 ** 6 / ((pixel * pixel_size) ** 2)
            all_densities = measured * 10 ** 6 / ((pixel * pixel_size) ** 2)

            plt.plot([0, max_density], [0, max_density], label='True density', linestyle='-', marker='',
                     color='black')
            for i, binsize in enumerate(bin_sizes):
                if binsize in (0.01, 0.5, 1):
                    density = all_densities[:, :, i]
                    plt.errorbar(true_density, np.mean(density, axis=1), yerr=np.std(density, axis=1),
                                label=f'bin size {binsize}', linestyle='', marker='', capsize=2)
            plt.legend()
            plt.title(f'Emitter density ({pixel} pixels, ' + r'$\eta$ = ' + f'{eta})')
            plt.xlabel(r'Density [$\mu m^{-2}$]')
            plt.ylabel(r'Estimated density [$\mu m^{-2}$]')
            # plt.ylim([0, 2.5])
            # plt.savefig("../figures" + f + ".png")
            plt.show()

            # true_density = 1 - 1 / np.arange(1, nr_emitters)
            for i, binsize in enumerate(bin_sizes):
                if binsize in (0.01, 0.5, 1):
                    density = all_densities[:, :, i]
                    # density = 1 - 1 / measured[1:, :, i]
                    error = 100 * (np.mean(density, axis=1) - true_density) / true_density
                    print(f"Bin size: {binsize}, average error: {np.mean(error[1:])}")
                    plt.plot(true_density, error, linestyle='-', marker='.', label=f'bin size {binsize}')

            # g = 1 - 1 / np.arange(1, nr_emitters)
            # est = 1 - 0.667/ np.arange(1, nr_emitters)
            # plt.plot(g, 100*(est-g)/g, marker='x')
            plt.axhline(y=0, color='grey', linestyle='--')
            plt.legend()
            plt.title(f'Relative error ({pixel} pixels, ' + r'$\eta$ = ' + f'{eta})')
            plt.xlabel(r'Density [$\mu m^{-2}$]')
            plt.ylabel(r'Relative error [%]')
            plt.ylim([-100, 100])
            # plt.savefig("../figures" + f + "_error.png")
            plt.show()


def plot_mean_error_binsize():
    # Nr pixels: 32, eta: 1, it: 10, density: 2
    plt.plot(
        [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        [0.7, 0.3, 2.3, 5.2, 11.7, 18.4, 25.4, 32.4, 39.5, 47.1, 54.5, 62.0, 69.9],
        marker='.',
        label='Average error'
    )
    plt.plot([0, 1], [0, 69.9], label='y = 69.9x')
    plt.xlabel('Bin size [ns]')
    plt.ylabel('Relative error [%]')
    plt.title('Relative error vs bin size')
    plt.legend()
    plt.show()


def main():
    run_emitter_density_few_emitters()
    # plt.rcParams.update({'font.size': 14,'figure.figsize': (7, 5.5)})  # 'figure.figsize': (7, 4.8)


if __name__ == "__main__":
    main()
