import numpy as np
import matplotlib.pyplot as plt

from project.model import coherence_from_data as coherence
from project.model.sample import Alexa647, Sensor, Emitter, expected_excitation_time


def single_emitter():
    e = Alexa647(x=665 * 3.5 / 4, y=665 * 3.5 / 4)
    s = Sensor(7, pixel_size=665 / 4)
    interval = 10 ** 5  # ns
    laser = 333 * 10 ** 3  # W / cm2
    bin_size = 0.05
    eta = 1
    statistics = "sub_poisson"

    e.generate_photons(laser_power=laser, time_interval=interval, seed=100, detection_efficiency=eta,
                       statistics=statistics)
    s.measure(emitters=[e])
    # correlation = second_order_coherence_all_pixels(s, interval, bin_size)
    # show_zero_coherence_all_pixels(s, correlation)

    c, l = coherence.auto_coherence(s.data[:, 2], interval, bin_size, 200, normalize=True)
    print(c[0])
    coherence.show_coherence(l, c, show_fit=True, title=f"Lifetime: {e.lifetime} ns")


def multiple_emitters_pixel_correlation(nr_emitters, nr_iterations):
    s = Sensor(7, 665 / 4)
    laser = 300 * 10 ** 3  # W / cm2
    interval = 10 ** 5  # ns
    statistics = "sub_poisson"
    eta = 1
    bin_size = 0.01
    positions = np.array([
        [665 * 3.5 / 4, 665 * 3.5 / 4],
        [665 * 3.5 / 4, 665 * 3.5 / 4],
        [665 * 3.5 / 4, 665 * 3.5 / 4],
        [665 * 3.5 / 4, 665 * 3.5 / 4]
    ])

    c_array = np.zeros((nr_iterations, 4, 7, 7))

    for k in range(nr_iterations):
        print(f"Iteration {k}")
        emitters = []
        for j in range(nr_emitters):
            e = Alexa647(x=positions[j, 0], y=positions[j, 1])
            e.generate_photons(laser_power=laser, time_interval=interval, seed=133 * k + j, detection_efficiency=eta,
                               statistics=statistics)
            emitters.append(e)
        s.measure(emitters)
        # s.show_measured_photons()
        # plt.show()

        for degree in range(3):
            nn_coherence = coherence.denoised_auto_coherence(s, interval, bin_size, degree)
            # nn_coherence = np.where(nn_coherence < 1.0, nn_coherence, 0)
            c_array[k, degree, :, :] = nn_coherence
            # s.show(nn_coherence, vmin=0.0, vmax=1.0, title=f"Auto-coherence - averaged with {degree * 4} nearest neighbours")
            # plt.show()

        kernel_coherence = coherence.auto_coherence_per_neighbourhood(s, interval, bin_size)
        c_array[k, 3, :, :] = kernel_coherence
        # s.show(kernel_coherence, vmin=0.0, vmax=1.0, title="Auto-coherence per 3x3 pixels area")
        # plt.show()

        s.clear_sensor()
    np.save(f"../data/meeting_23_12_07/denoised_autocoherence/binsize_{bin_size}_laser_{laser / 10 ** 3}kwcm2_"
            f"interval{interval / 10 ** 6}ms_eta_{eta}_emitters_{nr_emitters}_iterations_{nr_iterations}.npy", c_array)
    return c_array


def plot_pixel_correlation():
    s = Sensor(7, 665 / 4)
    n = 2
    distance = 40
    c_array = multiple_emitters_pixel_correlation(n)

    # c_array = np.load(f'../data/meeting_23_11_23_coherence/moving/emitter{n}_distance{distance}_all.npy')

    for i in range(np.shape(c_array)[1]):
        for k in range(np.shape(c_array)[0]):
            plt.plot(s.x_centers[:, 0], np.mean(c_array[k, i, :, :], axis=1), label=distance * k)
        plt.legend(title='Offset x (nm)')
        plt.xlabel('x (nm)')
        plt.title(r'$g^{(2)}(0)$ averaged over y direction' + f' for {n} emitters')
        plt.show()

        for k in range(np.shape(c_array)[0]):
            plt.plot(s.y_centers[0, :], np.mean(c_array[k, i, :, :], axis=0), label=distance * k)
        plt.legend(title='Offset x (nm)')
        plt.xlabel('y (nm)')
        plt.title(r'$g^{(2)}(0)$ averaged over x direction' + f' for {n} emitters')
        plt.show()


def plot_nn_pixel_correlation():
    s = Sensor(7, 665 / 4)
    laser = 300 * 10 ** 3  # W / cm2
    interval = 10 ** 5  # ns
    eta = 1
    bin_size = 0.01

    for i in range(4):
        c_array = np.load(
            f"../data/meeting_23_12_07/denoised_autocoherence/binsize_{bin_size}_laser_{laser / 10 ** 3}kwcm2_"
            f"interval{interval / 10 ** 6}ms_eta_{eta}_emitters_{i + 1}_iterations_10.npy")
        c_array_av = np.mean(c_array, axis=0)
        c_array_std = np.std(c_array, axis=0) * 100

        for degree in range(3):
            s.show(c_array_av[degree, :, :], vmin=0.0, vmax=1.0,
                   title=f"Auto-coherence - {degree * 4} NN, {i + 1} emitters")
            plt.errorbar(s.x_centers.flatten(), s.y_centers.flatten(), xerr=c_array_std[degree, :, :].flatten(),
                         yerr=c_array_std[degree, :, :].flatten(), linestyle='', marker='', color='red')
            plt.savefig(f'../figures/meeting_23_12_07/denoised_autocoherence/binsize_{bin_size}_laser_{laser / 10 ** 3}'
                        f'kwcm2_interval{interval / 10 ** 6}ms_eta_{eta}_emitters_{i + 1}_iterations_10_degree_{degree}'
                        f'.png')
            plt.show()
            print(f"Nearest neighbours {degree * 4}")
            print(np.round(c_array_av[degree, :, :], 2))
            print(np.round(c_array_std[degree, :, :], 2))

        s.show(c_array_av[3, :, :], vmin=0.0, vmax=1.0, title=f"Auto-coherence per 3x3 pixels area, {i + 1} emitters")
        plt.errorbar(s.x_centers.flatten(), s.y_centers.flatten(), xerr=c_array_std[3, :, :].flatten(),
                     yerr=c_array_std[3, :, :].flatten(), linestyle='', marker='', color='red')
        plt.savefig(f'../figures/meeting_23_12_07/denoised_autocoherence/binsize_{bin_size}_laser_{laser / 10 ** 3}'
                    f'kwcm2_interval{interval / 10 ** 6}ms_eta_{eta}_emitters_{i + 1}_iterations_10_degree_{3}.png')
        plt.show()
        print("3x3 area")
        print(np.round(c_array_av[3, :, :], 2))
        print(np.round(c_array_std[3, :, :], 2))


def multiple_emitters_accuracy():
    s = Sensor(7, 665 / 4)
    laser = 300 * 10 ** 3  # W / cm2
    interval = 10 ** 5  # ns
    statistics = "sub_poisson"
    eta = 1
    bin_size = 0.01

    positions = np.array([[665 * 3.5 / 4, 665 * 3.5 / 4]])
    nr_emitters = 4

    N = 100
    c_array = np.zeros((N, nr_emitters))

    for k in range(N):
        print(k)
        emitters = []
        for j in range(nr_emitters):
            e = Alexa647(x=positions[0, 0], y=positions[0, 1])
            e.generate_photons(laser_power=laser, time_interval=interval, seed=264 * k + j, detection_efficiency=eta,
                               statistics=statistics)
            emitters.append(e)

            arrival_times = s.measure(emitters)[:, 2]
            c, _ = coherence.auto_coherence(arrival_times, interval, bin_size, nr_steps=1, normalize=True)
            c_array[k, j] = c[0]
            s.clear_sensor()

    average = np.mean(c_array, axis=0)
    std = np.std(c_array, axis=0)
    for l in range(nr_emitters):
        print(f"Number of emitters: {l + 1}. Average g2(0): {np.round(average[l], 3)} +- {np.round(std[l], 4)}")


def multiple_emitters_off_center(nr_emitters, f):
    s = Sensor(7, 665 / 4)
    laser = 300 * 10 ** 3  # W / cm2
    interval = 10 ** 5  # ns
    statistics = "sub_poisson"
    eta = 1
    bin_size = 0.01

    N = 200
    positions = np.random.default_rng(seed=846).normal(
        loc=665 * 3.5 / 4,
        scale=665 * 3.5 / 4,
        size=(N, nr_emitters, 2)
    )
    positions[:, 0, :] = 665 * 3.5 / 4
    fraction = np.zeros(N)
    c_array = np.zeros(N)

    for k in range(N):
        print(k)
        emitters = []
        total_nr_photons = 0

        for j in range(nr_emitters):
            e = Alexa647(x=positions[k, j, 0], y=positions[k, j, 1])
            photons = e.generate_photons(laser_power=laser, time_interval=interval, seed=157 * k + j,
                                         detection_efficiency=eta, statistics=statistics)
            total_nr_photons += len(photons)
            emitters.append(e)

        arrival_times = s.measure(emitters)[:, 2]

        c, _ = coherence.auto_coherence(arrival_times, interval, bin_size, nr_steps=1, normalize=True)
        s.clear_sensor()

        fraction[k] = len(arrival_times) / total_nr_photons
        c_array[k] = c[0]

    np.save(f + f"{nr_emitters}/fraction.npy", fraction)
    np.save(f + f"{nr_emitters}/coherence.npy", c_array)


def plot_coherence_off_center(nr_emitters, f):
    fraction = np.load(f + f"{nr_emitters}/fraction.npy")
    c = np.load(f + f"{nr_emitters}/coherence.npy")

    plt.plot(fraction, c, linestyle='', marker='.')
    plt.xlabel("Fraction of detected photons")
    plt.ylabel(r"$g^{(2)}(0)$")
    plt.axhline(1 - 1 / nr_emitters, color='grey', linestyle='--', label='Expected')
    plt.title(f'Coherence for {nr_emitters} emitters')
    plt.legend()
    plt.ylim(-0.1, 1)
    plt.xlim(0, 1)
    plt.savefig(f + f"{nr_emitters}/plot.png")
    plt.show()


def dependency_excitation_emission_ratio(nr_ratios, nr_reps, nr_emitters, binsize, interval, eta):
    # Actually is alpha, but since alpha / laser_0 = 1 for the case lifetime = 1, alpha = laser_0
    laser_0 = expected_excitation_time(240000, 660, 1)
    lifetime_0 = 1

    ratio = np.logspace(-5, 5, num=nr_ratios, endpoint=True)
    d_lifetime = (lifetime_0 - ratio) / (-1 - ratio)

    lifetime = lifetime_0 + d_lifetime
    laser = laser_0 / (1 - d_lifetime)
    excitation = expected_excitation_time(240000, 660, laser)

    s = Sensor(7, 665 / 4)
    position = np.array([665 * 3.5 / 4, 665 * 3.5 / 4])
    c_array = np.zeros((nr_ratios, nr_emitters, nr_reps))

    for i in range(nr_ratios):
        print(f"ratio: {i}")
        for k in range(nr_reps):
            print(k)
            emitters = []
            for j in range(nr_emitters):
                e = Emitter(position[0], position[1], absorption_wavelength=660, emission_wavelength=665,
                            lifetime=lifetime[i], extinction_coefficient=240000)
                e.generate_photons(laser[i], interval, seed=309 * k + j, detection_efficiency=eta)
                emitters.append(e)

                arrival_times = s.measure(emitters)[:, 2]
                c, _ = coherence.auto_coherence(arrival_times, interval, binsize, nr_steps=1, normalize=True)
                c_array[i, j, k] = c[0]
                s.clear_sensor()

    f = f"../data/meeting_23_12_07/binsize_{binsize}/coherence_{nr_ratios}ratios_{nr_reps}reps_{nr_emitters}emitters_{eta}eta_{interval / 10 ** 6}msinterval"
    np.save(f + "_coherence.npy", c_array)
    np.save(f + "_ratio.npy", lifetime / excitation)


def excitation_emission_ratio_k_constant(nr_ratios, nr_reps, nr_emitters, binsize, interval, eta=1):
    laser_0 = expected_excitation_time(240000, 660, 1)  # Equals 1/\alpha
    alpha = 1 / laser_0
    lifetime_0 = 1

    ratio = np.logspace(-2, -0.001, num=nr_ratios, endpoint=True)
    d_laser = laser_0 * (ratio - lifetime_0) / (lifetime_0 * (ratio + 1))

    lifetime = lifetime_0 / (1 - alpha * d_laser * lifetime_0)
    laser = laser_0 + d_laser
    print(f"Ratio: {np.round(ratio, 7)}")
    print(f"Laser: {np.round(laser, 2)}")
    print(f"Lifetime: {np.round(lifetime, 2)}")
    print(f"Expected T: {np.round(expected_excitation_time(240000, 660, laser) + lifetime)}")
    print(f"Excitation rate: {1 / expected_excitation_time(240000, 660, laser)}")
    print(f"Emission rate: {1 / lifetime}")
    print(f"k: {1 / expected_excitation_time(240000, 660, laser) + 1 / lifetime}")

    s = Sensor(7, 665 / 4)
    position = np.array([665 * 3.5 / 4, 665 * 3.5 / 4])
    c_array = np.zeros((nr_ratios, nr_emitters, nr_reps))

    for i in range(nr_ratios):
        print(f"ratio: {i}")
        for k in range(nr_reps):
            print(k)
            emitters = []
            for j in range(nr_emitters):
                e = Emitter(position[0], position[1], absorption_wavelength=660, emission_wavelength=665,
                            lifetime=lifetime[i], extinction_coefficient=240000)
                e.generate_photons(laser[i], interval, seed=309 * k + j, detection_efficiency=eta,
                                   statistics="sub_poisson")
                emitters.append(e)

                arrival_times = s.measure(emitters)[:, 2]
                c, _ = coherence.auto_coherence(arrival_times, interval, binsize, nr_steps=1, normalize=True)
                c_array[i, j, k] = c[0]
                s.clear_sensor()

    f = f"../data/report/binsize_{binsize}_coherence_{nr_ratios}ratios_{nr_reps}reps_{nr_emitters}emitters_{eta}eta_{interval / 10 ** 6}msinterval"
    np.save(f + "_coherence.npy", c_array)
    np.save(f + "_ratio.npy", ratio)


def plot_ratio_dependency(nr_ratios, nr_reps, nr_emitters, binsize, interval, eta):
    f = f"../data/report/binsize_{binsize}_coherence_{nr_ratios}ratios_{nr_reps}reps_{nr_emitters}emitters_{eta}eta_{interval / 10 ** 6}msinterval"
    c_array = np.load(f + "_coherence.npy")
    ratio = np.load(f + "_ratio.npy")

    average = np.mean(c_array, axis=2)
    std = np.std(c_array, axis=2)

    for l in range(np.shape(c_array)[1]):
        plt.errorbar(ratio, average[:, l], yerr=std[:, l], label=l + 1, linestyle='', marker='.')
        plt.axhline(1 - 1 / (l + 1), color='grey', linestyle='--')
        # print(f"N = {l+1}. Expected: {np.round(1 - 1/(l+1), 3)}. Found: {np.round(average[12, l], 3)}"
        #       f" +- {np.round(std[12, l], 3)}. Absolute error: {np.round(average[12, l] - (1 - 1/(l+1)), 3)}")

    plt.legend(title="# emitters")
    plt.xscale('log')
    plt.xlabel(r"$\sigma I \tau_l$")
    plt.ylabel(r"$g^{(2)}(0)$")
    plt.ylim(-0.05, 1.05)
    plt.show()


def dependency_eta(nr_etas, nr_reps, nr_emitters, binsize, interval_at_one):
    # Actually is alpha, but since alpha / laser = 1 for the case lifetime = 1, alpha = laser
    laser = expected_excitation_time(240000, 660, 1)
    lifetime = 1
    eta_array = np.linspace(0.01, 1, nr_etas, endpoint=True)
    interval_array = (1 * interval_at_one) / eta_array

    s = Sensor(7, 665 / 4)
    position = np.array([665 * 3.5 / 4, 665 * 3.5 / 4])
    c_array = np.zeros((nr_etas, nr_emitters, nr_reps))

    for i in range(nr_etas):
        print(f"eta: {i}")
        for k in range(nr_reps):
            print(k)
            emitters = []
            for j in range(nr_emitters):
                e = Emitter(position[0], position[1], absorption_wavelength=660, emission_wavelength=665,
                            lifetime=lifetime, extinction_coefficient=240000)
                e.generate_photons(laser, interval_array[i], seed=309 * k + j, detection_efficiency=eta_array[i])
                emitters.append(e)

                arrival_times = s.measure(emitters)[:, 2]
                c, _ = coherence.auto_coherence(arrival_times, interval_array[i], binsize, nr_steps=1, normalize=True)
                c_array[i, j, k] = c[0]
                s.clear_sensor()

    f = f"../data/meeting_23_12_07/binsize_{binsize}/coherence_{nr_etas}etas_{nr_reps}reps_{nr_emitters}emitters_{interval_at_one / 10 ** 6}msintervalatone"
    np.save(f + "_coherence.npy", c_array)
    np.save(f + "_etas.npy", eta_array)


def plot_eta_dependency(nr_etas, nr_reps, nr_emitters, binsize, interval_at_one):
    f = f"../data/meeting_23_12_07/binsize_{binsize}/coherence_{nr_etas}etas_{nr_reps}reps_{nr_emitters}emitters_{interval_at_one / 10 ** 6}msintervalatone"
    c_array = np.load(f + "_coherence.npy")[:, :, :10]
    etas = np.load(f + "_etas.npy")
    percentage = etas * 100

    average = np.mean(c_array, axis=2)
    std = np.std(c_array, axis=2)

    for l in range(np.shape(c_array)[1]):
        plt.plot(percentage, average[:, l], label=l + 1, linestyle='-', marker='')
        plt.fill_between(percentage, average[:, l] - std[:, l], average[:, l] + std[:, l], alpha=0.4)
        # plt.errorbar(percentage, average[:, l], yerr=std[:, l], label=l+1, marker=".", linestyle='-')
        plt.axhline(1 - 1 / (l + 1), color='grey', linestyle='--')

    plt.legend(title="# emitters", loc='lower right')
    plt.xlabel(r"$\eta$ [%]")
    plt.ylabel(r"$g^{(2)}(0)$")
    plt.ylim(-0.05, 1.05)
    plt.show()


def dependency_nr_photons(nr_steps, nr_reps, nr_emitters, bin_size, max_nr_photons, eta):
    laser = expected_excitation_time(240000, 660, 1)
    nr_photons = np.linspace(10, max_nr_photons, nr_steps)
    intervals = nr_photons * 2 / eta  # The expected cycle time is 1 + 1 = 2.
    s = Sensor(7, 665 / 4)
    position = np.array([665 * 3.5 / 4, 665 * 3.5 / 4])
    c_array = np.zeros((nr_steps, nr_emitters, nr_reps))

    for i in range(nr_steps):
        print(f"interval: {i} / {nr_steps}")
        for k in range(nr_reps):
            print(k)
            emitters = []
            for j in range(nr_emitters):
                e = Alexa647(position[0], position[1])
                e.generate_photons(laser, intervals[i], seed=87 * i + 10 * k + j, detection_efficiency=eta)
                emitters.append(e)

                arrival_times = s.measure(emitters)[:, 2]
                c, _ = coherence.auto_coherence(arrival_times, intervals[i], bin_size, nr_steps=1, normalize=True)
                c_array[i, j, k] = c[0]
                s.clear_sensor()

    f = f"../data/meeting_23_12_15/binsize_{bin_size}/coherence_{nr_steps}intervals_{nr_reps}reps_{nr_emitters}emitters_{max_nr_photons}_maxphotons_{eta}eta"
    np.save(f + "_coherence.npy", c_array)
    np.save(f + "_nr_photons.npy", nr_photons)


def plot_nr_photons_dependency(nr_steps, nr_reps, nr_emitters, bin_size, max_nr_photons, eta):
    f = f"../data/meeting_23_12_15/binsize_{bin_size}/coherence_{nr_steps}intervals_{nr_reps}reps_{nr_emitters}emitters_{max_nr_photons}_maxphotons_{eta}eta"
    c_array = np.load(f + "_coherence.npy")
    nr_photons = np.load(f + "_nr_photons.npy")
    average = np.mean(c_array, axis=2)
    std = np.std(c_array, axis=2)

    for i in range(np.shape(c_array)[1]):
        plt.plot(nr_photons, average[:, i], label=i + 1, linestyle='-', marker='')
        plt.fill_between(nr_photons, average[:, i] - std[:, i], average[:, i] + std[:, i], alpha=0.4)
        # plt.errorbar(percentage, average[:, l], yerr=std[:, l], label=l+1, marker=".", linestyle='-')
        plt.axhline(1 - 1 / (i + 1), color='grey', linestyle='--')

    plt.legend(title="# emitters", loc='lower right')
    plt.xlabel("Expected number of photons per emitter")
    plt.ylabel(r"$g^{(2)}(0)$")
    plt.ylim(-0.05, 1.05)
    plt.savefig(f"../figures/meeting_23_12_15/binsize_{bin_size}/coherence_{nr_steps}intervals_{nr_reps}reps_"
                f"{nr_emitters}emitters_{max_nr_photons}_maxphotons_{eta}eta.png")
    plt.show()


def plot_coherence_function():
    def func(n, k, delay):
        return 1 - np.exp(-k * delay) / n

    s = Sensor(7, 665 / 4)
    # laser = 330 * 10 ** 3  # W / cm2
    laser = 10 * 10**3
    lifetime = 1
    eta = 0.1
    factor = 150
    interval = factor * 10 ** 5  # ns
    statistics = "sub_poisson"
    bin_size = 0.01
    kex = 1/expected_excitation_time(240000, 650, laser)
    kem = 1/lifetime
    k = kex + kem
    print(f"Excitation rate: {kex}")
    print(f"Emission rate: {kem}")
    print(f"k: {k}")
    print(f"Expected interval: {(1/eta) * (1/kex + 1/kem)}")

    positions = np.array([[665 * 3.5 / 4, 665 * 3.5 / 4]])

    emitters = []
    nr_emitters = 1
    for j in range(nr_emitters):
        e = Alexa647(x=positions[0, 0], y=positions[0, 1])
        e.lifetime = lifetime
        e.generate_photons(laser_power=laser, time_interval=interval, seed=4315 + j, detection_efficiency=eta,
                           statistics=statistics)
        emitters.append(e)

    arrival_times = s.measure(emitters)[:, 2]
    print(len(arrival_times))
    c, lag = coherence.auto_coherence(arrival_times, interval, bin_size, nr_steps=1, normalize=True)
    print(f"{nr_emitters} emitters. g2(0) = {c[0]}")
    f = f'../../report/results/coherence_sub_poisson_{nr_emitters}emitters_binsize{bin_size}_eta{eta}_{factor}T'
    # f = f'../../report/results/coherence_sub_poisson_{nr_emitters}emitters_binsize{bin_size}_ratio100_kconst_{factor}T'
    # f = f'../../report/results/coherence_poisson_{nr_emitters}emitters_binsize{bin_size}'

    # Plotting
    plt.plot(lag, c, label='Simulation', alpha=0.75)
    plt.plot(np.linspace(0, lag[-1], 500), func(nr_emitters, k, np.linspace(0, lag[-1], 500)), linestyle='--', color='black', marker='',
             label=r'$g^{(2)}[\ell]=1-e^{-k\ell \,\Delta t}$')
    # plt.legend(loc='lower right')
    plt.xlabel(r'$\ell \Delta t$ [ns]')
    plt.ylabel(r'$g^{(2)}[\ell]$')
    plt.ylim(-0.01, 1.25)
    plt.tight_layout()
    # plt.savefig(fname=f + '.svg')
    plt.show()

    s.clear_sensor()


def check_minor_differences():
    def func(n, k, delay):
        return 1 - np.exp(-k * delay) / n

    s = Sensor(7, 665 / 4)
    laser = 330 * 10 ** 3  # W / cm2
    # laser = 6.6 * 10**3
    lifetime = 1
    eta = 1
    factor = 1
    interval = factor * 10 ** 5  # ns
    statistics = "sub_poisson"
    bin_size = 1
    k = 1/expected_excitation_time(240000, 650, laser) + 1/lifetime
    print(k)
    print(expected_excitation_time(240000, 650, laser) + lifetime)

    positions = np.array([[665 * 3.5 / 4, 665 * 3.5 / 4]])

    lifetime_factor = [0.5, 1, 1, 1]
    laser_factor = [1, 1.5, 1, 1]
    nr_emitters = 2
    N = 50
    nr_steps = 2
    c = np.zeros((N, nr_steps))
    for l in range(N):
        if l % 10 == 0:
            print(l)
        emitters = []
        for j in range(nr_emitters):
            e = Alexa647(x=positions[0, 0], y=positions[0, 1])
            e.lifetime = lifetime * lifetime_factor[j]
            e.generate_photons(laser_power=laser*laser_factor[j], time_interval=interval, seed=l * 97 + 18 + j, detection_efficiency=eta,
                               statistics=statistics)
            emitters.append(e)

        arrival_times = s.measure(emitters)[:, 2]
        c_temp, lag = coherence.auto_coherence(arrival_times, interval, bin_size, nr_steps, normalize=True)
        c[l, :] = c_temp
        s.clear_sensor()

    # Plotting
    plt.plot(np.linspace(0, lag[-1], 500), func(nr_emitters, k, np.linspace(0, lag[-1], 500)), linestyle='--', color='black', marker='')
    plt.plot(lag, np.mean(c, axis=0), label='Simulation', alpha=1, color='red')
    plt.fill_between(lag, np.mean(c, axis=0) - np.std(c, axis=0), np.mean(c, axis=0) + np.std(c, axis=0))
    print(np.std(c[:, 0]))
    print(np.mean(c[:, 0]))

    # plt.legend(loc='lower right')
    plt.xlabel(r'$\ell \Delta t$ [ns]')
    plt.ylabel(r'$g^{(2)}[\ell]$')
    plt.ylim(-0.01, 1.25)
    plt.tight_layout()
    # plt.savefig(fname=f + '.svg')
    plt.show()



def main():
    plt.rcParams.update({'font.size': 14, 'figure.figsize': (7, 5.5)})  # 'figure.figsize': (7, 4.8)


if __name__ == '__main__':
    main()
