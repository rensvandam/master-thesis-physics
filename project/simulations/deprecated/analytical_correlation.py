import time
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import curve_fit

import project.model.coherence_analytical as analytical
from project.model.sample import expected_excitation_time


def plot_coherence_at_zero():
    max_bin = 5
    param = '1ratio'
    f = f"/report/bin{max_bin}/32pixels_100iter_4nr_emitters_1eta_" + param
    coh = np.load("../data" + f + "_coherence.npy")[0, 1:, :, :]  # Coherence for only one ratio
    coh_per_bin = np.mean(coh, axis=1)  # coh_per_bin[nr emitters, bin size]
    std_per_bin = np.std(coh, axis=1)
    bin_sizes = np.load("../data" + f + "_bin_size.npy")
    colors = ['r', 'b', 'g', 'orange']

    laser = np.load("../data" + f + "_laser.npy")[0]
    excitation_rate = 1 / expected_excitation_time(240000, 650, laser)
    lt = np.load("../data" + f + "_lifetime.npy")[0]
    emission_rate = 1 / lt
    print(f"Laser: {laser}, excitation rate: {excitation_rate}")
    print(f"Lifetime: {lt}, emission rate: {emission_rate}")

    # # Legend
    # plt.plot(0, -10, linestyle="--", marker='', label='Prediction', color='black')
    # plt.plot(0, -10, linestyle='-', marker='', label='Simulation', color='black', alpha=0.4)
    # handles, labels = plt.gca().get_legend_handles_labels()
    for i in range(0, 4):
        # # Legend
        # handles.append(mpatches.Patch(color=colors[i]))
        # labels.append(f'n = {i+1}')

        # Plot data
        plt.fill_between(
            bin_sizes,
            coh_per_bin[i, :] - std_per_bin[i, :],
            coh_per_bin[i, :] + std_per_bin[i, :],
            color=colors[i], alpha=0.3
        )
        plt.plot(
            bin_sizes,
            analytical.expected_coherence(i + 1, bin_sizes, excitation_rate, emission_rate),
            linestyle=(0, (5, 5)), color=colors[i]
        )
    # plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
    # plt.ylim([-0.01, 1.01])
    plt.xlabel(r'$\Delta t$ [ns]')
    plt.ylabel(r'$g^{(2)}[0]$')
    plt.tight_layout()
    # plt.savefig(f"../../report/results/coherence_zero_bin{max_bin}_" + param + ".svg")
    plt.show()


def plot_coherence_at_zero_variance(param):
    max_bin = 5
    f = f"/report/bin{max_bin}/32pixels_100iter_4nr_emitters_1eta_" + param
    coh = np.load("../data" + f + "_coherence.npy")[0, 1:, :, :]  # Coherence for only one ratio
    coh_per_bin = np.mean(coh, axis=1)  # coh_per_bin[nr emitters, bin size]
    std_per_bin = np.std(coh, axis=1)
    bin_sizes = np.load("../data" + f + "_bin_size.npy")
    colors = ['r', 'b', 'g', 'orange']

    laser = np.load("../data" + f + "_laser.npy")[0]
    excitation_rate = 1 / expected_excitation_time(240000, 650, laser)
    lt = np.load("../data" + f + "_lifetime.npy")[0]
    emission_rate = 1 / lt
    plt.plot(-1, -1, label='Simulation', color='black', linestyle='-', marker='')
    plt.plot(-1, -1, label='Analytical result', color='black', linestyle='--', marker='')
    for i in range(0, 4):
        plt.plot(bin_sizes, std_per_bin[i, :], linestyle='-', marker='', color=colors[i], alpha=1)
        var = analytical.variance_coherence(i+1, bin_sizes, 10**5, excitation_rate, emission_rate)
        plt.plot(bin_sizes, np.sqrt(var), color=colors[i], linestyle='--', marker='')
    plt.ylim([0, 0.02])
    plt.xlim([-0.2, 5.2])
    # plt.legend()
    plt.xlabel(r'$\Delta t$ [ns]')
    # plt.ylabel(r'$\sigma_g$')
    plt.tight_layout()
    # plt.savefig(f"../../report/results/emitters_bin{max_bin}_" + param + "_std_g.svg")
    plt.show()


def plot_number_of_emitters():
    max_bin = 5
    param = '1ratio'
    f = f"/report/bin{max_bin}/32pixels_100iter_4nr_emitters_1eta_" + param
    coh = np.load("../data" + f + "_coherence.npy")[0, 1:, :, :]  # Coherence for only one ratio
    coh_mean = np.mean(coh, axis=1)
    bin_sizes = np.load("../data" + f + "_bin_size.npy")
    laser = np.load("../data" + f + "_laser.npy")[0]
    lifetime = np.load("../data" + f + "_lifetime.npy")[0]
    excitation_rate = 1 / expected_excitation_time(240000, 650, laser)
    emission_rate = 1 / lifetime
    print(f"Laser: {laser}, excitation rate: {excitation_rate}")
    print(f"Lifetime: {lifetime}, emission rate: {emission_rate}")

    nr_emitters = analytical.expected_number_of_emitters(coh, bin_sizes, excitation_rate, emission_rate)  # [n, it, bins]
    nr_emitters_per_bin = np.mean(nr_emitters, axis=1)  # nr_emitters[n, bin size]
    std_per_bin = np.std(nr_emitters, axis=1)

    # # Legend
    # plt.plot(0, -10, linestyle="--", marker='', label='Analytical solution', color='black')
    # plt.plot(0, -10, linestyle='-', marker='', label='Simulation', color='black', alpha=0.4)
    # handles, labels = plt.gca().get_legend_handles_labels()
    colors = ['r', 'b', 'g', 'orange']
    for i in range(4):
        # # Legend
        # handles.append(mpatches.Patch(color=colors[i]))
        # labels.append(f'n = {i + 1}')

        # Plot data
        plt.fill_between(
            bin_sizes,
            nr_emitters_per_bin[i, :] - std_per_bin[i, :],
            nr_emitters_per_bin[i, :] + std_per_bin[i, :],
            color=colors[i], alpha=0.3
        )
        plt.plot(bin_sizes, (i + 1) * np.ones(len(bin_sizes)), linestyle=(0, (5, 5)), color=colors[i])

        sigma_n = np.sqrt(
            analytical.variance_number_of_emitters(i + 1, bin_sizes, 10 ** 5, excitation_rate, emission_rate))
        plt.plot(bin_sizes, i + 1 + sigma_n, color=colors[i])

    # plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
    # plt.ylim([0.55, 4.45])
    plt.xlabel(r'$\Delta t$ [ns]')
    plt.ylabel(r'$n$')
    plt.tight_layout()
    # plt.savefig(f"../../report/results/emitters_bin{max_bin}_" + param + ".svg")
    plt.show()


def plot_number_of_emitters_variance(param):
    max_bin = 5
    f = f"/report/bin{max_bin}/32pixels_100iter_4nr_emitters_1eta_" + param
    coh = np.load("../data" + f + "_coherence.npy")[0, 1:, :, :]  # Coherence for only one ratio
    coh_mean = np.mean(coh, axis=1)
    bin_sizes = np.load("../data" + f + "_bin_size.npy")
    laser = np.load("../data" + f + "_laser.npy")[0]
    lifetime = np.load("../data" + f + "_lifetime.npy")[0]
    excitation_rate = 1 / expected_excitation_time(240000, 650, laser)
    emission_rate = 1 / lifetime

    nr_emitters = analytical.expected_number_of_emitters(coh, bin_sizes, excitation_rate,
                                                         emission_rate)  # [n, it, bins]
    std_per_bin = np.std(nr_emitters, axis=1)
    colors = ['r', 'b', 'g', 'orange']
    plt.plot(-1, -1, label='Simulation', color='black', linestyle='-', marker='')
    plt.plot(-1, -1, label='Analytical result', color='black', linestyle='--', marker='')
    for i in range(4):
        plt.plot(bin_sizes, std_per_bin[i, :], linestyle='-', marker='', color=colors[i])

        b = np.linspace(0.01, 10, 1000)
        sigma_n = np.sqrt(analytical.variance_number_of_emitters(i+1, b, 10**5, excitation_rate, emission_rate))
        plt.plot(b, sigma_n, color=colors[i], linestyle='--')

        print(f"EMITTER {i}")
        print(b[np.argmin(sigma_n)])
    # plt.legend()
    plt.ylim([0, 0.5])
    plt.xlim([-0.2, 5.2])
    plt.xlabel(r'$\Delta t$ [ns]')
    # plt.ylabel(r'$\sigma_n$')
    plt.tight_layout()
    # plt.savefig(f"../../report/results/emitters_bin{max_bin}_" + param + "_std_n.svg")
    plt.show()


def plot_large_number_of_emitters(param):
    bin_size = "1"
    max_nr_emitters = 100
    f = f"/report/bin{bin_size}/32pixels_100iter_{max_nr_emitters}nr_emitters_1eta_" + param
    coh = np.load("../data" + f + "_coherence.npy")[0, 1:, :, 0]  # coh[n, it]
    coh_mean = np.mean(coh, axis=1)
    bin_sizes = np.load("../data" + f + "_bin_size.npy")
    laser = np.load("../data" + f + "_laser.npy")[0]
    lifetime = np.load("../data" + f + "_lifetime.npy")[0]
    excitation_rate = 1 / expected_excitation_time(240000, 650, laser)
    emission_rate = 1 / lifetime
    print(f"Laser: {laser}, excitation rate: {excitation_rate}")
    print(f"Lifetime: {lifetime}, emission rate: {emission_rate}")

    nr_emitters_it = analytical.expected_number_of_emitters(coh, bin_sizes, excitation_rate, emission_rate)  # [n, it]
    # nr_emitters_it = np.round(nr_emitters_it, decimals=0)
    nr_emitters = np.mean(nr_emitters_it, axis=1)  # nr_emitters[n]
    std = np.std(nr_emitters_it, axis=1)
    n_0 = np.arange(1, max_nr_emitters + 1)
    print(np.mean(nr_emitters/n_0))
    print(np.mean(std/n_0))
    print(np.std(std/n_0))

    y_data = nr_emitters
    error = std
    sigma_n = np.sqrt(analytical.variance_number_of_emitters(n_0, bin_sizes, 10 ** 5, excitation_rate, emission_rate)/n_0)
    # plt.plot(n_0, n_0 + sigma_n, color='red', linestyle='--')
    plt.errorbar(n_0, y_data, yerr=error, linestyle='', marker='.', capsize=2, markersize=4)
    plt.xlabel(r'$n_0$')
    plt.ylabel(r'$n$')
    plt.tight_layout()
    # plt.savefig(f"../../report/results/{max_nr_emitters}emitters_bin{max_bin}_" + param + "_large.svg")
    plt.show()

    y_data = nr_emitters - n_0
    error = std
    # plt.plot(n_0, sigma_n, color='red', linestyle='--')
    _, caps, bars = plt.errorbar(n_0, y_data, yerr=error, linestyle='', marker='.', capsize=0, markersize=4)
    [bar.set_alpha(0.5) for bar in bars]
    [cap.set_alpha(0.5) for cap in caps]
    plt.xlabel(r'$n_0$')
    plt.ylabel(r'$n - n_0$')
    # plt.ylim([-0.05, 0.05])
    plt.tight_layout()
    # plt.savefig(f"../../report/results/{max_nr_emitters}emitters_bin{max_bin}_" + param + "_large_absolute_error.svg")
    plt.show()

    y_data = (nr_emitters - n_0) / n_0
    error = std / n_0
    # plt.plot(n_0, sigma_n/n_0, color='red', linestyle='--')
    _, caps, bars = plt.errorbar(n_0, y_data, yerr=error, linestyle='', marker='.', capsize=0, markersize=4)
    [bar.set_alpha(0.5) for bar in bars]
    [cap.set_alpha(0.5) for cap in caps]
    plt.xlabel(r'$n_0$')
    plt.ylabel(r'$(n-n_0)/n_0$')
    # plt.ylim([-0.05, 0.05])
    plt.tight_layout()
    # plt.savefig(f"../../report/results/{max_nr_emitters}emitters_bin{max_bin}_" + param + "_large_relative_error.svg")
    plt.show()

    # plt.plot(n_0, std, color='black', linestyle='-', label='Simulation')
    # sigma_g = np.sqrt(analytical.variance_coherence(n_0, bin_sizes, 10 ** 5, 1, 1))[:, 0]
    # sigma_n = n_0**2 * sigma_g / analytical.discrete_deviation_coherence(bin_sizes, k=2)
    # plt.plot(n_0, sigma_n, color='black', linestyle='--', label='Analytical result')
    # plt.legend()
    # plt.show()


def load_ratio_data(cst='k'):
    f = '../data/report/32pixels_10iter_1_nr_emitters_1eta_ratio_1_100_' + cst + 'const_'
    bin_sizes = np.load(f + 'bin_size.npy')

    coherence = np.load(f + 'coherence.npy')  # [ratio, emitter, iterations, bin size]
    coherence = coherence[:, 1, :, :]  # [ratio, iterations, bin size]
    coherence = np.mean(coherence, axis=1)  # [ratio, bin size]

    lifetime = np.load(f + 'lifetime.npy')
    laser = np.load(f + 'laser.npy')
    excitation_time = expected_excitation_time(240000, 660, laser)

    k = 1 / excitation_time + 1 / lifetime
    interval = excitation_time + lifetime

    if cst == 'k':
        ratio = (1 / excitation_time) / (1 / lifetime)
        print(f"k ratio: {ratio}")
    else:
        ratio = lifetime / excitation_time
        print(f"T ratio: {ratio}")

    print(f"Total rate k: {k}")
    print(f"Interval time: {interval}")
    return bin_sizes, coherence, ratio, lifetime, excitation_time


def plot_ratio_dependency():
    bin_sizes, coherence, ratio, _, _ = load_ratio_data('k')
    for i in range(len(ratio)):
        current_coherence = coherence[i, :]
        plt.plot(bin_sizes, current_coherence, label=f"ratio {np.round(ratio[i], 2)}")
    plt.legend()
    plt.ylim([-0.01, 1.01])
    plt.show()


def plot_analytical_variance():
    nr_emitters = 1
    bin_size = np.linspace(0.1, 50, 1000)
    excitation_rate = 1
    var = analytical.variance_number_of_emitters(nr_emitters, bin_size, interval=10**5, excitation_rate=excitation_rate, emission_rate=0.01)
    plt.plot(bin_size, np.sqrt(var))
    plt.show()


def main():
    plt.rcParams.update({'font.size': 14, 'figure.figsize': (7, 5.5)})  # 'figure.figsize': (7, 4.8)
    param = "meastime0_2"


if __name__ == "__main__":
    main()
