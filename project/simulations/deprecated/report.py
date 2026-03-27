import numpy as np
from matplotlib import pyplot as plt

from project.model.sample import Alexa647, Sensor
from project.model import coherence_from_data


def psf():
    def gaussian(x, y, x0, y0, sigma):
        term1 = (x - x0) ** 2 / (2 * sigma ** 2)
        term2 = (y - y0) ** 2 / (2 * sigma ** 2)
        return np.exp(-1 * (term1 + term2))

    nr_pixels = 15
    pixel_size = 665 / 4
    s = Sensor(nr_pixels, pixel_size=pixel_size)
    middle = 7.5 * pixel_size
    positions = middle + np.array([
        [-319, 223],  # 1
        [156, -515],  # 1
        [410, 264],  # 1  p3
        [150, 600],  # 2
        [-298, -474],  # 2
        [0, 0],  # 2  p1
        [-119, 567],  # 3
        [-200, 30],  # 3  p1
        [-132, -237],
        [271, 89],  # p3
        [88, -278],
        [0, 364],  # p3
    ])
    points = np.linspace(0, pixel_size * nr_pixels, 1000)
    X, Y = np.meshgrid(points, points, indexing='ij')
    total_gaus = np.zeros(np.shape(X))
    nr_emitters = len(positions)
    emitters = []
    for i in range(nr_emitters):
        e = Alexa647(x=positions[i, 0], y=positions[i, 1])
        # e.generate_photons(laser_power=300 * 10 ** 3, time_interval=10 ** 5, seed=0)
        emitters.append(e)
        total_gaus += gaussian(X, Y, e.x, e.y, 10)

    # s.measure_photons(emitters)
    # s.show_measured_photons()
    # plt.show()
    # s.clear_sensor()

    plt.pcolormesh(X, Y, total_gaus, vmin=0, vmax=np.amax(total_gaus))
    # for i in range(nr_emitters):
    #     plt.scatter(positions[:,0], positions[:,1], color='red')
    plt.colorbar()
    # plt.savefig('../../report/introduction/puppet_problem3.png')
    plt.show()


def coherence_method():
    def analytical_single_emitter(tau, n, k_rate):
        return 1 - (1 / n) * np.exp(-k_rate * tau)

    # nr_pixels = 16
    # pixel_size = 665 / 4
    # s = Sensor(nr_pixels, pixel_size=pixel_size)
    #
    # e = Alexa647(x=pixel_size*nr_pixels/2, y=pixel_size*nr_pixels/2)
    # e.generate_photons(laser_power=300 * 10 ** 3, time_interval=10 ** 5, seed=200, statistics="poisson")
    # s.measure_photons([e])
    # c, lag = coherence.auto_coherence(s.data[:, s.T], interval=10**5, bin_size=0.01, nr_steps=250)
    ex_rate = 1
    em_rate = 1
    k = ex_rate + em_rate
    lag = np.linspace(0, 2.5, 100)
    c = analytical_single_emitter(lag, 1, k)
    plt.xticks([0, 1, 2, 3, 4, 5])
    coherence_from_data.show_coherence(lag * k, c,
                                       save_as='../../defense/coherence_sub_poisson_analytical_k_1emitters.svg')


def plot_interval_time(rate1, rate2, label):
    def pdf(time, rate1, rate2):
        if rate1 == rate2:
            return rate1 ** 2 * time * np.exp(-rate1 * time)
        return (rate1 * rate2 / (rate1 - rate2)) * (np.exp(-rate2 * time) - np.exp(-rate1 * time))

    x = np.linspace(0, 8, 200)
    plt.plot(x, pdf(x, rate1, rate2), label=label, color='black')
    plt.xlabel(r'$T_i = t_i - t_{i-1}$ [ns]')
    # plt.xticks([])
    plt.ylabel('Probability')
    plt.ylim(0, 0.4)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4])
    plt.tight_layout()
    plt.savefig(f"../figures/report/method/interval_distribution_gamma_2_1_small.svg", transparent=True)
    plt.show()


def test_correlation_photon_train():
    # Check how much you need to cut off in order to avoid correlation due to the startup phase
    t_ex = 1
    t_em = 1
    t_exp = t_em + t_ex
    p = 20
    for i in range(10):
        photons = np.cumsum(np.random.exponential(t_em, p) + np.random.exponential(t_ex, p))
        plt.plot(photons, np.zeros(p) + i, linestyle='', marker='.')
    plt.axvline(t_exp * 10, color='grey')
    plt.show()


def generate_poisson_sequence():
    rate = 0.5
    n = 40
    np.random.seed(4)
    photons = np.cumsum(np.random.exponential(rate, n))
    plt.plot(photons, np.zeros(n), linestyle='', marker='.', markersize=13)
    # plt.savefig('../figures/report/results/poisson_sequence.svg')
    plt.show()


def main():
    # plt.rcParams.update({'font.size': 14, 'figure.figsize': (7, 5.5)})  # 'figure.figsize': (7, 4.8)
    pass


if __name__ == "__main__":
    main()
