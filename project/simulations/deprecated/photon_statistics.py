import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import poisson

from project.model.sample import Alexa647


def check_photon_distribution():
    e = Alexa647(x=0, y=0)
    laser = 333 * 10 ** 3  # W / cm2
    time_interval = 10 ** 4  # ns

    nr_reps = 1
    nr_photons = np.zeros(nr_reps)
    for i in range(nr_reps):
        if i % 1000 == 0:
            print(i)
        photons = e.generate_photons(laser, time_interval, seed=i)
        nr_photons[i] = np.shape(photons)[0]
        # plt.hist(photons[:, 2], bins=100)
        # plt.show()

    average = np.mean(nr_photons)
    print(f"Average nr of photons:\t{average}"
          f"\nVariance:\t\t\t\t{np.var(nr_photons)}"
          f"\nSqrt of average:\t\t{np.sqrt(average)}"
          f"\nStandard deviation:\t\t{np.std(nr_photons)}")
    plt.hist(nr_photons, bins=20, density=True)
    x = np.arange(max(np.min(nr_photons) - 5, 0), np.max(nr_photons) + 5)
    plt.plot(x, poisson.pmf(x, average), linestyle='-', marker='', color='red', label='Poisson')
    plt.subplots_adjust(bottom=0.2)
    # plt.title(fr"$\Delta t$ = {time_interval / 10 ** 3} $\mu$s, $P_l$ = {laser / 10 ** 3} $kW/cm^2$, $\eta$ = {eta}%")
    plt.xlabel("Number of photons\n"
               + r"$\sqrt{\bar{n}}$ = " + f"{np.sqrt(average):#.3f}"
               + r", $\Delta n$ = " + f"{np.std(nr_photons):#.3f}"
               )
    plt.ylabel("Probability")
    plt.legend()
    # plt.savefig(fname=f'figures/meeting_23_11_03/poisson_p_{laser/10**3}kwcm_t_{time_interval / 10 ** 3}mus_eta_{eta}.png')
    plt.show()