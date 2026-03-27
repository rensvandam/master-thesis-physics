import numpy as np


def expected_coherence(nr_emitters, bin_size, excitation_rate, emission_rate):
    """
    Calculates the coherence value that corresponds to this number of emitters.

    Parameters
    ----------
    nr_emitters : int
        The number of emitters.
    bin_size : float
        The length of one bin. (ns)
    excitation_rate : float
        The excitation rate of one fluorophore. (ns^-1)
    emission_rate : float
        The emission rate of one fluorophore. (ns^-1)

    Returns
    -------
    float
        The value of the second-order quantum coherence that is measured for a certain bin size.
    """
    k = excitation_rate + emission_rate
    return 1 - (1 / nr_emitters) * discrete_deviation_coherence(bin_size, k)


def expected_number_of_emitters(measured_coherence, bin_size, excitation_rate, emission_rate):
    """
    Calculates the number of emitters that corresponds to this coherence, measured with a certain bin size.

    Parameters
    ----------
    measured_coherence : float
        The value of the second-order quantum coherence at lag 0.
    bin_size : float
        The length of one bin. (ns)
    excitation_rate : float
        The excitation rate of one fluorophore. (ns^-1)
    emission_rate : float
        The emission rate of one fluorophore. (ns^-1)

    Returns
    -------
    float
        The number of emitters associated with this coherence.
    """
    k = excitation_rate + emission_rate
    return discrete_deviation_coherence(bin_size, k) / (1 - measured_coherence)


def discrete_deviation_coherence(bin_size, k):
    """
    Calculates the deviation factor that is added to the coherence due to the discretization of the formula.

    Parameters
    ----------
    bin_size : float
        The length of one bin. (ns)
    k : float
        The sum of the excitation and the emission rate of a fluorophore. (ns^-1)

    Returns
    -------
    float
        The extra factor that influences the coherence through g^(2)[0]=1-(1/n)*factor.
    """
    return (2 / (k * bin_size)) * (1 - (1 - np.exp(-k * bin_size)) / (k * bin_size))


def first_term_variance(nr_emitters, bin_size, nr_bins, k):
    rate_bin = (k / 2) * bin_size

    def per_bin(infinite_difference, difference):
        e_n = rate_bin
        e_n2 = rate_bin ** 2 + rate_bin
        e_n3 = rate_bin ** 3 + 3 * rate_bin ** 2 + rate_bin
        e_n4 = rate_bin ** 4 + 6 * rate_bin ** 3 + 7 * rate_bin ** 2 + rate_bin

        p = (1 + np.exp(-2 * rate_bin)) / 2
        p_bar = (1 - np.exp(-2 * rate_bin)) / 2
        e_u = p_bar

        e_n_u = p_bar * rate_bin * (1 + np.exp(-2 * rate_bin)) / (1 - np.exp(-2 * rate_bin))
        e_n2_u2 = p_bar * (rate_bin ** 2 + rate_bin * (1 + np.exp(-2 * rate_bin)) / (1 - np.exp(-2 * rate_bin)))

        e_x = e_n / 2
        e_x2 = (e_n2 + e_u) / 4
        if not infinite_difference and difference == 0:
            e_x_x = e_x2
            e_x2_x = e_n3 / 8 + 3 * e_n_u / 8
            e_x2_x2 = e_n4 / 16 + 3 * e_n2_u2 / 8 + e_u / 16
        else:
            if infinite_difference:
                e_power = 0
            else:
                e_power = np.exp(-2 * rate_bin * np.abs(difference))
            e_x_x = (e_n ** 2) / 4 + (e_u ** 2) * e_power / 4
            e_x2_x = e_n * (e_n2 + e_u) / 8 + e_n_u * e_u * e_power / 4
            e_x2_x2 = (e_n2 ** 2 + e_u ** 2) / 16 + e_n2 * e_u / 8 + (e_n_u ** 2) * e_power / 16

        val = ((e_x ** 4) * nr_emitters * (nr_emitters - 1) * (nr_emitters - 2) * (nr_emitters - 3)
               + e_x2 * (e_x ** 2) * 2 * nr_emitters * (nr_emitters - 1) * (nr_emitters - 2)
               + e_x_x * (e_x ** 2) * 4 * nr_emitters * (nr_emitters - 1) * (nr_emitters - 2)
               + (e_x2 ** 2) * nr_emitters * (nr_emitters - 1)
               + (e_x_x ** 2) * 2 * nr_emitters * (nr_emitters - 1)
               + e_x2_x * e_x * 4 * nr_emitters * (nr_emitters - 1)
               + e_x2_x2 * nr_emitters
               )
        return val

    diag = nr_bins * per_bin(infinite_difference=False, difference=0)
    max_off_diagonal = 100
    off_diags = 0
    for j in range(1, max_off_diagonal + 1):
        off_diags += 2 * (nr_bins - j) * per_bin(infinite_difference=False, difference=j)
    remainder = (((nr_bins - max_off_diagonal) ** 2 - (nr_bins - max_off_diagonal))
                 * per_bin(infinite_difference=True, difference=max_off_diagonal))
    expected_squared_sum_c = diag + off_diags + remainder
    return expected_squared_sum_c * (1 / nr_bins ** 2) * (2 / (rate_bin * nr_emitters)) ** 4


def second_term_variance(nr_emitters, bin_size, k):
    return (1 + 2 / (nr_emitters * k * bin_size) + 2 * (1 - np.exp(-k * bin_size)) / (
                nr_emitters * (k * bin_size) ** 2)) ** 2


def variance_coherence(nr_emitters, bin_size, interval, excitation_rate, emission_rate):
    k = excitation_rate + emission_rate
    nr_bins = np.array(interval / bin_size).astype(np.int64)
    return (first_term_variance(nr_emitters, bin_size, nr_bins, k)
            - second_term_variance(nr_emitters, bin_size, k))


def variance_number_of_emitters(nr_emitters, bin_size, interval, excitation_rate, emission_rate):
    variance = variance_coherence(nr_emitters, bin_size, interval, excitation_rate, emission_rate)
    f = discrete_deviation_coherence(bin_size, excitation_rate + emission_rate)
    return nr_emitters ** 4 * variance / f ** 2
