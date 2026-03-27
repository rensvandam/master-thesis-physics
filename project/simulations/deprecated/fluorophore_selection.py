import numpy as np
from project.model.sample import expected_excitation_time


def required_laser_power(extinction_coefficient, absorption_wavelength, lifetime):
    """
    The laser power that is required for equal excitation and emission times.
    """
    alpha = expected_excitation_time(extinction_coefficient, absorption_wavelength, 1)
    required = alpha / lifetime
    print(f"Required laser power:\n{np.round(required / 10 ** 3, 0)} kW / cm2")
    return required


def KU_dyes():
    # Source: https://www.ku-dyes.com/ku-dyes-home/ku-dyes-catalogue/
    epsilon = np.array([120000, 132000, 60300, 64600, 92100, 9800, 134600, 16800])
    abs = np.array([451, 471, 483, 500, 510, 530, 542, 560])
    lifetime = np.array([2, 3.6, 2.2, 3.6, 1.9, 24, 4.1, 20])
    print(abs)
    print(lifetime)
    required_laser_power(epsilon, abs, lifetime)


def compare_measurement_times(lifetime, laser, extinction_coef, abs_wavelength):
    # Process A: Alexa647, 0.1 ms, 100% eta, kex=kem=1
    T_A = 2
    meas_A = 0.001

    # Process B: different fluorophore
    eta = 0.10
    kex_B = 1 / (expected_excitation_time(extinction_coef, abs_wavelength, laser))
    kem_B = 1 / lifetime
    k_B = kex_B + kem_B
    T_B = (1 / kex_B + 1 / kem_B) / eta

    # Process C: kex=kem=0.5k_B
    k_C = k_B
    T_C = 1 / (0.5 * k_C) + 1 / (0.5 * k_C)

    meas_C = (T_C / T_A) * meas_A
    meas_B = (T_B / T_C)**2 * meas_C
    return meas_B


def main():
    # KU_dyes()
    x = compare_measurement_times(1, 10*10**3, 240000, 650)
    print(x)


if __name__ == "__main__":
    main()