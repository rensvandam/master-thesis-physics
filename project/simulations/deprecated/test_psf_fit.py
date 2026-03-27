from project.model.helper_functions import gaussian_2d, get_psf_params
from project.simulations.run_scanning_experiment import run_scanning_experiment
from project.model.localization import localize, extract_psf, extract_psf_improved


# test extract_psf function
if __name__ == "__main__":
    # Define parameters
    # extract_psf(laser_power = 200e3,
    #             pixel_size = 0.050,
    #             dwell_time = 0.1,
    #             dead_time = 50,
    #             plot = True)
    extract_psf_improved(laser_power = 200e3,
                         pixel_size = 0.050,
                         dwell_time = 0.1,
                         dead_time = 0,
                         n_simulations=1,
                         plot = True,)
