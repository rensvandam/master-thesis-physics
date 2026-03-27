import numpy as np
import matplotlib.pyplot as plt

from project.model.coherence_from_data import auto_coherence, show_coherence, coherence
from project.model.detection import show_photons, Spad23, Spad512, merge_photons
from project.model.sample import Alexa647
from project.model.setup import Setup, WidefieldSetup, ScanningSetup

# Basic example of widefield simulation using the Spad512 sensor

def basic_simulation_spad512_widefield():

    sensor = Spad512()
    setup = WidefieldSetup(sensor)

    emitters = []
    emitter_locations = [(-3, -3), (3,-3), (3, 0)] # in um
    for i, (x, y) in enumerate(emitter_locations):
        emitter = Alexa647(x=x, y=y)
        emitter.generate_photons(laser_power=330 * 10 ** 3, time_interval=10**5, widefield=True, seed=6 * i)
        emitters.append(emitter)
    photons = merge_photons(emitters)
    
    setup.translate_photons(photons)



# def basic_simulation_spad23_scanning():

#     sensor = Spad23()
#     setup = ScanningSetup(sensor, scan_speed = 1, dwell_time = ,)

#     emitters = []
#     emitter_locations = [(-3, -3), (3,-3), (3, 0)] # in um
#     for i, (x, y) in enumerate(emitter_locations):
#         emitter = Alexa647(x=x, y=y)
#         emitter.generate_photons(laser_power=330 * 10 ** 3, time_interval=10**5, widefield=True, seed=6 * i)
#         emitters.append(emitter)