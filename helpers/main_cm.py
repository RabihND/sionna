#############################
### Setup the environment ###
#############################

import math
import os
import pymongo
gpu_num = "" # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define the resolution of the renderings
resolution = [1280,720] # increase for higher quality of renderings

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

tf.random.set_seed(1) # Set global random seed for reproducibility

# Import the necessary libraries
import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt
import numpy as np
import time

# Import Sionna RT components
from sionna.constants import PI
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera ,LambertianPattern, DirectivePattern, BackscatteringPattern, Antenna, compute_gain,visualize, scene
from sionna.rt.previewer import InteractiveDisplay
# For link-level simulations
from sionna.channel import cir_to_ofdm_channel,cir_to_time_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement
from helpers.functions import calculate_rsrp

################################
### Start of the main script ###
################################

# Load the scene
scene = load_scene("data/office_scene/humanitas.xml")

# Print the objects in the scene
print("Objects in the scene:\n"+"-"*30)
for i, obj in enumerate(scene.objects.values()):
    # print(f"- Object{obj.name}: {obj.radio_material.name}")
    print("Object '{:^20}'  \u279F  Material='{:^20}'".format(obj.name, obj.radio_material.name))
print("-"*30)


# Define the transmitter and receiver arrays
scene.tx_array =PlanarArray(num_rows=1,
                            num_cols=1,
                            vertical_spacing=0.5,
                            horizontal_spacing=0.5,
                            pattern="dipole",
                            polarization="VH")
scene.rx_array = PlanarArray(num_rows=1,
                            num_cols=1,
                            vertical_spacing=0.5,
                            horizontal_spacing=0.5,
                            pattern="dipole",
                            polarization="cross")

# show the radiation pattern of the transmitter and receiver antennas
show_antennas_pattern = False

if show_antennas_pattern:
    # Transmitting antenna
    scene.tx_array.show() # Show the antenna pattern
    # Show the radiation pattern of the transmitter (vertical, horizontal, 3D)
    fig_v, fig_h, fig_3d =visualize(scene.tx_array.antenna.patterns[1])

    # Receiving antenna
    scene.rx_array.show() # Show the antenna pattern
    # Show the radiation pattern of the receiver (vertical, horizontal, 3D)
    fig_v, fig_h, fig_3d =visualize(scene.rx_array.antenna.patterns[1])


# plt.show()
# exit()
# Define the transmitter and receiver devices
# tx_1 = Transmitter("tx", position=[4,2,9])
tx_1 = Transmitter("tx", position=[10.8, 11.76, 2.1]) #vivaldi antenna
# tx_2 = Transmitter("tx2", position=[4,-2,9])

# rx = Receiver("rx", position=[-4,0,8])
rx = Receiver("rx_1", position=[8.2,2.5,1.2])
tx_1.look_at(rx)
# tx_2.look_at(rx) # Point the transmitter towards the receiver (orientation of the transmitter)
# Add the transmitter and receiver to the scene
scene.add(tx_1)
# scene.add(tx_2)
scene.add(rx)

#
num_tx = len(list(scene.transmitters.values()))
num_rx = len(list(scene.receivers.values()))
num_tx_ant = scene.tx_array.array_size
num_rx_ant = scene.rx_array.array_size

print(f"Number of transmitters: {num_tx}")
print(f"Number of receivers: {num_rx}")
print(f"Number of transmitter antennas: {num_tx_ant}")
print(f"Number of receiver antennas: {num_rx_ant}")
# Variables for the coverage map
# RT parameters
los = True
reflection = True
diffraction = True
edge_diffraction = True
scattering = True
# Coverage map parameters
max_depth = 5              # Maximum number of reflections default=32
num_samples=0.1e6            # Number of rays default=1e6
cm_cell_size = (0.2,0.2)    # cell size in meters i.e.(0.5m x 0.5m) (affect the resolution of the coverage map)
show_coverage_map = True
# Scene Parameters
scene.frequency = 3.31968e9 # scene frequency default=3.5e9
Ptx = 0                     # Transmit power in dBm
num_rb = 106                # Number of resource blocks 
bandwidth_mhz = 40          # Bandwidth in MHz
show_scene = False

##############################
## Compute the coverage map ##
##############################
start_time = time.time()
print("{:^10} Computing the coverage map...".format("WAIT"))
cm = scene.coverage_map(max_depth=max_depth,
                        diffraction=diffraction,
                        reflection=reflection,
                        scattering=scattering,
                        edge_diffraction=edge_diffraction,
                        los=los,
                        num_samples=num_samples,
                        cm_cell_size=cm_cell_size)
print("{:^10} Coverage Map computed in {:.2f} seconds".format("DONE", time.time()-start_time))

# Show the coverage map
if show_coverage_map:
    for i in range(num_tx):
        cm_tx = cm.show(tx=i,show_tx=True, show_rx=True)                 

#
# show all the figures
plt.show()

print("{:^10} Rendering the scene...".format("WAIT"))
# scene.render_to_file(camera="scene-cam-0", filename="data/scene.png", resolution=[1280,720],num_samples=1000,show_devices=True,paths=paths,coverage_map=cm)
fig_scene = scene.render(camera="scene-cam-1", resolution=[1280,720],num_samples=2048,show_devices=True,coverage_map=cm)
print("{:^10} Scene rendered".format("DONE"))

# show all the figures
plt.show()


