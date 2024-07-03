import os
gpu_num = 0 # Use "" to use the CPU
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
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera ,LambertianPattern, DirectivePattern, BackscatteringPattern, Antenna, compute_gain,visualize, scene
from sionna.rt.previewer import InteractiveDisplay
# For link-level simulations
from sionna.channel import cir_to_ofdm_channel,cir_to_time_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement
####################################################################################################
############ Start of the main script ############################################################
####################################################################################################

# Load the scene
# scene = load_scene(scene.floor_wall)
scene = load_scene("data/scene2/floor_wall.xml")

# Print the objects in the scene
print("Objects in the scene:")
for i, obj in enumerate(scene.objects.values()):
    # print(f"- Object{obj.name}: {obj.radio_material.name}")
    print("Object '{:^10}': Material='{:^20}'".format(obj.name, obj.radio_material.name))


# Define the transmitter and receiver arrays
scene.tx_array =PlanarArray(num_rows=1,
                            num_cols=1,
                            vertical_spacing=0.5,
                            horizontal_spacing=0.5,
                            pattern="tr38901",
                            polarization="VH")
scene.rx_array = PlanarArray(num_rows=1,
                            num_cols=1,
                            vertical_spacing=0.5,
                            horizontal_spacing=0.5,
                            pattern="dipole",
                            polarization="cross")

# show the radiation pattern of the transmitter
# scene.tx_array.show().savefig("data/antenna/tx_array.png")

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


# Define the transmitter and receiver devices
tx_1 = Transmitter("tx_1", position=[4,2,9])
tx_2 = Transmitter("tx_2", position=[4,-2,9])

rx_1 = Receiver("rx_1", position=[-4,0,1.5])
rx_2 = Receiver("rx_2", position=[-4,-3,1.5])
tx_1.look_at(rx_1)
tx_2.look_at(rx_1) # Point the transmitter towards the receiver (orientation of the transmitter)
# Add the transmitter and receiver to the scene
scene.add(tx_1)
scene.add(tx_2)
scene.add(rx_1)
scene.add(rx_2)

#
num_tx = len(list(scene.transmitters.values()))
num_rx = len(list(scene.receivers.values()))
num_tx_ant = scene.tx_array.array_size
num_rx_ant = scene.rx_array.array_size

print(f"Number of transmitters: {num_tx}")
print(f"Number of receivers: {num_rx}")
print(f"Number of transmitter antennas: {num_tx_ant}")
print(f"Number of receiver antennas: {num_rx_ant}")

# RT Effects (Diffraction, Reflection, Scattering, LOS)
diffraction = True
edge_diffraction = True
reflection = True
scattering = True
los = True
num_samples=2.5e6 # Number of rays default=1e6
max_depth = 16
cm_cell_size = (0.1,0.1) # cell size in meters i.e.(0.5m x 0.5m) (affect the resolution of the coverage map)
# Scene frequency
scene.frequency = 3.6e9 # default=3.5e9

# # Compute the coverage map
cm = scene.coverage_map(max_depth=max_depth,
                        diffraction=diffraction,
                        reflection=reflection,
                        scattering=scattering,
                        edge_diffraction=edge_diffraction,
                        los=los,
                        num_samples=num_samples,
                        cm_cell_size=cm_cell_size) # cell size in meters i.e.(0.5m x 0.5m) (affect the resolution of the coverage map)
print("{:^10} Coverage Map computed".format("DONE"))
# Show the coverage map
fig_cm = cm.show(tx=0,show_tx=True, show_rx=True)
fig_cm2 = cm.show(tx=1,show_tx=True, show_rx=True)                 


# Get the coverage map data
data = cm.as_tensor()
# based on the cm_cell_size get the cell index [x,y] of the receiver position
rx_pos = scene.receivers["rx_1"].position
rx_2_pos = scene.receivers["rx_2"].position
rx_value = cm.get_value(rx_pos)

# clear the cm
# del cm
# en db
rx_value_db = 10*np.log10(rx_value)
print(f"Coverage map value in dB at the receiver position: {rx_value_db}")

# Calculate the received power at the receiver from all transmitters
rx_tx1_pathloss = rx_value_db[0]
rx_tx2_pathloss = rx_value_db[1]

# Transmit power
tx1_power = 20 # dBm
tx2_power = 25 # dBm

# Calculate the received power at the receiver from all transmitters
rx_power1 = tx1_power + rx_tx1_pathloss
rx_power2 = tx2_power + rx_tx2_pathloss

print(f"Received power at the receiver from tx1: {rx_power1:.2f} dBm")
print(f"Received power at the receiver from tx2: {rx_power2:.2f} dBm")



Pt = 20 # dBm
rsrp = Pt -10*np.log10(12*106) + rx_tx1_pathloss
print(f"RSRP: {rsrp:.2f} dBm")

plt.show()