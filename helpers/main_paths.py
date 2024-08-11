import os
gpu_num = "0" # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# for cpu , set memory growth to tru



# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

tf.random.set_seed(1) # Set global random seed for reproducibility

# Import the necessary libraries
import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt
import numpy as np
import time
import math

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
# scene = load_scene("data/scene2/floor_wall.xml")
scene = load_scene("data/office_scene/humanitas.xml")

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
                            pattern="dipole",
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
    # scene.rx_array.show() # Show the antenna pattern
    # # Show the radiation pattern of the receiver (vertical, horizontal, 3D)
    # fig_v, fig_h, fig_3d =visualize(scene.rx_array.antenna.patterns[1])


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

# RT Effects (Diffraction, Reflection, Scattering, LOS)
diffraction = True
edge_diffraction = False
reflection = True
scattering = False
los = True
num_samples=0.0001e6 # Number of rays default=1e6
max_depth = 2
cm_cell_size = (0.1,0.1) # cell size in meters i.e.(0.5m x 0.5m) (affect the resolution of the coverage map)
# Scene frequency
scene.frequency = 3.31968e9 # default=3.5e9
# Define the resolution of the renderings
resolution = [1280,720] # increase for higher quality of renderings

# Compute the paths
start_time = time.time()
paths = scene.compute_paths(max_depth=max_depth, los=los, reflection=reflection, diffraction=diffraction,edge_diffraction=edge_diffraction, scattering=scattering, num_samples=num_samples)
print("{:^10} Paths computed in {:.2f} seconds".format("DONE", time.time()-start_time))


paths.export("paths.obj")

# Show the coordinates of the starting points of all rays.
# These coincide with the location of the transmitters.
print("Source coordinates: ", paths.sources.numpy())
print("Transmitter coordinates: ", list(scene.transmitters.values())[0].position.numpy())

# Show the coordinates of the endpoints of all rays.
# These coincide with the location of the receivers.
print("Target coordinates: ",paths.targets.numpy())
print("Receiver coordinates: ",list(scene.receivers.values())[0].position.numpy())

# Show the types of all paths:
# 0 - LoS, 1 - Reflected, 2 - Diffracted, 3 - Scattered
# Note that Diffraction and scattering are turned off by default.
print("Path types: ", paths.types.numpy())
print("Number of paths: ", paths.types.shape[1])

# We can now access for every path the channel coefficient, the propagation delay,
# as well as the angles of departure and arrival, respectively (zenith and azimuth).

# # Let us inspect a specific path in detail 
# path_idx = 0# Try out other values in the range [0, 13]

# # # For a detailed overview of the dimensions of all properties, have a look at the API documentation
path_idx = [0,1]
for i in path_idx:
    print(f"\n--- Detailed results for path {i} ---")
    print(f"Channel coefficient (a): {paths.a[0,0,0,0,0,i, 0].numpy()}")
    print(f"Propagation delay (tau): {paths.tau[0,0,0,i].numpy()*1e9:.5f} ns")
    print(f"Zenith angle of departure: {paths.theta_t[0,0,0,i]:.4f} rad")
    print(f"Azimuth angle of departure: {paths.phi_t[0,0,0,i]:.4f} rad")
    print(f"Zenith angle of arrival: {paths.theta_r[0,0,0,i]:.4f} rad")
    print(f"Azimuth angle of arrival: {paths.phi_r[0,0,0,i]:.4f} rad")
    print(f"Path type: {paths.types[0,i].numpy()}")



Pr_db = 10*np.log10(tf.reduce_sum(tf.abs(paths.a)**2))
# Compute the channel impulse response (CIR) 
# RSRP computation
# bandwidth=40e6 # bandwidth of the receiver (= sampling frequency)
# # Compute the baseband-equivalent CIR
# h_time = np.squeeze(cir_to_time_channel(bandwidth,*paths.cir(), 0, 100, normalize=True))

# # Compute the Received Power
# Pr = np.sum(np.abs(h_time) ** 2) # based on discussions/229
# Convert the received power to dB
# Pr_db = 10 * np.log10(Pr)
# Add the transmit power (in dB)
# Pt_db = 0  # Assuming a transmit power of 20 dBm
# Pr_db = Pr_db + Pt_db
print(f"Received Power: {Pr_db:.2f} dB")

# RSRP computation
# rsrp = Pr_db -10*np.log10(12*106)
# print(f"RSRP: {rsrp:.2f} dBm")

def calculate_rsrp(rssi_dbm, num_rb, bandwidth_mhz):
    """
    Calculate the Reference Signal Received Power (RSRP) based on 3GPP standards.

    Parameters:
    rssi_dbm (float): Total received power in dBm.
    num_rb (int): Number of resource blocks across which the RSSI is measured.
    bandwidth_mhz (float): Bandwidth of the measurement in MHz.

    Returns:
    float: Calculated RSRP in dBm.
    """
    # Convert RSSI from dBm to Watts
    rssi_watts = 10 ** (rssi_dbm / 10)
    
    # Calculate RSRP using the formula
    rsrp_watts = rssi_watts / (num_rb * bandwidth_mhz)
    
    # Convert RSRP back to dBm
    rsrp_dbm = 10 * math.log10(rsrp_watts)
    
    # Round to the nearest whole number to match 1 dB resolution
    rsrp_dbm_rounded = round(rsrp_dbm, 1)
    
    return rsrp_dbm_rounded

#
rssi_dbm = Pr_db
num_rb = 106
bandwidth_mhz = 40
# rsrp = calculate_rsrp(rssi_dbm, num_rb, bandwidth_mhz)
# print(f"RSRP: {rsrp:.2f} dBm")


# Render the scene
print("{:^10} Rendering the scene...".format("WAIT"))
# scene.render_to_file(camera="scene-cam-0", filename="data/scene.png", resolution=[1280,720],num_samples=1000,show_devices=True,paths=paths,coverage_map=cm)
fig_scene = scene.render(camera="scene-cam-1", resolution=[1280,720],num_samples=4096,show_devices=True,paths=paths)
print("{:^10} Scene rendered".format("DONE"))
# Show all the figures
plt.show()