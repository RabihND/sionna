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



import matplotlib.pyplot as plt
import pymongo
# import tensorflow as tf
import numpy as np

from utils import get_value, from_mongo, show, show_sinr


# Connect to the MongoDB server
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["coverage_map"]
collection = db["cm_data"]



# Retrieve the coverage map data
coverage_map_data = from_mongo(collection, 'cm_data')


# Get the value at a specific position
rx_pos = coverage_map_data["rx_positions"]
data = coverage_map_data["data"]
center = coverage_map_data["center"]
orientation = coverage_map_data["orientation"]
size = coverage_map_data["size"]
cell_size = coverage_map_data["cell_size"]
rdtype = coverage_map_data["rdtype"]
tx_pos= coverage_map_data["tx_positions"]
tx_ori = coverage_map_data["tx_orientations"]
tx_num = coverage_map_data["tx_number"]
transmitters = coverage_map_data["transmitters"]

# Get the value at the receiver position
rx_value = get_value(rx_pos, data)

print(f"Coverage map value at the receiver position{rx_pos}: {rx_value.numpy()}")
# Show the coverage map for tx=0

# show(data, tx=2, show_tx_orientation=True, show_rx=True, rx_positions=rx_pos, tx_positions=tx_pos, tx_orientations=tx_ori,num_tx=tx_num,transmitters=transmitters)
# show_plotly(value, tx=0, show_tx_orientation=True, show_rx=True, rx_positions=rx_pos, tx_positions=tx_pos, tx_orientations=tx_ori,num_tx=tx_num,transmitters=transmitters).show()
# plt.show()


# convert to dB
value_db = 10 * np.log10(rx_value) / np.log10(10.0)

print(f"Coverage map value in dB at the receiver position: {value_db}")

# calculate SNR = P_signal / P_noise
# based on https://www.mathworks.com/help/5g/ug/include-path-loss-in-nr-link-level-simulations.html
Nfft = 2048
Nsize_gride = 106 # prb at scs=30khz and bw=40mhz
Ptx_dBm = 0 # dBm (to db = dbm -30)
fftOccupancy  =  Nfft / (Nsize_gride*12)
from scipy.constants import Boltzmann as k
NF = 6 # dB noise figure
NF = 10**(NF/10) # linear
T = 290 # antenna temperature
Teq = T + 290*(NF-1) # equivalent noise temperature
samplingRate = 61.44e6 # 61.44 MHz OAI 5g config
N0 = np.sqrt(k*samplingRate*Teq/2)
snr_db = (Ptx_dBm -30 )- (-value_db) -10*np.log10(fftOccupancy)- 10*np.log10(2*N0**2)

print(f"SNR at the receiver position: {snr_db}")


# SNIR
total_interference = []
for tx in range(tx_num):
    interference =0
    for tx2 in range(tx_num):
        if tx2 != tx:
            interference += value_db[tx2]
    total_interference.append(interference)

    print(f"Total interference for tx {tx}: {interference}")

snir = value_db - total_interference
print(f"SNIR at the receiver position: {snir}")

show_sinr(data, tx=1, show_tx_orientation=True, show_rx=True, rx_positions=rx_pos, tx_positions=tx_pos, tx_orientations=tx_ori,num_tx=tx_num,transmitters=transmitters)

plt.show()