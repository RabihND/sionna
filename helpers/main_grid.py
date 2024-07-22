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


import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt


from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer

rg = ResourceGrid(num_ofdm_symbols=14,
                  fft_size=12*106,
                  subcarrier_spacing=30e3,
                  num_tx=1,
                  num_streams_per_tx=1,
                  cyclic_prefix_length=288/2,
                  num_guard_carriers=[205,205],
                  dc_null=True,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=[2,11])
rg.show()
rg.pilot_pattern.show()
plt.show()