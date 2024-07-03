import math
import pickle
import pymongo

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