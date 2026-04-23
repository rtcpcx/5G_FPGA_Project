import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from sionna.ofdm import ResourceGrid, ResourceGridMapper, RemoveNulledSubcarriers
from sionna.mapping import Mapper
from sionna.channel.tr38901 import TDL
from sionna.channel import OFDMChannel
from sionna.utils import BinarySource

def generate_big_dataset_chunked(total_samples=50000, chunk_size=5000):
    print(f"--- Generating {total_samples} Slots in Chunks of {chunk_size} to save RAM ---")
    
    rg = ResourceGrid(num_ofdm_symbols=14, fft_size=64, subcarrier_spacing=30e3, 
                      num_tx=1, num_streams_per_tx=1, cyclic_prefix_length=4,
                      num_guard_carriers=[8, 8], dc_null=False,
                      pilot_pattern="kronecker", pilot_ofdm_symbol_indices=[2, 11])
    
    binary_source = BinarySource()
    mapper = Mapper("qam", num_bits_per_symbol=4)
    rg_mapper = ResourceGridMapper(rg)
    tdl = TDL("A", delay_spread=100e-9, carrier_frequency=3.5e9)
    channel = OFDMChannel(tdl, rg, add_awgn=True, normalize_channel=True, return_channel=False)
    remove_nulls = RemoveNulledSubcarriers(rg)

    all_y_stacked = []
    all_bits = []

    for i in range(total_samples // chunk_size):
        print(f"Processing chunk {i+1}/{(total_samples // chunk_size)}...")
        
        snr_db = tf.random.uniform([chunk_size, 1, 1, 1], 5.0, 25.0)
        no = 10.0 ** (-snr_db / 10.0)
        num_bits = int(rg.num_data_symbols * 4) 
        
        b = binary_source([chunk_size, 1, 1, num_bits])
        x_qam = mapper(b)
        x_grid = rg_mapper(x_qam)
        y_noisy = channel([x_grid, no])
        y_active = remove_nulls(y_noisy)
        
        y_siso = tf.squeeze(y_active, axis=[1, 2])
        y_siso = tf.transpose(y_siso, perm=[0, 2, 1]) 
        y_stacked = tf.stack([tf.math.real(y_siso), tf.math.imag(y_siso)], axis=1) 
        bits_squeezed = tf.squeeze(b, axis=[1, 2])
        
        all_y_stacked.append(y_stacked.numpy())
        all_bits.append(bits_squeezed.numpy())
        
    print("--- Stitching chunks together... ---")
    final_y = np.concatenate(all_y_stacked, axis=0)
    final_bits = np.concatenate(all_bits, axis=0)
    
    np.save('siso_fpga_input_grid_50k.npy', final_y)
    np.save('siso_fpga_target_bits_50k.npy', final_bits)
    print("--- Success: siso_fpga_input_grid_50k.npy saved safely! ---")

if __name__ == "__main__":
    generate_big_dataset_chunked(50000, 5000)
