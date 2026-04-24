import hls4ml
import warnings
import os
warnings.filterwarnings('ignore')

# 1. Verify existence of ONNX export
model_path = 'micro_cnn_hls4ml.onnx'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"{model_path} not found! Please run train_micro_hls4ml.py first.")

print("--- 1. Loading Micro-CNN ONNX Profile ---")
# Parse the standard ONNX block
config = hls4ml.utils.config_from_onnx_model(model_path)

# 2. Inject FPGA-friendly Synthesis Constraints
print("--- 2. Applying Ultra-Low Resource HLS Config ---")
# 'Resource' Strategy forces the Vivado compiler to heavily reuse DSP blocks
# This is explicitly what stops the 'SCSYNTH Failed' out-of-memory errors.
config['Model']['Strategy'] = 'Resource'

# ReuseFactor forces massive operational folding over time on the DSP slices.
# 1024 or 4096 ensures an ultra-low DSP footprint, fitting any Zynq easily at the cost of slight latency.
config['Model']['ReuseFactor'] = 1024 

# Apply Post-Training Quantization (PTQ) directly in hls4ml, bypassing Brevitas!
config['Model']['Precision'] = 'ap_fixed<16,6>' 

# You can inspect the config locally
print("HLS Configuration Constraints:", config)

# 3. Target ZCU104 and compile
print("--- 3. Converting to HLS C++ ---")
# Target Part: Zynq UltraScale+ ZCU104 board
hls_model = hls4ml.converters.convert_from_onnx_model(
    model_path,
    hls_config=config,
    output_dir='hls4ml_zcu104_cnn',
    part='xczu7ev-ffvc1156-2-e' 
)

print("--- 4. Ready for Vivado Building ---")
# hls_model.compile() # Compiles the C block natively
# hls_model.build(csim=False, synth=True) # Runs Vivado synthesizer to gen IP

print("HLS Model configured successfully. Uncomment hls_model.build() to generate the RTL IP!")
