import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import onnxruntime as ort

print("--- 1. Loading Saved 5G Data ---")
X = np.load('siso_fpga_input_grid.npy').astype(np.float32)
Y = np.load('siso_fpga_target_bits.npy').astype(np.float32)

# Grab exactly one slot for the visualization
X_viz = X[0:1]  # Shape: (1, 2, 48, 14)
Y_viz = Y[0]    # Shape: (2304,)

print("--- 2. Loading ONNX Hardware Model ---")
# This runs the EXACT file you will put on the FPGA
session = ort.InferenceSession('deeprx_siso_8bit_GOLDEN.onnx')
input_name = session.get_inputs()[0].name

print("--- 3. Running Instant Inference ---")
# Run the ONNX model
onnx_outputs = session.run(None, {input_name: X_viz})[0]

# Apply Sigmoid to get bit probabilities, then threshold at 0.5
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
preds_viz = (sigmoid(onnx_outputs[0]) > 0.5).astype(np.float32)
final_acc = np.mean(preds_viz == Y_viz)

print("--- 4. Generating the Heatmap ---")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- Left Plot: Noisy I/Q Scatter ---
real_noisy = X_viz[0, 0, :, :].flatten()
imag_noisy = X_viz[0, 1, :, :].flatten()
ax1.scatter(real_noisy, imag_noisy, alpha=0.6, color='royalblue', edgecolors='black', linewidth=0.5)
ax1.axhline(0, color='black', linewidth=1)
ax1.axvline(0, color='black', linewidth=1)
ax1.set_title("Before AI: 5G Received Signal\n(TDL-A Fading + AWGN Noise)")
ax1.set_xlabel("In-Phase (I)")
ax1.set_ylabel("Quadrature (Q)")
ax1.grid(True, linestyle='--')

# --- Right Plot: Bit Recovery Heatmap ---
correct_mask = (preds_viz == Y_viz)
# Reshape the 2304 bits into a 48x48 square for the heatmap
error_grid = correct_mask.reshape(48, 48)

cmap = ListedColormap(['crimson', 'mediumseagreen'])
cax = ax2.imshow(error_grid, cmap=cmap, aspect='auto')
ax2.set_title(f"After AI: ONNX Hardware Bit Recovery Heatmap\n(Direct Demapping Accuracy: {final_acc:.2%})")
ax2.set_xlabel("Payload Frame Segment")
ax2.set_ylabel("Subcarrier Data Stream")

green_patch = mpatches.Patch(color='mediumseagreen', label='Correctly Recovered Bit')
red_patch = mpatches.Patch(color='crimson', label='Bit Error (Failed Recovery)')
ax2.legend(handles=[green_patch, red_patch], loc='upper right', bbox_to_anchor=(1.0, 1.15))

plt.tight_layout()
plt.savefig('visualization_demo_onnx_50k.png', dpi=300)
print("--- SUCCESS: visualization_demo_onnx.png created! Zero epochs run. ---")
