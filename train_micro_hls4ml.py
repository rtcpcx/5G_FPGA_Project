import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

print("--- 1. Loading 50k 5G Dataset ---")
X = np.load('siso_fpga_input_grid_50k.npy').astype(np.float32)
Y = np.load('siso_fpga_target_bits_50k.npy').astype(np.float32)
dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

# 2. Micro CNN Architecture explicitly designed for hls4ml
class MicroCNN_HLS4ML(nn.Module):
    def __init__(self):
        super(MicroCNN_HLS4ML, self).__init__()
        # Input: [Batch, 2, 48, 14]
        self.conv1 = nn.Conv2d(2, 4, kernel_size=3, padding=1) 
        self.relu1 = nn.ReLU()
        # Downsample time/freq axes
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))  
        # Output after pool1: [Batch, 4, 24, 7]
        
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # Downsample again
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        # Output after pool2: [Batch, 8, 12, 3] = 288 features
        
        self.flatten = nn.Flatten()
        
        # 288 in, 2304 out = ~663k parameters! Extremely small DSP footprint.
        self.fc = nn.Linear(288, 2304)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)

model = MicroCNN_HLS4ML()
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.BCEWithLogitsLoss()

print("--- 2. Training Micro-CNN for hls4ml (15 Epochs) ---")
# 15 epochs to hit BER target 0.2 - 0.3
for epoch in range(15):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        
    preds = (torch.sigmoid(outputs) > 0.5).float()
    acc = (preds == labels).float().mean()
    print(f"Epoch {epoch+1}/15 - Loss: {running_loss/len(dataset):.4f} - Acc: {acc:.2%}")

print("--- 3. Exporting Standard .pth and .onnx for hls4ml ---")
model.eval()
model.cpu()

torch.save(model.state_dict(), "micro_cnn_hls4ml.pth")
torch.save(model, "micro_cnn_hls4ml_full.pth")

dummy_input = torch.randn(1, 2, 48, 14)
torch.onnx.export(
    model, 
    dummy_input, 
    "micro_cnn_hls4ml.onnx",
    export_params=True,
    opset_version=13, 
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

print("--- ALL DONE! micro_cnn_hls4ml files created successfully ---")
