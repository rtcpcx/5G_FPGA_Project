import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import brevitas.nn as qnn
import numpy as np

# 1. Load the massive 50k Dataset
print("--- 1. Loading 50k 5G Dataset ---")
X = np.load('siso_fpga_input_grid_50k.npy').astype(np.float32)
Y = np.load('siso_fpga_target_bits_50k.npy').astype(np.float32)
dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
# Increased batch size to speed up training on large data
train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

# 2. 8-Bit Architecture
class DeepRxBrevitas(nn.Module):
    def __init__(self):
        super(DeepRxBrevitas, self).__init__()
        self.conv1 = qnn.QuantConv2d(2, 16, kernel_size=3, padding=1, weight_bit_width=8)
        self.bn1 = nn.BatchNorm2d(16) 
        self.relu1 = qnn.QuantReLU(bit_width=8)
        self.conv2 = qnn.QuantConv2d(16, 32, kernel_size=3, padding=1, stride=2, weight_bit_width=8)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = qnn.QuantReLU(bit_width=8)
        self.flatten = nn.Flatten()
        self.fc = qnn.QuantLinear(5376, 2304, bias=True, weight_bit_width=8)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)

model = DeepRxBrevitas()
optimizer = optim.Adam(model.parameters(), lr=0.002)
criterion = nn.BCEWithLogitsLoss()

# 3. Train for 10 Epochs (10 passes over 50k is massive)
print("--- 2. Training Golden Model (10 Epochs) ---")
for epoch in range(10):
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
    print(f"Epoch {epoch+1}/10 - Loss: {running_loss/len(dataset):.4f} - Acc: {acc:.2%}")

# 4. Final Export 
print("--- 3. Exporting GOLDEN ONNX (Legacy Path) ---")
model.cpu()
dummy_input = torch.randn(1, 2, 48, 14)
torch.onnx.export(
    model, 
    dummy_input, 
    "deeprx_siso_8bit_GOLDEN.onnx",
    export_params=True,
    opset_version=11, 
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)
print("--- ALL DONE! deeprx_siso_8bit_GOLDEN.onnx created ---")
