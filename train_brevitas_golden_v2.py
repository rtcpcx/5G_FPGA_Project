import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import brevitas.nn as qnn
import numpy as np

print("--- 1. Loading 50k 5G Dataset ---")
X = np.load('siso_fpga_input_grid_50k.npy').astype(np.float32)
Y = np.load('siso_fpga_target_bits_50k.npy').astype(np.float32)
dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

# 2. Upgraded 8-Bit Architecture (Wider CNN)
class DeepRxBrevitasV2(nn.Module):
    def __init__(self):
        super(DeepRxBrevitasV2, self).__init__()
        # Layer 1: 32 Filters
        self.conv1 = qnn.QuantConv2d(2, 32, kernel_size=3, padding=1, weight_bit_width=8)
        self.bn1 = nn.BatchNorm2d(32) 
        self.relu1 = qnn.QuantReLU(bit_width=8)
        
        # Layer 2: 64 Filters
        self.conv2 = qnn.QuantConv2d(32, 64, kernel_size=3, padding=1, stride=2, weight_bit_width=8)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = qnn.QuantReLU(bit_width=8)
        
        self.flatten = nn.Flatten()
        # Adjusted math for wider layer: 64 channels * 24 * 7 = 10752
        self.fc = qnn.QuantLinear(10752, 2304, bias=True, weight_bit_width=8)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)

model = DeepRxBrevitasV2()
optimizer = optim.Adam(model.parameters(), lr=0.002)
# Learning Rate Scheduler: Cut LR in half every 5 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.BCEWithLogitsLoss()

print("--- 2. Training Upgraded Golden Model (20 Epochs) ---")
for epoch in range(20):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    # Step the scheduler
    scheduler.step()
    
    preds = (torch.sigmoid(outputs) > 0.5).float()
    acc = (preds == labels).float().mean()
    print(f"Epoch {epoch+1}/20 - Loss: {running_loss/len(dataset):.4f} - Acc: {acc:.2%} - LR: {scheduler.get_last_lr()[0]}")

print("--- 3. Exporting V2 GOLDEN ONNX ---")
model.eval() # <--- Fixed the warning!
model.cpu()
dummy_input = torch.randn(1, 2, 48, 14)
torch.onnx.export(
    model, 
    dummy_input, 
    "deeprx_siso_8bit_GOLDEN_V2.onnx",
    export_params=True,
    opset_version=11, 
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)
print("--- ALL DONE! deeprx_siso_8bit_GOLDEN_V2.onnx created ---")
