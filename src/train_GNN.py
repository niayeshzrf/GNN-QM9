import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from model import GIN
from custom_dataset import QM9ProcessedDataset
import numpy as np
import os

HOMO_MEAN = 0.011124
HOMO_STD = 0.046936

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
dataset = QM9ProcessedDataset(root='../data/QM9Custom')

# Shuffle and split
torch.manual_seed(42)
dataset = dataset.shuffle()
train_dataset = dataset[:110000]
val_dataset = dataset[110000:120000]
test_dataset = dataset[120000:]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# Model
in_channels = dataset.num_node_features  # Should be 5 for your current setup
model = GIN(in_channels=in_channels, hidden_channels=64, out_channels=1, num_layers=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.mse_loss(out.squeeze(), data.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

# Validation loop
def evaluate(loader):
    model.eval()
    total_error = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = F.mse_loss(out.squeeze(), data.y.squeeze())
            total_error += loss.item() * data.num_graphs
    return total_error / len(loader.dataset)

train_losses = []
val_losses = []

# Training
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)

for epoch in range(1, 201):
    loss = train()
    val_loss = evaluate(val_loader)
    scheduler.step(val_loss)

    train_losses.append(loss)
    val_losses.append(val_loss)

    print(f'Epoch {epoch:03d}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')


model.eval()
y_true_all = []
y_pred_all = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)

        y_pred = data.y.cpu() * HOMO_STD + HOMO_MEAN
        y_true = out.cpu() * HOMO_STD + HOMO_MEAN
        y_true_all.append(y_pred)
        y_pred_all.append(y_true)

# Concatenate
y_true_all = torch.cat(y_true_all, dim=0).squeeze().numpy()
y_pred_all = torch.cat(y_pred_all, dim=0).squeeze().numpy()

# Save losses
os.makedirs("../results", exist_ok=True)
np.save("../results/train_losses.npy", np.array(train_losses))
np.save("../results/val_losses.npy", np.array(val_losses))

# Save predictions and true values
np.save("../results/y_true.npy", y_true_all)
np.save("../results/y_pred.npy", y_pred_all)

print("Saved training logs and predictions to '../results/' folder.")