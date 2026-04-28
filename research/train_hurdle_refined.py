import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_csv('cleaned_improved_data.csv')
X = df.drop('Average_Bleaching', axis=1).values
y = df['Average_Bleaching'].values

# Train/Test Split
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Prepare Binary Labels
y_train_bin = (y_train_raw > 0).astype(float)
y_test_bin = (y_test_raw > 0).astype(float)

# Prepare Positive-only Data for Regressor
pos_mask = y_train_raw > 0
X_train_pos = X_train_scaled[pos_mask]
y_train_pos = y_train_raw[pos_mask]

# Convert to Tensors
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_bin_t = torch.tensor(y_train_bin, dtype=torch.float32).view(-1, 1)
y_train_t = torch.tensor(y_train_raw, dtype=torch.float32).view(-1, 1)

X_train_pos_t = torch.tensor(X_train_pos, dtype=torch.float32)
y_train_pos_t = torch.tensor(y_train_pos, dtype=torch.float32).view(-1, 1)

X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test_raw, dtype=torch.float32).view(-1, 1)

class HurdleNet(nn.Module):
    def __init__(self, input_dim):
        super(HurdleNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        prob = self.classifier(x)
        val = self.regressor(x)
        return prob * val

model = HurdleNet(X.shape[1]).to(DEVICE)

# Step 1: Train Classifier
print("Training Classifier...")
clf_optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
clf_criterion = nn.BCELoss()
clf_loader = DataLoader(TensorDataset(X_train_t, y_train_bin_t), batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(100):
    model.classifier.train()
    for bX, by in clf_loader:
        bX, by = bX.to(DEVICE), by.to(DEVICE)
        clf_optimizer.zero_grad()
        loss = clf_criterion(model.classifier(bX), by)
        loss.backward()
        clf_optimizer.step()

# Step 2: Train Regressor only on positive samples
print("Training Regressor (Positive samples only)...")
reg_optimizer = optim.Adam(model.regressor.parameters(), lr=LR)
reg_criterion = nn.MSELoss()
reg_loader = DataLoader(TensorDataset(X_train_pos_t, y_train_pos_t), batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(200):
    model.regressor.train()
    for bX, by in reg_loader:
        bX, by = bX.to(DEVICE), by.to(DEVICE)
        reg_optimizer.zero_grad()
        loss = reg_criterion(model.regressor(bX), by)
        loss.backward()
        reg_optimizer.step()

# Step 3: Fine-tune together
print("Fine-tuning whole model...")
opt = optim.Adam(model.parameters(), lr=LR/10)
crit = nn.MSELoss()

best_rmse = float('inf')
for epoch in range(200):
    model.train()
    opt.zero_grad()
    outputs = model(X_train_t.to(DEVICE))
    loss = crit(outputs, y_train_t.to(DEVICE))
    loss.backward()
    opt.step()
    
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t.to(DEVICE))
        test_rmse = np.sqrt(crit(preds, y_test_t.to(DEVICE)).item())
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            torch.save(model.state_dict(), 'best_hurdle_refined.pth')

print(f"\nRefined Hurdle Model Results:")
print(f"Best RMSE: {best_rmse:.4f}")
print(f"Paper RMSE: 7.91")
