import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load data
df = pd.read_csv('cleaned_improved_data.csv')
X = df.drop('Average_Bleaching', axis=1).values
y = df['Average_Bleaching'].values

# Architecture
class HurdleNet(nn.Module):
    def __init__(self, input_dim):
        super(HurdleNet, self).__init__()
        # Shared layers? Or separate? Let's go separate for simplicity
        
        # Classifier: Is bleaching > 0?
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
        
        # Regressor: How much bleaching?
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
        return prob * val # Expected value: P(y>0) * E[y|y>0]

def train_fold(fold, train_idx, val_idx, X, y):
    X_train_raw, X_val_raw = X[train_idx], X[val_idx]
    y_train_raw, y_val_raw = y[train_idx], y[val_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_raw, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_raw, dtype=torch.float32).view(-1, 1)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    
    model = HurdleNet(X.shape[1]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    best_rmse = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t.to(DEVICE))
            val_loss = criterion(val_outputs, y_val_t.to(DEVICE)).item()
            val_rmse = np.sqrt(val_loss)
            
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                torch.save(model.state_dict(), f'best_model_fold_{fold}.pth')
                
    return best_rmse

# 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Training Fold {fold+1}...")
    rmse = train_fold(fold, train_idx, val_idx, X, y)
    print(f"Fold {fold+1} Best RMSE: {rmse:.4f}")
    rmses.append(rmse)

print(f"\nAverage Cross-Validation RMSE: {np.mean(rmses):.4f} (+/- {np.std(rmses):.4f})")
print(f"Paper RMSE: 7.91")
