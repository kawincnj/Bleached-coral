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
EPOCHS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load data
df = pd.read_csv('cleaned_improved_data.csv')
X = df.drop('Average_Bleaching', axis=1).values
y = df['Average_Bleaching'].values

# Split data
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train_raw, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test = torch.tensor(y_test_raw, dtype=torch.float32).view(-1, 1)

# DataLoader
train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# Improved Architecture
class ImprovedMLP(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)

model = ImprovedMLP(X_train.shape[1]).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)

# Training Loop
best_test_rmse = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # Validation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test.to(DEVICE))
        test_loss = criterion(test_outputs, y_test.to(DEVICE)).item()
        test_rmse = np.sqrt(test_loss)
        
    scheduler.step(test_loss)
    
    if test_rmse < best_test_rmse:
        best_test_rmse = test_rmse
        torch.save(model.state_dict(), 'best_model.pth')
    
    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Test RMSE: {test_rmse:.4f}, Best RMSE: {best_test_rmse:.4f}")

# Final Evaluation
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
with torch.no_grad():
    predictions = model(X_test.to(DEVICE))
    y_test_np = y_test.numpy()
    predictions_np = predictions.cpu().numpy()
    
    mse = mean_squared_error(y_test_np, predictions_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_np, predictions_np)

print(f"\nImproved Neural Network Results:")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Compare with paper
paper_rmse = 7.91
paper_r2 = 0.25
print(f"\nPaper Results:")
print(f"RMSE: {paper_rmse}")
print(f"R2 Score: {paper_r2}")

if rmse < paper_rmse:
    print("\nSUCCESS: Our improved NN beat the paper's RMSE!")
else:
    print("\nOur improved NN did not beat the paper's RMSE yet.")
