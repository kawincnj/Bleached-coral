import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load cleaned data
df = pd.read_csv('cleaned_data.csv')
X = df.drop('Average_Bleaching', axis=1).values
y = df['Average_Bleaching'].values

# Split data (80/20 as per paper)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Simple MLP Architecture
class BaselineMLP(nn.Module):
    def __init__(self, input_dim):
        super(BaselineMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

model = BaselineMLP(X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    mse = criterion(predictions, y_test).item()
    rmse = np.sqrt(mse)
    
    y_test_np = y_test.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    r2 = r2_score(y_test_np, predictions_np)

print(f"\nBaseline Neural Network Results:")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Compare with paper
paper_rmse = 7.91
paper_r2 = 0.25
print(f"\nPaper Results:")
print(f"RMSE: {paper_rmse}")
print(f"R2 Score: {paper_r2}")

if rmse < paper_rmse:
    print("\nSUCCESS: Our baseline NN beat the paper's RMSE!")
else:
    print("\nBaseline NN did not beat the paper's RMSE yet.")
