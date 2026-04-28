import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def train_and_eval(features):
    df = pd.read_csv('cleaned_improved_data.csv')
    X = df[features].values
    y = df['Average_Bleaching'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    model = nn.Sequential(
        nn.Linear(len(features), 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t)
        rmse = np.sqrt(mean_squared_error(y_test_t, preds))
    return rmse

original_features = ['Latitude_Degrees', 'Longitude_Degrees', 'Depth', 'ClimSST', 'SSTA']
all_features = ['Latitude_Degrees', 'Longitude_Degrees', 'Depth', 'ClimSST', 'SSTA', 'TSA_DHW', 'SSTA_DHW', 'SSTA_Frequency', 'TSA_Frequency']

print("Training with original features...")
rmse_orig = train_and_eval(original_features)
print(f"Original features RMSE: {rmse_orig:.4f}")

print("Training with ALL features...")
rmse_all = train_and_eval(all_features)
print(f"All features RMSE: {rmse_all:.4f}")

print(f"\nImprovement: {rmse_orig - rmse_all:.4f}")
