import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Re-define the architecture (must match train_improved.py)
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

def predict(input_data):
    # input_data should be a list of 9 features or a list of lists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    input_dim = 9
    model = ImprovedMLP(input_dim).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    
    # Load or recreate scaler
    # For a real app, we should save the scaler during training.
    # Let's quickly recreate it from the training data for this demo.
    df = pd.read_csv('cleaned_improved_data.csv')
    X = df.drop('Average_Bleaching', axis=1).values
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Preprocess input
    input_array = np.array(input_data).reshape(-1, input_dim)
    input_scaled = scaler.transform(input_array)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor)
    
    return prediction.cpu().numpy().flatten()

if __name__ == "__main__":
    # Example: Take a row from the cleaned data
    test_df = pd.read_csv('cleaned_improved_data.csv').head(5)
    test_X = test_df.drop('Average_Bleaching', axis=1).values
    test_y = test_df['Average_Bleaching'].values
    
    print("Making predictions on the first 5 rows of the dataset:")
    preds = predict(test_X)
    
    for i in range(5):
        print(f"Sample {i+1}: Actual Bleaching: {test_y[i]:.2f}%, Predicted: {max(0, preds[i]):.2f}%")
