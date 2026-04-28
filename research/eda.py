import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('PreviousDataset.csv')

# Basic Info
print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# Check for missing values in key columns mentioned in the paper
key_features = ['Latitude_Degrees', 'Longitude_Degrees', 'Depth', 'ClimSST', 'SSTA', 'Average_Bleaching']
missing = df[key_features].isnull().sum()
print("\nMissing values in key features:\n", missing)

# Summary statistics for target
print("\nTarget Summary (Average_Bleaching):")
print(df['Average_Bleaching'].describe())

# Check for CUDA
print("\nCUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))
