import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('PreviousDataset.csv')
features = ['Latitude_Degrees', 'Longitude_Degrees', 'Depth', 'ClimSST', 'SSTA', 'TSA_DHW', 'SSTA_DHW', 'SSTA_Frequency', 'TSA_Frequency']

# Convert to numeric
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

plt.figure(figsize=(12, 10))
sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Potential Features')
plt.savefig('feature_correlation.png')
print("Saved feature_correlation.png")
