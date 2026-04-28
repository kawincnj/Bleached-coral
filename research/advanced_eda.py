import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('PreviousDataset.csv')

# Convert to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop columns with too many NaNs
thresh = len(df) * 0.5
df = df.dropna(thresh=thresh, axis=1)

# Correlation with Average_Bleaching
correlations = df.corr()['Average_Bleaching'].sort_values(ascending=False)
print("\nTop 20 correlations with Average_Bleaching:")
print(correlations.head(20))

print("\nBottom 10 correlations with Average_Bleaching:")
print(correlations.tail(10))

# Check for zero-inflation
zero_count = (df['Average_Bleaching'] == 0).sum()
print(f"\nZero-inflation: {zero_count}/{len(df)} ({zero_count/len(df)*100:.2f}%)")

# Distribution of target when > 0
plt.figure(figsize=(10, 6))
sns.histplot(df[df['Average_Bleaching'] > 0]['Average_Bleaching'], bins=50, kde=True)
plt.title('Distribution of Average_Bleaching (values > 0)')
plt.savefig('target_distribution_non_zero.png')
print("\nSaved target_distribution_non_zero.png")
