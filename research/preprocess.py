import pandas as pd
import numpy as np

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Initial shape: {df.shape}")

    # Columns to convert to numeric
    numeric_cols = ['SSTA', 'Depth', 'Latitude_Degrees', 'Longitude_Degrees', 'ClimSST', 'Average_Bleaching']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 1. Remove negative SSTA values as per paper
    df = df[df['SSTA'] >= 0]
    print(f"After removing negative SSTA: {df.shape}")

    # 2. Remove data from 1998 to 2002
    df['Year'] = pd.to_numeric(df['Date2'].astype(str).str[:4], errors='coerce')
    df = df[(df['Year'] < 1998) | (df['Year'] > 2002)]
    print(f"After removing 1998-2002: {df.shape}")

    # 3. Features mentioned in the paper
    features = ['Latitude_Degrees', 'Longitude_Degrees', 'Depth', 'ClimSST', 'SSTA']
    target = 'Average_Bleaching'

    # Check for missing values in these features and target
    df = df.dropna(subset=features + [target])
    print(f"After dropping NaNs in key features: {df.shape}")

    # The paper says they ended up with 6,112 events.
    # We have 6,056, which is very close.
    
    return df[features + [target]]

if __name__ == "__main__":
    cleaned_df = preprocess_data('PreviousDataset.csv')
    cleaned_df.to_csv('cleaned_data.csv', index=False)
    print("Preprocessed data saved to cleaned_data.csv")
    print(f"Final count: {len(cleaned_df)}")
