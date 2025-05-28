import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('Bengaluru_House_Data.csv')

# Copy for cleaning
df_clean = df.copy()

# 1. Extract number of bedrooms (BHK) from 'size'
df_clean['bhk'] = df_clean['size'].str.extract(r'(\d+)', expand=False).astype(float)

# 2. Clean 'total_sqft' to convert ranges and remove non-numeric values
def convert_sqft(x):
    try:
        return float(x)
    except:
        if '-' in str(x):
            parts = x.split('-')
            if len(parts) == 2:
                try:
                    return (float(parts[0]) + float(parts[1])) / 2
                except:
                    return None
        return None

df_clean['total_sqft_cleaned'] = df_clean['total_sqft'].apply(convert_sqft)

# 3. Drop rows with missing critical values
df_clean = df_clean.dropna(subset=['location', 'total_sqft_cleaned', 'bhk', 'price'])

# 4. Remove unrealistic entries
df_clean = df_clean[df_clean['total_sqft_cleaned'] / df_clean['bhk'] >= 300]  # At least 300 sqft per BHK

# 5. Compute price per square foot (₹)
df_clean['price_per_sqft'] = (df_clean['price'] * 100000) / df_clean['total_sqft_cleaned']

# 6. Clean location names (remove spaces)
df_clean['location'] = df_clean['location'].str.strip()

# 7. Group rare locations as 'Other'
location_stats = df_clean['location'].value_counts()
rare_locations = location_stats[location_stats <= 10].index
df_clean['location'] = df_clean['location'].apply(lambda x: 'Other' if x in rare_locations else x)

# 8. Optional: Drop irrelevant columns (like 'society', 'availability')
df_clean = df_clean.drop(['area_type', 'availability', 'size', 'society', 'total_sqft'], axis=1)

# 9. Reset index
df_clean = df_clean.reset_index(drop=True)

# 10. Save cleaned data
df_clean.to_csv('Cleaned_Bengaluru_House_Data.csv', index=False)

print("✅ Data cleaning complete. Saved to 'Cleaned_Bengaluru_House_Data.csv'")
print(f"Final shape: {df_clean.shape}")
