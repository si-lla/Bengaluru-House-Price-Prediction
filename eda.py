import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_clean = pd.read_csv('Cleaned_Bengaluru_House_Data.csv')

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Histogram of price
sns.histplot(df_clean['price'], bins=50, kde=True, ax=axs[0, 0])
axs[0, 0].set_title('Price Distribution')

# Histogram of total_sqft_cleaned (house size)
sns.histplot(df_clean['total_sqft_cleaned'], bins=50, kde=True, ax=axs[0, 1])
axs[0, 1].set_title('Total Sqft Distribution')

# Histogram of bhk (number of bedrooms)
sns.histplot(df_clean['bhk'], bins=range(int(df_clean['bhk'].min()), int(df_clean['bhk'].max())+2), 
             kde=False, ax=axs[1, 0])
axs[1, 0].set_title('BHK Distribution')

# Histogram of price_per_sqft
sns.histplot(df_clean['price_per_sqft'], bins=50, kde=True, ax=axs[1, 1])
axs[1, 1].set_title('Price per Sqft Distribution')

plt.tight_layout()
plt.show()
