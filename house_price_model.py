import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the cleaned dataset
df = pd.read_csv('Cleaned_Bengaluru_House_Data.csv')

# Select relevant features and target
X = df[['bath', 'balcony', 'bhk', 'total_sqft_cleaned', 'price_per_sqft']]
y = df['price']

# Drop rows with any NaN values in features or target
data = pd.concat([X, y], axis=1).dropna()
X = data[['bath', 'balcony', 'bhk', 'total_sqft_cleaned', 'price_per_sqft']]
y = data['price']

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2 ): {r2:.2f}')

# Optional: Print feature coefficients
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print('\nFeature coefficients:')
print(coeff_df)
