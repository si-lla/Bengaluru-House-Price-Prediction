import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# 1. Load cleaned data
df = pd.read_csv('Cleaned_Bengaluru_House_Data.csv')

# 2. Select features and target
# Make sure these columns exist in your CSV exactly as below
features = ['bath', 'balcony', 'bhk', 'total_sqft_cleaned', 'price_per_sqft']
X = df[features]
y = df['price']

# 3. Handle missing values by dropping rows with NaNs
X = X.dropna()
y = y.loc[X.index]

# 4. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluate model
y_pred = model.predict(X_test)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared (R2):", r2_score(y_test, y_pred))

# 7. Save model to a file
with open('bengaluru_house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as bengaluru_house_price_model.pkl")

# 8. Load the saved model (to verify loading works)
with open('bengaluru_house_price_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

print("Loaded model type:", type(loaded_model))

# 9. Predict on a new sample input (example data)
# Make sure the order and count of features match the training data!
import pandas as pd

# Define the feature names your model was trained on
features = ['bath', 'balcony', 'bhk', 'total_sqft_cleaned', 'price_per_sqft']

# Create a DataFrame with your sample input data
features = ['bath', 'balcony', 'bhk', 'total_sqft_cleaned', 'price_per_sqft']

# Create a DataFrame with your sample input data
sample_input_df = pd.DataFrame([[2, 1, 3, 1200, 5000]], columns=features)

predicted_price = loaded_model.predict(sample_input_df)
print("Predicted price for sample input:", predicted_price[0])