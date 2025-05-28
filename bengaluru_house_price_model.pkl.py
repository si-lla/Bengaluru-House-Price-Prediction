import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv('Cleaned_Bengaluru_House_Data.csv')

# Features and target
X = df.drop('price', axis=1)
y = df['price']

# Identify categorical and numerical columns
cat_cols = ['location']
num_cols = ['bath', 'balcony', 'bhk', 'total_sqft_cleaned', 'price_per_sqft']

# Preprocessing pipelines
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

# Define model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 20],
    'model__min_samples_split': [2, 5]
}

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid Search
grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)

# Predict on test set
y_pred = grid_search.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 ): {r2:.2f}")

# Plot Predicted vs Actual
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line
plt.show()

# Save model
joblib.dump(grid_search.best_estimator_, 'bengaluru_house_price_model.pkl')
print("Model saved as 'bengaluru_house_price_model.pkl'")

# Prediction function for new data
def predict_price(new_data_dict):
    """
    Predict price for new house data.
    new_data_dict: dictionary with keys:
      'location', 'bath', 'balcony', 'bhk', 'total_sqft_cleaned', 'price_per_sqft'
    """
    new_df = pd.DataFrame([new_data_dict])
    loaded_model = joblib.load('bengaluru_house_price_model.pkl')
    prediction = loaded_model.predict(new_df)
    return prediction[0]

# Example usage:
# new_house = {
#     'location': 'Whitefield',
#     'bath': 2,
#     'balcony': 1,
#     'bhk': 3,
#     'total_sqft_cleaned': 1200,
#     'price_per_sqft': 5000
# }
# print("Predicted price:", predict_price(new_house))
