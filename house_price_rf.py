import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load data
df = pd.read_csv('Cleaned_Bengaluru_House_Data.csv')

# Check columns
print("Columns in dataset:", df.columns.tolist())

# Features and target
X = df.drop('price', axis=1)
y = df['price']

# Identify categorical and numerical features
categorical_features = ['location']
numerical_features = [col for col in X.columns if col != 'location']

# Preprocessing pipelines
numerical_transformer = SimpleImputer(strategy='mean')  # fill missing numeric with mean
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # fill missing cat with mode
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Split dataset (optional but good)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    'regressor__n_estimators': [50, 100],
    'regressor__max_depth': [10, 20, None],
    'regressor__min_samples_split': [2, 5],
}

grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV MSE: {-grid_search.best_score_:.2f}")

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test set Mean Squared Error (MSE): {mse:.2f}")
print(f"Test set R-squared (R2): {r2:.2f}")

# Feature importance extraction for numeric and categorical features
# We need to get feature names after preprocessing (OneHotEncoding expands categories)
onehot_features = best_model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
all_features = numerical_features + list(onehot_features)

importances = best_model.named_steps['regressor'].feature_importances_

feat_imp_df = pd.DataFrame({'feature': all_features, 'importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=False)
print("\nFeature importances:")
print(feat_imp_df)

