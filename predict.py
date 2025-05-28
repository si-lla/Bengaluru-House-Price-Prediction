import pickle
import pandas as pd

# Load the saved model
with open('bengaluru_house_price_model.pkl', 'rb') as file:
    loaded_obj = pickle.load(file)

    print(type(loaded_obj))
    model = pickle.load(file)

# Example new data - replace values as needed
new_data = pd.DataFrame({
    'bath': [2],
    'balcony': [1],
    'bhk': [3],
    'total_sqft_cleaned': [1200],
    'price_per_sqft': [5000]
})

# Predict the price
predicted_price = model.predict(new_data)

print(f"Predicted house price: {predicted_price[0]:.2f} lakhs")
