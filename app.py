import streamlit as st
import pickle
import numpy as np

# Load model
with open('bengaluru_house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Bengaluru House Price Prediction")

bath = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)
balcony = st.number_input("Balconies", min_value=0, max_value=10, value=1)
bhk = st.number_input("BHK", min_value=0, max_value=10, value=3)
total_sqft = st.number_input("Total Sqft", min_value=100, max_value=10000, value=1200)
price_per_sqft = st.number_input("Price per Sqft", min_value=100, max_value=10000, value=5000)

if st.button("Predict Price"):
    features = np.array([[bath, balcony, bhk, total_sqft, price_per_sqft]])
    prediction = model.predict(features)
    st.success(f"Predicted Price: {prediction[0]:.2f} lakhs")
