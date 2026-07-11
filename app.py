import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Rain Prediction App", page_icon="🌧️", layout="centered")

st.title("🌧️ Rain Prediction Web App")
st.write("Enter your weather details for tomorrow raining prediction !")

@st.cache_resource
def load_models():
    try:
        # Make sure these files are in the same folder as app.py
        with open('lightgbm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

model, scaler = load_models()

st.subheader("🌡️ Enter Weather Details")

col1, col2 = st.columns(2)

with col1:
    min_temp = st.number_input("Min Temperature (°C)", value=13.0)
    max_temp = st.number_input("Max Temperature (°C)", value=25.0)
    rainfall = st.number_input("Rainfall Today (mm)", value=0.0)

with col2:
    humidity9am = st.slider("Humidity at 9 AM (%)", 0, 100, 60)
    humidity3pm = st.slider("Humidity at 3 PM (%)", 0, 100, 40)
    wind_speed = st.slider("Wind Speed (km/h)", 0, 150, 20)

if st.button("Predict Rain Tomorrow 🚀"):
    if model is not None and scaler is not None:
        # Creating a dictionary with user inputs and default medians for the rest
        # Ensure these column names perfectly match your training data
        input_data = {
            'Location': [0], # Assuming encoded value
            'MinTemp': [min_temp],
            'MaxTemp': [max_temp],
            'Rainfall': [rainfall],
            'Evaporation': [5.0],   
            'Sunshine': [8.0],      
            'WindGustDir': [0],
            'WindGustSpeed': [wind_speed],
            'WindDir9am': [0],
            'WindDir3pm': [0],
            'WindSpeed9am': [wind_speed],
            'WindSpeed3pm': [wind_speed],
            'Humidity9am': [humidity9am],
            'Humidity3pm': [humidity3pm],
            'Pressure9am': [1010.0],
            'Pressure3pm': [1008.0],
            'Cloud9am': [4.0],
            'Cloud3pm': [4.0],
            'Temp9am': [min_temp + 2],
            'Temp3pm': [max_temp - 2],
            'RainToday': [1 if rainfall > 1.0 else 0]
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)
        
        # Scale the data
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)
        
        # Display Result
        st.divider()
        if prediction[0] == 1:
            st.error("### 🌧️ Prediction: Yes, it will Rain Tomorrow!")
        else:
            st.success("### ☀️ Prediction: No, it will be a Dry Day!")
    else:
        st.warning("Please ensure 'lightgbm_model.pkl' and 'scaler.pkl' are in the same folder.")