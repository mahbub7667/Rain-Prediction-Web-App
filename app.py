import streamlit as st
import pandas as pd
import pickle

# PAGE CONFIGURATION & CUSTOM CSS 
st.set_page_config(page_title="Rain Predicter AI ", page_icon="⛈️", layout="wide", initial_sidebar_state="expanded")

# Customizing the UI 
st.markdown("""
<style>
    /* Main Titles */
    .main-title { font-size: 48px !important; font-weight: 800; color: #1E90FF; text-align: center; margin-bottom: -15px; letter-spacing: 1px;}
    .sub-title { font-size: 18px; color: #808080; text-align: center; margin-bottom: 40px; font-weight: 500;}
    
    /* P Output Boxes */
    .rain-box { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 30px; border-radius: 20px; color: white; text-align: center; box-shadow: 0 10px 20px rgba(0,0,0,0.3); transition: transform 0.3s;}
    .rain-box:hover { transform: translateY(-5px); }
    .sun-box { background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%); padding: 30px; border-radius: 20px; color: white; text-align: center; box-shadow: 0 10px 20px rgba(0,0,0,0.3); transition: transform 0.3s;}
    .sun-box:hover { transform: translateY(-5px); }
    
    /* Result Text Styling */
    .result-text { font-size: 32px; font-weight: 800; margin-bottom: 5px; }
    .result-subtext { font-size: 16px; font-weight: 400; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# 2. LOAD MODELS
@st.cache_resource
def load_models():
    try:
        with open('lightgbm_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models. Ensure 'lightgbm_model.pkl' and 'scaler.pkl' are in the same folder. Error: {e}")
        return None, None

model, scaler = load_models()

#  MAIN DASHBOARD UI
st.markdown('<p class="main-title">☁️ Rain Prediction AI </p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Adjust today\'s weather parameters to generate a highly accurate forecast for tomorrow</p>', unsafe_allow_html=True)
st.divider()

# Creating 3 columns for inputs
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🌡️ Temperature")
    min_temp = st.number_input("Min Temperature (°C)", value=13.0, step=0.5)
    max_temp = st.number_input("Max Temperature (°C)", value=25.0, step=0.5)
    temp9am = st.slider("Temp at 9 AM (°C)", 0.0, 40.0, 16.9)
    temp3pm = st.slider("Temp at 3 PM (°C)", 0.0, 45.0, 21.8)

with col2:
    st.markdown("### 💧 Moisture & Clouds")
    rainfall = st.number_input("Rainfall Today (mm)", value=0.0, step=0.1)
    humidity9am = st.slider("Humidity at 9 AM (%)", 0, 100, 71)
    humidity3pm = st.slider("Humidity at 3 PM (%)", 0, 100, 40)
    cloud9am = st.slider("Cloud Cover 9 AM (0-8)", 0.0, 8.0, 4.0)
    cloud3pm = st.slider("Cloud Cover 3 PM (0-8)", 0.0, 8.0, 4.0)

with col3:
    st.markdown("### 🌬️ Wind & Pressure")
    sunshine = st.slider("Sunshine (hours)", 0.0, 14.0, 8.0)
    wind_speed = st.slider("Wind Gust Speed (km/h)", 0, 150, 44)
    wind9am = st.slider("Wind Speed 9 AM (km/h)", 0, 100, 20)
    wind3pm = st.slider("Wind Speed 3 PM (km/h)", 0, 100, 24)
    pressure9am = st.slider("Pressure at 9 AM (hPa)", 980.0, 1040.0, 1007.7)
    pressure3pm = st.slider("Pressure at 3 PM (hPa)", 980.0, 1040.0, 1007.1)

st.write("") # Spacer

#  PREDICTION BUTTON & RESULT
# Big, full-width button
predict_clicked = st.button("🔮 Generate Tomorrow's Forecast", use_container_width=True, type="primary")

if predict_clicked:
    if model is not None and scaler is not None:
        # Professional UI Notification
        st.toast('Analyzing weather patterns...', icon='⏳')
        
        # Preparing the input dictionary perfectly aligned with training data
        input_data = {
            'Location': [0], 
            'MinTemp': [min_temp],
            'MaxTemp': [max_temp],
            'Rainfall': [rainfall],
            'Evaporation': [5.0],   
            'Sunshine': [sunshine],      
            'WindGustDir': [0],
            'WindGustSpeed': [wind_speed],
            'WindDir9am': [0],
            'WindDir3pm': [0],
            'WindSpeed9am': [wind9am],
            'WindSpeed3pm': [wind3pm],
            'Humidity9am': [humidity9am],
            'Humidity3pm': [humidity3pm],
            'Pressure9am': [pressure9am],
            'Pressure3pm': [pressure3pm],
            'Cloud9am': [cloud9am],
            'Cloud3pm': [cloud3pm],
            'Temp9am': [temp9am],
            'Temp3pm': [temp3pm],
            'RainToday': [1 if rainfall > 1.0 else 0]
        }
        
        # Convert to DataFrame and Scale
        new_data_df = pd.DataFrame(input_data)
        new_data_scaled = scaler.transform(new_data_df)
        
        # Predict
        prediction = model.predict(new_data_scaled)
        
        st.toast('Prediction Generated!', icon='✅')
        st.divider()
        
        # Result
        if prediction[0] == 1:
            st.markdown("""
            <div class="rain-box">
                <div class="result-text">🌧️ HIGH PROBABILITY OF RAIN</div>
                <div class="result-subtext">Tomorrow's conditions are highly favorable for rainfall.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sun-box">
                <div class="result-text">☀️ CLEAR & DRY DAY</div>
                <div class="result-subtext">No significant rainfall is expected tomorrow.</div>
            </div>
            """, unsafe_allow_html=True)
