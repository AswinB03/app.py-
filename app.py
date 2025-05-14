import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open('lr_model.pkl', 'rb'))

st.title('Air Quality Index Prediction')

# Input fields
pm25 = st.number_input('PM2.5')
pm10 = st.number_input('PM10')
no2 = st.number_input('NO2')
so2 = st.number_input('SO2')
co = st.number_input('CO')
o3 = st.number_input('O3')

# Predict button
if st.button('Predict AQI'):
    input_data = np.array([[pm25, pm10, no2, so2, co, o3]])
    prediction = model.predict(input_data)
    st.success(f'Predicted AQI: {prediction[0]:.2f}')
  
