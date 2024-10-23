import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model from the .pkl file
model_path = "rainfall_prediction_model.pkl"
lr = joblib.load(model_path)

# Load the dataset and scale it, if necessary (just for scaling new inputs)
df = pd.read_csv(r"rainfall_prediction_dataset.csv")
X = df.drop(columns=['rainfall'])  # Features
Y = df['rainfall']  # Target labels

# Scaling (just to align with the training)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)

# Streamlit App
logo_path = r"C:\Users\Olufemi George\Desktop\Data Science Class Environment\Project Presentation\rain_logo.png" 
st.image(logo_path, use_column_width='auto')

st.title("Rainfall Prediction for Optimized Agricultural Operations")

st.write("""
Input the corresponding data below to predict if there will be rainfall.
""")

# Sidebar for user inputs
st.header('Input Parameters')

def user_input_features():
    pressure = st.number_input('Atmospheric Pressure ', min_value=0.0, max_value=1200.0, format="%.2f")
    maxtemp = st.number_input('Maximum Temperature ', min_value=0.0, max_value=40.0, format="%.2f")
    humidity = st.number_input('Humidity Level', min_value=0, max_value=100)  
    dewpoint = st.number_input('Dew Point', min_value=-30.0, max_value=30.0, format="%.2f")
    cloud = st.number_input('Cloud level', min_value=0, max_value=100)  
    sunshine = st.number_input('Sun intensity', min_value=0.0, max_value=50.0, format="%.2f")
    windspeed = st.number_input('Wind Speed', min_value=0.0, max_value=150.0, format="%.2f")
    
    data = {
        'pressure': pressure,
        'maxtemp': maxtemp,
        'humidity': humidity,
        'dewpoint': dewpoint,  
        'cloud': cloud,
        'sunshine': sunshine,
        'windspeed': windspeed
    }

    features = pd.DataFrame(data, index=[0])
    features = features.reindex(columns=X.columns, fill_value=0)
    return features

input_df = user_input_features()

if st.button('Submit'):
    input_scaled = scaler.transform(input_df)

    # Predict using the pre-trained model
    prediction = lr.predict(input_scaled)[0]  
    prediction_proba = lr.predict_proba(input_scaled)  
    
    prob_rain = prediction_proba[0][1]  
    prob_no_rain = prediction_proba[0][0]  

    st.subheader('Prediction')
    if prediction == 1:
        st.write("Rain is expected.")
    else:
        st.write("No rain is expected.")
    
    st.subheader('Prediction Probability')
    st.write(f"Probability of Rain: {prob_rain:.2f}")
    st.write(f"Probability of No Rain: {prob_no_rain:.2f}")
