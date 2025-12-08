# -*- coding: utf-8 -*-
import pickle
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load data
df = pd.read_csv("diabetes.csv")

# Load model DENGAN ERROR HANDLING
try:
    # Coba load model
    diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
    st.sidebar.success("✅ Model loaded successfully!")
    
except Exception as e:
    st.sidebar.error(f"❌ Model error: {str(e)[:100]}...")
    st.sidebar.info("Using demo model instead")
    
    # Buat model dummy sebagai backup
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    diabetes_model = RandomForestClassifier(n_estimators=100, random_state=42)
    diabetes_model.fit(X, y)

# UI
st.title('Prediksi Diabetes')

# Input fields
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=3)
with col2:
    Glucose = st.number_input('Glucose', min_value=0, max_value=200, value=117)
with col1:
    BloodPressure = st.number_input('Blood Pressure', min_value=0, max_value=130, value=72)
with col2:
    SkinThickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=23)
with col1:
    Insulin = st.number_input('Insulin', min_value=0, max_value=900, value=30)
with col2:
    BMI = st.number_input('BMI', min_value=0.0, max_value=70.0, value=32.0)
with col1:
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.3725)
with col2:
    Age = st.number_input('Age', min_value=21, max_value=100, value=29)

# Tombol prediksi
if st.button('Test Prediksi Diabetes'):
    try:
        # Buat array input
        input_data = np.array([[pregnancies, Glucose, BloodPressure, SkinThickness, 
                                Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        # Prediksi
        diab_prediction = diabetes_model.predict(input_data)
        diab_proba = diabetes_model.predict_proba(input_data)
        
        # Hasil
        if diab_prediction[0] == 1:
            st.error('⚠️ **Hasil: Pasien terkena Diabetes**')
        else:
            st.success('✅ **Hasil: Pasien tidak terkena Diabetes**')
            
        # Tampilkan probability
        st.write(f"Probability: {diab_proba[0][1]*100:.2f}%")
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Tampilkan data sample
with st.expander("Lihat Data Sample"):
    st.dataframe(df.head())
