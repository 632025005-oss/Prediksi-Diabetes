import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Title
st.title('🩺 Prediksi Diabetes')

# Load data
df = pd.read_csv("diabetes.csv")

# Buat model langsung dari data (TIDAK pakai file .sav)
diabetes_model = RandomForestClassifier(n_estimators=100, random_state=42)
X = df.drop('Outcome', axis=1)
y = df['Outcome']
diabetes_model.fit(X, y)

st.sidebar.success("✅ Model siap digunakan!")

# Input
st.header("Data Pasien")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Kehamilan', 0, 20, 3)
    glucose = st.number_input('Glukosa', 0, 200, 117)
    blood_pressure = st.number_input('Tekanan Darah', 0, 130, 72)
    skin_thickness = st.number_input('Ketebalan Kulit', 0, 100, 23)

with col2:
    insulin = st.number_input('Insulin', 0, 900, 30)
    bmi = st.number_input('BMI', 0.0, 70.0, 32.0)
    dpf = st.number_input('Diabetes Pedigree', 0.0, 2.5, 0.3725)
    age = st.number_input('Umur', 21, 100, 29)

# Predict
if st.button('🔍 Prediksi'):
    input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, 
                   insulin, bmi, dpf, age]]
    
    prediction = diabetes_model.predict(input_data)[0]
    probability = diabetes_model.predict_proba(input_data)[0][1] * 100
    
    if prediction == 1:
        st.error(f'**DIABETES** ({probability:.1f}% probability)')
    else:
        st.success(f'**SEHAT** ({probability:.1f}% probability)')

# Tampilkan data sample
with st.expander("📊 Lihat Data Sample"):
    st.dataframe(df.head(10))
