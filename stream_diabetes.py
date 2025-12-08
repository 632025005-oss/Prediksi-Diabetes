import streamlit as st
import pandas as pd
import numpy as np

# Title
st.title('🩺 Prediksi Diabetes')

try:
    from sklearn.ensemble import RandomForestClassifier
    
    # Load data
    df = pd.read_csv("diabetes.csv")
    
    # Create model
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    st.success("Model machine learning siap!")
    
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input('Pregnancies', 0, 20, 3)
        glucose = st.number_input('Glucose', 0, 200, 117)
        blood_pressure = st.number_input('Blood Pressure', 0, 130, 72)
        skin_thickness = st.number_input('Skin Thickness', 0, 100, 23)
    
    with col2:
        insulin = st.number_input('Insulin', 0, 900, 30)
        bmi = st.number_input('BMI', 0.0, 70.0, 32.0)
        dpf = st.number_input('Diabetes Pedigree', 0.0, 2.5, 0.3725)
        age = st.number_input('Age', 21, 100, 29)
    
    # Predict button
    if st.button('Prediksi'):
        input_data = [[pregnancies, glucose, blood_pressure, skin_thickness,
                      insulin, bmi, dpf, age]]
        prediction = model.predict(input_data)[0]
        
        if prediction == 1:
            st.error('Hasil: Diabetes')
        else:
            st.success('Hasil: Tidak Diabetes')
            
except ImportError:
    st.error("Error: scikit-learn tidak terinstall!")
    st.write("Pastikan requirements.txt berisi 'scikit-learn'")
except Exception as e:
    st.error(f"Error: {e}")
