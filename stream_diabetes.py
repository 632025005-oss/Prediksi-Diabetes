import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
try:
    diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
    st.sidebar.success("✅ Model loaded")
except:
    st.sidebar.error("❌ Model error")
    diabetes_model = None

# UI
st.title('🩺 Prediksi Diabetes')

# Input fields
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=6)
with col2:
    Glucose = st.number_input('Glucose', min_value=0, max_value=200, value=148)
with col1:
    BloodPressure = st.number_input('Blood Pressure', min_value=0, max_value=130, value=72)
with col2:
    SkinThickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=35)
with col1:
    Insulin = st.number_input('Insulin', min_value=0, max_value=900, value=0)
with col2:
    BMI = st.number_input('BMI', min_value=0.0, max_value=70.0, value=33.6)
with col1:
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.627)
with col2:
    Age = st.number_input('Age', min_value=1, max_value=120, value=50)

# Tombol prediksi - FIXED VERSION
if st.button('Test Prediksi Diabetes'):
    if diabetes_model is not None:
        try:
            # Input data
            input_data = np.array([[pregnancies, Glucose, BloodPressure, SkinThickness,
                                   Insulin, BMI, DiabetesPedigreeFunction, Age]])
            
            # PREDIKSI SAJA (tanpa predict_proba)
            prediction = diabetes_model.predict(input_data)[0]
            
            # Tampilkan hasil
            if prediction == 1:
                st.error('**⚠️ HASIL: DIABETES**')
                st.write("Pasien diprediksi terkena diabetes")
            else:
                st.success('**✅ HASIL: TIDAK DIABETES**')
                st.write("Pasien diprediksi tidak terkena diabetes")
                
            # Info model
            st.info(f"Menggunakan model: **{type(diabetes_model).__name__}**")
            
        except AttributeError:
            # Jika error karena SVC, coba cara lain
            st.warning("Model SVC tidak support probability")
            
            # Coba dengan decision_function jika SVC
            if hasattr(diabetes_model, 'decision_function'):
                score = diabetes_model.decision_function(input_data)[0]
                st.write(f"Decision score: {score:.2f}")
                
                # Interpretasi manual
                if score > 0:
                    st.error("Kemungkinan Diabetes")
                else:
                    st.success("Kemungkinan Sehat")
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Model tidak tersedia")

# Tampilkan sample data
st.header("📊 Data Sample")
df = pd.read_csv("diabetes.csv")
st.dataframe(df.head())
