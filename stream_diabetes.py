import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Diabetes",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS kustom
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .positive {
        background-color: #FEE2E2;
        border-left: 5px solid #DC2626;
    }
    .negative {
        background-color: #DCFCE7;
        border-left: 5px solid #16A34A;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Judul dengan style
st.markdown('<h1 class="main-header">🏥 Aplikasi Prediksi Diabetes</h1>', unsafe_allow_html=True)
st.markdown("***Aplikasi untuk mendeteksi risiko diabetes berdasarkan data medis***")

# Sidebar untuk navigasi
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/diabetes.png", width=100)
    st.title("Navigasi")
    
    menu = st.radio(
        "Pilih Menu:",
        ["🏠 Beranda", "📊 Prediksi", "📈 Analisis", "📋 Data", "ℹ️ Tentang"]
    )
    
    st.markdown("---")
    st.info("""
    **Cara Penggunaan:**
    1. Pilih menu **Prediksi**
    2. Isi data pasien
    3. Klik tombol **Prediksi**
    4. Lihat hasil dan rekomendasi
    """)

# Load model
@st.cache_resource
def load_model():
    try:
        with open('diabetes_model.sav', 'rb') as file:
            model = pickle.load(file)
        return model, True
    except:
        return None, False

model_diabetes, model_loaded = load_model()

# ==================== BERANDA ====================
if menu == "🏠 Beranda":
    st.title("🏥 Aplikasi Prediksi Diabetes")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Selamat Datang!
        
        Aplikasi ini membantu mendeteksi risiko **diabetes** berdasarkan data medis pasien.
        
        **Fitur:**
        ✅ Prediksi risiko diabetes
        ✅ Analisis 8 parameter kesehatan
        ✅ Rekomendasi kesehatan personal
        
        **Parameter yang dianalisis:**
        1. Jumlah Kehamilan
        2. Kadar Glukosa
        3. Tekanan Darah
        4. Ketebalan Kulit
        5. Insulin
        6. BMI
        7. Riwayat Diabetes Keluarga
        8. Usia
        """)
    
    with col2:
        st.image("https://img.icons8.com/color/200/000000/health-checkup.png")
        st.write("**Status Model:**", model_status)

# ==================== PREDIKSI ====================
elif menu == "📊 Prediksi":
    st.title("🔍 Prediksi Diabetes")
    
    # Input data
    st.header("Data Pasien")
    
    col1, col2 = st.columns(2)
    
    with col1:
        kehamilan = st.slider("Jumlah Kehamilan", 0, 17, 3)
        glukosa = st.number_input("Glukosa (mg/dL)", 0, 200, 100)
        tekanan_darah = st.number_input("Tekanan Darah", 0, 130, 70)
        ketebalan_kulit = st.number_input("Ketebalan Kulit (mm)", 0, 100, 25)
    
    with col2:
        insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)
        riwayat_diabetes = st.number_input("Riwayat Diabetes", 0.0, 2.5, 0.3, step=0.001)
        usia = st.number_input("Usia (tahun)", 1, 100, 30)
    
    st.markdown("---")
    
    # Tombol prediksi - PERBAIKAN DI SINI
    if st.button("🚀 LAKUKAN PREDIKSI", type="primary", use_container_width=True):
        if model_diabetes is not None:  # PERUBAHAN: hanya cek model_diabetes
            try:
                # Format input
                input_data = np.array([[kehamilan, glukosa, tekanan_darah, ketebalan_kulit,
                                      insulin, bmi, riwayat_diabetes, usia]])
                
                # Prediksi
                hasil = model_diabetes.predict(input_data)[0]
                
                # Tampilkan hasil
                st.balloons()
                
                if hasil == 1:
                    st.markdown('<div class="positive">', unsafe_allow_html=True)
                    st.error("## ⚠️ HASIL: RISIKO DIABETES TINGGI")
                    st.write("""
                    **Rekomendasi:**
                    1. Konsultasi dokter segera
                    2. Cek gula darah rutin
                    3. Diet rendah gula
                    4. Olahraga teratur
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="negative">', unsafe_allow_html=True)
                    st.success("## ✅ HASIL: RISIKO DIABETES RENDAH")
                    st.write("""
                    **Pertahankan:**
                    1. Pola makan sehat
                    2. Aktivitas fisik
                    3. Cek kesehatan rutin
                    4. Hindari stres
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Analisis parameter
                st.subheader("📊 Analisis Parameter")
                
                analisis = []
                if glukosa >= 126:
                    analisis.append(f"❌ **Glukosa tinggi** ({glukosa} mg/dL)")
                elif glukosa >= 100:
                    analisis.append(f"⚠️ **Glukosa perbatasan** ({glukosa} mg/dL)")
                else:
                    analisis.append(f"✅ **Glukosa normal** ({glukosa} mg/dL)")
                    
                if bmi >= 30:
                    analisis.append(f"❌ **BMI obesitas** ({bmi})")
                elif bmi >= 25:
                    analisis.append(f"⚠️ **BMI overweight** ({bmi})")
                else:
                    analisis.append(f"✅ **BMI normal** ({bmi})")
                    
                # Tampilkan analisis
                for item in analisis:
                    st.write(item)
                    
            except Exception as e:
                st.error(f"Error prediksi: {str(e)}")
            
            # Tampilkan data input
            st.subheader("📋 Data yang Dimasukkan")
            data_dict = {
                'Parameter': ['Kehamilan', 'Glukosa', 'Tekanan Darah', 'Ketebalan Kulit',
                             'Insulin', 'BMI', 'Riwayat Diabetes', 'Usia'],
                'Nilai': [kehamilan, glukosa, tekanan_darah, ketebalan_kulit,
                         insulin, bmi, riwayat_diabetes, usia]
            }
            st.table(pd.DataFrame(data_dict))
            
        else:
            st.error("❌ Model tidak tersedia.")
            st.write("Pastikan file diabetes_model.sav ada di repository.")

# ==================== DATA ====================
elif menu == "📋 Data":
    st.title("📊 Dataset Diabetes")
    
    try:
        df = pd.read_csv("diabetes.csv")
        
        # Ubah nama kolom ke Indonesia
        df_indonesia = df.rename(columns={
            'Pregnancies': 'Kehamilan',
            'Glucose': 'Glukosa',
            'BloodPressure': 'Tekanan Darah',
            'SkinThickness': 'Ketebalan Kulit',
            'Insulin': 'Insulin',
            'BMI': 'BMI',
            'DiabetesPedigreeFunction': 'Riwayat Diabetes',
            'Age': 'Usia',
            'Outcome': 'Hasil (0=Sehat, 1=Diabetes)'
        })
        
        # Tampilkan data
        st.dataframe(df_indonesia, use_container_width=True)
        
        # Statistik
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Pasien", len(df))
        with col2:
            st.metric("Pasien Diabetes", df['Outcome'].sum())
        with col3:
            st.metric("% Diabetes", f"{df['Outcome'].mean()*100:.1f}%")
        
        # Grafik sederhana
        st.subheader("📈 Distribusi Data")
        
        tab1, tab2 = st.tabs(["Histogram", "Scatter"])
        
        with tab1:
            col_selected = st.selectbox(
                "Pilih parameter:",
                ['Glukosa', 'Usia', 'BMI', 'Tekanan Darah']
            )
            
            if col_selected == 'Glukosa':
                hist_data = df['Glucose']
            elif col_selected == 'Usia':
                hist_data = df['Age']
            elif col_selected == 'BMI':
                hist_data = df['BMI']
            else:
                hist_data = df['BloodPressure']
            
            st.bar_chart(pd.DataFrame({col_selected: hist_data}))
        
        with tab2:
            scatter_data = pd.DataFrame({
                'Glukosa': df['Glucose'],
                'BMI': df['BMI'],
                'Diabetes': df['Outcome'].map({0: 'Tidak', 1: 'Ya'})
            })
            st.scatter_chart(scatter_data, x='Glukosa', y='BMI', color='Diabetes')
            
    except Exception as e:
        st.error(f"Error loading data: {e}")

# ==================== INFO ====================
elif menu == "ℹ️ Info":
    st.title("ℹ️ Tentang Aplikasi")
    
    st.write("""
    ## Aplikasi Prediksi Diabetes
    
    **Deskripsi:**
    Aplikasi ini menggunakan model Machine Learning untuk memprediksi risiko diabetes
    berdasarkan parameter kesehatan pasien.
    
    **Teknologi:**
    - Python 3.9+
    - Streamlit untuk UI
    - Scikit-learn untuk ML
    - Pandas untuk data processing
    
    **Parameter:**
    1. Kehamilan
    2. Glukosa
    3. Tekanan Darah
    4. Ketebalan Kulit
    5. Insulin
    6. BMI
    7. Riwayat Diabetes Keluarga
    8. Usia
    
    **Disclaimer:**
    Hasil prediksi bersifat informatif dan tidak menggantikan diagnosis dokter.
    Selalu konsultasikan dengan tenaga medis profesional.
    """)
    
    st.markdown("---")
    st.write(f"**Versi:** 1.0.0")
    st.write(f"**Update terakhir:** {datetime.now().strftime('%d %B %Y')}")

# Footer
st.markdown("---")
st.caption("🩺 Aplikasi Prediksi Diabetes • Untuk edukasi kesehatan • © 2025")
