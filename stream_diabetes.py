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

# === TAMBAHKAN INI DI BAWAH IMPORTS ===
# Inisialisasi session state untuk menyimpan data prediksi
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# === LOAD MODEL YANG DIPERBAIKI ===
@st.cache_resource
def load_model():
    """Load model dengan error handling"""
    try:
        with open('diabetes_model.sav', 'rb') as file:
            model = pickle.load(file)
        
        # Debug info (opsional)
        st.sidebar.success("✅ Model berhasil dimuat")
        if hasattr(model, '__class__'):
            st.sidebar.write(f"Model type: {model.__class__.__name__}")
        
        return model
    except FileNotFoundError:
        st.sidebar.error("❌ File model tidak ditemukan")
        st.sidebar.info("Pastikan 'diabetes_model.sav' ada di folder yang sama")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        return None

# Load model
model_diabetes = load_model()
model_loaded = model_diabetes is not None

# CSS kustom (tetap sama)
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
    .stButton>button {
        width: 100%;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Judul
st.markdown('<h1 class="main-header">🏥 Aplikasi Prediksi Diabetes</h1>', unsafe_allow_html=True)
st.markdown("***Tugas Artificial Intelligence - Magister Sains Data - Universitas Kristen Satya Wacana Salatiga***")
st.markdown("***Regina Ria Aurellia (632025005)***")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/diabetes.png", width=100)
    st.title("Navigasi")
    
    menu = st.radio(
        "Pilih Menu:",
        ["🏠 Beranda", "📊 Prediksi", "📈 Analisis", "📋 Data", "ℹ️ Tentang"]
    )
    
    # Refresh button untuk testing
    if st.button("🔄 Refresh Model Cache"):
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    st.info("""
    **Cara Penggunaan:**
    1. Pilih menu **Prediksi**
    2. Isi data pasien
    3. Klik tombol **Prediksi**
    4. Lihat hasil dan rekomendasi
    """)

# ==================== HALAMAN PREDIKSI (DIPERBAIKI) ====================
elif menu == "📊 Prediksi":
    st.header("🔍 Prediksi Risiko Diabetes")
    
    tab1, tab2 = st.tabs(["📝 Input Data", "⚡ Input Cepat"])
    
    # Inisialisasi variabel
    kehamilan, glukosa, tekanan_darah, ketebalan_kulit = 3, 117, 72, 23
    insulin, bmi, riwayat_diabetes, usia = 30, 32.0, 0.3725, 29
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Medis")
            kehamilan = st.slider('Jumlah Kehamilan', 0, 17, 3, 
                                 help="Total kehamilan yang pernah dialami",
                                 key="kehamilan_input")
            glukosa = st.number_input('Kadar Glukosa (mg/dL)', 0, 200, 117,
                                     help="Kadar glukosa darah puasa",
                                     key="glukosa_input")
            tekanan_darah = st.number_input('Tekanan Darah (mm Hg)', 0, 130, 72,
                                           key="tekanan_darah_input")
            ketebalan_kulit = st.number_input('Ketebalan Kulit (mm)', 0, 100, 23,
                                             key="ketebalan_kulit_input")
        
        with col2:
            st.subheader("Data Fisik")
            insulin = st.number_input('Insulin Serum (mu U/ml)', 0, 900, 30,
                                     key="insulin_input")
            bmi = st.number_input('Indeks Massa Tubuh (BMI)', 0.0, 70.0, 32.0, step=0.1,
                                 key="bmi_input")
            riwayat_diabetes = st.number_input('Skor Riwayat Diabetes Keluarga', 0.0, 2.5, 0.3725, step=0.01,
                                              key="riwayat_input")
            usia = st.number_input('Usia (tahun)', 1, 100, 29,
                                  key="usia_input")
    
    with tab2:
        st.write("Pilih contoh data pasien:")
        contoh_data = st.selectbox(
            "Pilihan Contoh:",
            ["Pasien Standar", "Risiko Rendah", "Risiko Tinggi", "Lansia", "Ibu Hamil"],
            key="contoh_data_select"
        )
        
        # Reset nilai berdasarkan pilihan
        if contoh_data == "Pasien Standar":
            kehamilan, glukosa, tekanan_darah, ketebalan_kulit = 3, 117, 72, 23
            insulin, bmi, riwayat_diabetes, usia = 30, 32.0, 0.3725, 29
        elif contoh_data == "Risiko Tinggi":
            kehamilan, glukosa, tekanan_darah, ketebalan_kulit = 6, 148, 72, 35
            insulin, bmi, riwayat_diabetes, usia = 0, 33.6, 0.627, 50
        elif contoh_data == "Risiko Rendah":
            kehamilan, glukosa, tekanan_darah, ketebalan_kulit = 1, 89, 66, 23
            insulin, bmi, riwayat_diabetes, usia = 94, 28.1, 0.167, 21
        elif contoh_data == "Lansia":
            kehamilan, glukosa, tekanan_darah, ketebalan_kulit = 0, 120, 80, 30
            insulin, bmi, riwayat_diabetes, usia = 0, 28.0, 0.400, 65
        else:  # Ibu Hamil
            kehamilan, glukosa, tekanan_darah, ketebalan_kulit = 5, 130, 70, 25
            insulin, bmi, riwayat_diabetes, usia = 100, 30.0, 0.300, 32
            
        # Tampilkan nilai yang dipilih
        data_contoh = pd.DataFrame({
            'Parameter': ['Kehamilan', 'Glukosa', 'Tekanan Darah', 'Ketebalan Kulit',
                         'Insulin', 'BMI', 'Riwayat Diabetes', 'Usia'],
            'Nilai': [kehamilan, glukosa, tekanan_darah, ketebalan_kulit,
                     insulin, bmi, riwayat_diabetes, usia]
        })
        st.dataframe(data_contoh, use_container_width=True)
        
        # Update nilai di tab input dengan session state
        st.session_state.kehamilan_input = kehamilan
        st.session_state.glukosa_input = glukosa
        st.session_state.tekanan_darah_input = tekanan_darah
        st.session_state.ketebalan_kulit_input = ketebalan_kulit
        st.session_state.insulin_input = insulin
        st.session_state.bmi_input = bmi
        st.session_state.riwayat_input = riwayat_diabetes
        st.session_state.usia_input = usia
    
    st.markdown("---")
    
    # Tombol prediksi dengan key yang unik
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button('🚀 LAKUKAN PREDIKSI', 
                                 type="primary", 
                                 use_container_width=True,
                                 key="predict_button")
    
    # Logika prediksi (dipisah dari button untuk menghindari rerun)
    if predict_button:
        if model_loaded:
            # Format data
            data_input = np.array([[kehamilan, glukosa, tekanan_darah, ketebalan_kulit,
                                   insulin, bmi, riwayat_diabetes, usia]])
            
            # Prediksi
            hasil_prediksi = model_diabetes.predict(data_input)[0]
            
            # Simpan ke session state
            st.session_state.last_prediction = {
                'data': [kehamilan, glukosa, tekanan_darah, ketebalan_kulit,
                        insulin, bmi, riwayat_diabetes, usia],
                'hasil': hasil_prediksi,
                'waktu': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model': type(model_diabetes).__name__
            }
            
            # Simpan ke history
            st.session_state.predictions_history.append(
                st.session_state.last_prediction.copy()
            )
            
            # Tampilkan hasil
            st.balloons()
            
            # Tampilkan hasil prediksi
            st.markdown("## 📊 Hasil Prediksi")
            
            if hasil_prediksi == 1:
                st.markdown('<div class="result-box positive">', unsafe_allow_html=True)
                st.error('## ⚠️ **HASIL: RISIKO DIABETES TINGGI**')
                st.markdown("""
                **Rekomendasi Medis:**
                1. **Segera konsultasi dengan dokter** untuk pemeriksaan lebih lanjut
                2. **Tes HbA1c** untuk konfirmasi diagnosis
                3. **Pantau gula darah** secara rutin
                4. **Diet rendah gula** dan karbohidrat sederhana
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-box negative">', unsafe_allow_html=True)
                st.success('## ✅ **HASIL: RISIKO DIABETES RENDAH**')
                st.markdown("""
                **Pertahankan Kesehatan Anda:**
                1. **Cek kesehatan rutin** setiap 6 bulan
                2. **Pola makan seimbang** dengan gizi lengkap
                3. **Tetap aktif** secara fisik
                4. **Kelola stres** dengan baik
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # === ANALISIS PARAMETER YANG AKAN BERUBAH ===
            st.markdown("---")
            st.subheader("📋 Analisis Parameter")
            
            # Buat DataFrame untuk analisis
            params_df = pd.DataFrame({
                'Parameter': ['Kehamilan', 'Glukosa', 'Tekanan Darah', 'Ketebalan Kulit',
                            'Insulin', 'BMI', 'Riwayat Diabetes', 'Usia'],
                'Nilai': [kehamilan, glukosa, tekanan_darah, ketebalan_kulit,
                         insulin, bmi, riwayat_diabetes, usia],
                'Status': ['Normal', 'Normal', 'Normal', 'Normal', 
                          'Normal', 'Normal', 'Normal', 'Normal']
            })
            
            # Analisis masing-masing parameter (INI YANG AKAN BERUBAH)
            analisis_list = []
            
            # Analisis Glukosa
            if glukosa >= 126:
                params_df.loc[1, 'Status'] = 'Tinggi'
                analisis_list.append(f"❌ **Glukosa tinggi** ({glukosa} mg/dL) - Risiko diabetes meningkat")
            elif glukosa >= 100:
                params_df.loc[1, 'Status'] = 'Perbatasan'
                analisis_list.append(f"⚠️ **Glukosa perbatasan** ({glukosa} mg/dL) - Perlu pemantauan")
            else:
                params_df.loc[1, 'Status'] = 'Normal'
                analisis_list.append(f"✅ **Glukosa normal** ({glukosa} mg/dL)")
            
            # Analisis BMI
            if bmi >= 30:
                params_df.loc[5, 'Status'] = 'Obesitas'
                analisis_list.append(f"❌ **BMI obesitas** ({bmi}) - Faktor risiko tinggi")
            elif bmi >= 25:
                params_df.loc[5, 'Status'] = 'Overweight'
                analisis_list.append(f"⚠️ **BMI overweight** ({bmi}) - Perlu penurunan berat badan")
            else:
                params_df.loc[5, 'Status'] = 'Normal'
                analisis_list.append(f"✅ **BMI normal** ({bmi})")
            
            # Analisis Usia
            if usia >= 45:
                params_df.loc[7, 'Status'] = 'Risiko Tinggi'
                analisis_list.append(f"⚠️ **Usia ≥45 tahun** ({usia} tahun) - Faktor risiko diabetes")
            else:
                params_df.loc[7, 'Status'] = 'Normal'
                analisis_list.append(f"✅ **Usia <45 tahun** ({usia} tahun)")
            
            # Analisis Tekanan Darah
            if tekanan_darah >= 140:
                params_df.loc[2, 'Status'] = 'Hipertensi'
                analisis_list.append(f"❌ **Tekanan darah tinggi** ({tekanan_darah} mmHg)")
            elif tekanan_darah >= 130:
                params_df.loc[2, 'Status'] = 'Pra-hipertensi'
                analisis_list.append(f"⚠️ **Tekanan darah perbatasan** ({tekanan_darah} mmHg)")
            else:
                params_df.loc[2, 'Status'] = 'Normal'
                analisis_list.append(f"✅ **Tekanan darah normal** ({tekanan_darah} mmHg)")
            
            # Tampilkan tabel parameter dengan warna
            def color_status(val):
                if val == 'Normal':
                    return 'color: green; font-weight: bold'
                elif val in ['Perbatasan', 'Overweight', 'Pra-hipertensi']:
                    return 'color: orange; font-weight: bold'
                else:
                    return 'color: red; font-weight: bold'
            
            styled_df = params_df.style.applymap(color_status, subset=['Status'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Tampilkan analisis point-by-point
            st.subheader("📝 Rekomendasi Berdasarkan Parameter:")
            for analisis in analisis_list:
                st.write(analisis)
            
            # Visualisasi parameter
            st.subheader("📊 Visualisasi Parameter")
            fig_bar = px.bar(params_df, x='Parameter', y='Nilai', 
                           color='Status',
                           color_discrete_map={
                               'Normal': 'green',
                               'Perbatasan': 'orange',
                               'Overweight': 'orange',
                               'Pra-hipertensi': 'orange',
                               'Tinggi': 'red',
                               'Obesitas': 'red',
                               'Hipertensi': 'red',
                               'Risiko Tinggi': 'red'
                           })
            st.plotly_chart(fig_bar, use_container_width=True)
            
        else:
            st.error("❌ Model tidak dapat dimuat. Pastikan file 'diabetes_model.sav' tersedia.")

# ... (halaman lainnya tetap sama)
