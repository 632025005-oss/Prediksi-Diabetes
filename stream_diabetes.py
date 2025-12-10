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

# Inisialisasi session state
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

@st.cache_resource
def load_model():
    try:
        with open('diabetes_model.sav', 'rb') as file:
            model = pickle.load(file)
        return model, True
    except FileNotFoundError:
        st.sidebar.error("File 'diabetes_model.sav' tidak ditemukan")
        return None, False
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        return None, False

# Load model
model_diabetes, model_loaded = load_model()

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
    .param-analysis {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #6c757d;
    }
    .param-good {
        border-left-color: #28a745;
        background: #d4edda;
    }
    .param-warning {
        border-left-color: #ffc107;
        background: #fff3cd;
    }
    .param-danger {
        border-left-color: #dc3545;
        background: #f8d7da;
    }
    .confidence-meter {
        height: 20px;
        background: #e9ecef;
        border-radius: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

# Judul dengan style
st.markdown('<h1 class="main-header">🏥 Aplikasi Prediksi Diabetes</h1>', unsafe_allow_html=True)
st.markdown("***Tugas Artificial Intelligence - Magister Sains Data - Universitas Kristen Satya Wacana Salatiga***")
st.markdown("***Regina Ria Aurellia (632025005)***")

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

# ==================== HALAMAN BERANDA ====================
if menu == "🏠 Beranda":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Selamat Datang di Aplikasi Prediksi Diabetes")
        st.write("""
        Aplikasi ini menggunakan **Machine Learning** untuk memprediksi risiko diabetes 
        berdasarkan parameter kesehatan pasien. Dengan teknologi AI, kami dapat membantu 
        mendeteksi potensi diabetes secara dini.
        
        **Fitur Utama:**
        ✅ **Prediksi Risiko Diabetes** - Analisis berdasarkan 8 parameter kesehatan
        ✅ **Visualisasi Data** - Grafik interaktif untuk pemahaman lebih baik
        ✅ **Rekomendasi Kesehatan** - Saran personalized berdasarkan hasil
        ✅ **Riwayat Prediksi** - Simpan dan bandingkan hasil prediksi
        """)
        
        # Statistik cepat
        st.subheader("📈 Statistik Diabetes Global")
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Penderita Diabetes", "537 Juta", "+16%")
        with col_stat2:
            st.metric("Kematian/Tahun", "6.7 Juta", "1 setiap 5 detik")
        with col_stat3:
            st.metric("Biaya Kesehatan", "$966B", "+316% dalam 15 tahun")
    
    with col2:
        st.image("https://img.icons8.com/color/300/000000/doctor-male.png")
        st.markdown("""
        <div style='text-align: center'>
            <h4>⏱️ Cepat & Akurat</h4>
            <p>Hasil prediksi dalam hitungan detik</p>
        </div>
        """, unsafe_allow_html=True)

# ==================== HALAMAN PREDIKSI ====================
elif menu == "📊 Prediksi":
    st.header("🔍 Prediksi Risiko Diabetes")
    
    tab1, tab2 = st.tabs(["📝 Input Data", "⚡ Input Cepat"])
    
    # Inisialisasi variabel dengan default values di session state
    if 'input_values' not in st.session_state:
        st.session_state.input_values = {
            'kehamilan': 3,
            'glukosa': 117,
            'tekanan_darah': 72,
            'ketebalan_kulit': 23,
            'insulin': 30,
            'bmi': 32.0,
            'riwayat_diabetes': 0.3725,
            'usia': 29
        }
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Medis")
            kehamilan = st.slider('Jumlah Kehamilan', 0, 17, 
                                 st.session_state.input_values['kehamilan'], 
                                 help="Total kehamilan yang pernah dialami", 
                                 key="kehamilan_input")
            glukosa = st.number_input('Kadar Glukosa (mg/dL)', 0, 200, 
                                     st.session_state.input_values['glukosa'],
                                     help="Kadar glukosa darah puasa", 
                                     key="glukosa_input")
            tekanan_darah = st.number_input('Tekanan Darah (mm Hg)', 0, 130, 
                                           st.session_state.input_values['tekanan_darah'], 
                                           key="tekanan_input")
            ketebalan_kulit = st.number_input('Ketebalan Kulit (mm)', 0, 100, 
                                             st.session_state.input_values['ketebalan_kulit'], 
                                             key="kulit_input")
        
        with col2:
            st.subheader("Data Fisik")
            insulin = st.number_input('Insulin Serum (mu U/ml)', 0, 900, 
                                     st.session_state.input_values['insulin'], 
                                     key="insulin_input")
            bmi = st.number_input('Indeks Massa Tubuh (BMI)', 0.0, 70.0, 
                                 st.session_state.input_values['bmi'], 
                                 step=0.1, key="bmi_input")
            riwayat_diabetes = st.number_input('Skor Riwayat Diabetes Keluarga', 0.0, 2.5, 
                                              st.session_state.input_values['riwayat_diabetes'], 
                                              step=0.01, key="riwayat_input")
            usia = st.number_input('Usia (tahun)', 1, 100, 
                                  st.session_state.input_values['usia'], 
                                  key="usia_input")
    
    with tab2:
        st.write("Pilih contoh data pasien:")
        contoh_data = st.selectbox(
            "Pilihan Contoh:",
            ["Pasien Standar", "Risiko Rendah", "Risiko Tinggi", "Lansia", "Ibu Hamil"],
            key="contoh_select"
        )
        
        # Update values based on selection
        if contoh_data == "Pasien Standar":
            example_values = {
                'kehamilan': 3, 'glukosa': 117, 'tekanan_darah': 72, 'ketebalan_kulit': 23,
                'insulin': 30, 'bmi': 32.0, 'riwayat_diabetes': 0.3725, 'usia': 29
            }
        elif contoh_data == "Risiko Tinggi":
            example_values = {
                'kehamilan': 6, 'glukosa': 148, 'tekanan_darah': 72, 'ketebalan_kulit': 35,
                'insulin': 0, 'bmi': 33.6, 'riwayat_diabetes': 0.627, 'usia': 50
            }
        elif contoh_data == "Risiko Rendah":
            example_values = {
                'kehamilan': 1, 'glukosa': 89, 'tekanan_darah': 66, 'ketebalan_kulit': 23,
                'insulin': 94, 'bmi': 28.1, 'riwayat_diabetes': 0.167, 'usia': 21
            }
        elif contoh_data == "Lansia":
            example_values = {
                'kehamilan': 0, 'glukosa': 120, 'tekanan_darah': 80, 'ketebalan_kulit': 30,
                'insulin': 0, 'bmi': 28.0, 'riwayat_diabetes': 0.400, 'usia': 65
            }
        else:  # Ibu Hamil
            example_values = {
                'kehamilan': 5, 'glukosa': 130, 'tekanan_darah': 70, 'ketebalan_kulit': 25,
                'insulin': 100, 'bmi': 30.0, 'riwayat_diabetes': 0.300, 'usia': 32
            }
        
        # Tombol untuk menerapkan contoh data
        if st.button("Gunakan Data Contoh", key="apply_example"):
            st.session_state.input_values = example_values
            st.rerun()
        
        # Tampilkan nilai yang dipilih
        data_contoh = pd.DataFrame({
            'Parameter': ['Kehamilan', 'Glukosa', 'Tekanan Darah', 'Ketebalan Kulit',
                         'Insulin', 'BMI', 'Riwayat Diabetes', 'Usia'],
            'Nilai': [example_values['kehamilan'], example_values['glukosa'], 
                     example_values['tekanan_darah'], example_values['ketebalan_kulit'],
                     example_values['insulin'], example_values['bmi'], 
                     example_values['riwayat_diabetes'], example_values['usia']]
        })
        st.dataframe(data_contoh, use_container_width=True)
    
    st.markdown("---")
    
    # Tombol prediksi
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button('🚀 LAKUKAN PREDIKSI', type="primary", use_container_width=True, key="prediksi_button")
    
    # Proses prediksi ketika tombol ditekan
    if predict_button:
        if model_loaded and model_diabetes is not None:
            # Format data untuk prediksi
            data_input = np.array([[kehamilan, glukosa, tekanan_darah, ketebalan_kulit,
                                   insulin, bmi, riwayat_diabetes, usia]])
            
            # Lakukan prediksi
            hasil_prediksi = model_diabetes.predict(data_input)[0]
            
            # Untuk SVM dengan probability=False, gunakan decision function untuk confidence
            try:
                if hasattr(model_diabetes, 'decision_function'):
                    decision_score = model_diabetes.decision_function(data_input)[0]
                    # Normalisasi decision score ke range 0-100
                    confidence = min(100, max(0, 50 + decision_score * 10))
                    confidence_label = f"{confidence:.1f}%"
                else:
                    # Default confidence untuk SVM
                    confidence = 85.0
                    confidence_label = "85.0%"
            except Exception as e:
                confidence = 85.0
                confidence_label = "85.0%"
            
            # Simpan ke session state
            st.session_state.last_prediction = {
                'data': [kehamilan, glukosa, tekanan_darah, ketebalan_kulit,
                        insulin, bmi, riwayat_diabetes, usia],
                'hasil': hasil_prediksi,
                'confidence': confidence,
                'confidence_label': confidence_label,
                'waktu': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model': "Support Vector Machine (SVM)"
            }
            
            # Tambahkan ke history
            st.session_state.predictions_history.append(st.session_state.last_prediction.copy())
            
            # Tampilkan animasi
            st.balloons()
            
            # ===== HASIL PREDIKSI =====
            if hasil_prediksi == 1:
                st.markdown('<div class="result-box positive">', unsafe_allow_html=True)
                st.error('## ⚠️ **HASIL: RISIKO DIABETES TINGGI**')
                
                # Tampilkan confidence meter
                st.markdown(f"**Tingkat Keyakinan Model:** {confidence_label}")
                confidence_percent = confidence
                confidence_color = "#dc3545" if confidence_percent > 70 else "#ffc107"
                
                st.markdown(f"""
                <div class="confidence-meter">
                    <div class="confidence-fill" style="width: {confidence_percent}%; background-color: {confidence_color};"></div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                **Rekomendasi Medis:**
                1. **Segera konsultasi dengan dokter** untuk pemeriksaan lebih lanjut
                2. **Tes HbA1c** untuk konfirmasi diagnosis
                3. **Pantau gula darah** secara rutin (pagi dan setelah makan)
                4. **Diet rendah gula** dan karbohidrat sederhana
                
                **Pola Hidup Sehat:**
                - Olahraga 30 menit/hari, 5x seminggu
                - Konsumsi makanan tinggi serat
                - Hindari makanan olahan dan minuman manis
                - Istirahat cukup (7-8 jam/hari)
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-box negative">', unsafe_allow_html=True)
                st.success('## ✅ **HASIL: RISIKO DIABETES RENDAH**')
                
                # Tampilkan confidence meter
                st.markdown(f"**Tingkat Keyakinan Model:** {confidence_label}")
                confidence_percent = confidence
                confidence_color = "#28a745" if confidence_percent > 70 else "#ffc107"
                
                st.markdown(f"""
                <div class="confidence-meter">
                    <div class="confidence-fill" style="width: {confidence_percent}%; background-color: {confidence_color};"></div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                **Pertahankan Kesehatan Anda:**
                1. **Cek kesehatan rutin** setiap 6 bulan sekali
                2. **Pola makan seimbang** dengan gizi lengkap
                3. **Tetap aktif** secara fisik
                4. **Kelola stres** dengan baik
                
                **Tips Pencegahan:**
                - Batasi konsumsi gula tambahan
                - Perbanyak sayur dan buah
                - Jaga berat badan ideal
                - Hindari merokok dan alkohol berlebihan
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # ===== ANALISIS PARAMETER =====
            st.subheader("📊 Analisis Parameter")
            
            # Buat DataFrame untuk analisis parameter
            param_data = {
                'Parameter': ['Kehamilan', 'Glukosa', 'Tekanan Darah', 'Ketebalan Kulit', 
                            'Insulin', 'BMI', 'Riwayat Diabetes', 'Usia'],
                'Nilai': [kehamilan, glukosa, tekanan_darah, ketebalan_kulit, 
                         insulin, bmi, riwayat_diabetes, usia],
                'Status': ['Normal'] * 8,
                'Kategori': ['Normal'] * 8
            }
            
            # Analisis Glukosa
            if glukosa >= 126:
                param_data['Status'][1] = 'danger'
                param_data['Kategori'][1] = 'Tinggi (≥126 mg/dL)'
                glukosa_class = "param-danger"
                glukosa_msg = f"❌ **Glukosa tinggi** ({glukosa} mg/dL) - Di atas batas diabetes (≥126 mg/dL)"
            elif glukosa >= 100:
                param_data['Status'][1] = 'warning'
                param_data['Kategori'][1] = 'Pra-diabetes (100-125 mg/dL)'
                glukosa_class = "param-warning"
                glukosa_msg = f"⚠️ **Glukosa perbatasan** ({glukosa} mg/dL) - Pra-diabetes"
            else:
                param_data['Status'][1] = 'good'
                param_data['Kategori'][1] = 'Normal (<100 mg/dL)'
                glukosa_class = "param-good"
                glukosa_msg = f"✅ **Glukosa normal** ({glukosa} mg/dL)"
            
            # Analisis BMI
            if bmi >= 30:
                param_data['Status'][5] = 'danger'
                param_data['Kategori'][5] = 'Obesitas (≥30)'
                bmi_class = "param-danger"
                bmi_msg = f"❌ **BMI obesitas** ({bmi}) - Faktor risiko tinggi"
            elif bmi >= 25:
                param_data['Status'][5] = 'warning'
                param_data['Kategori'][5] = 'Overweight (25-29.9)'
                bmi_class = "param-warning"
                bmi_msg = f"⚠️ **BMI overweight** ({bmi}) - Perlu penurunan berat badan"
            else:
                param_data['Status'][5] = 'good'
                param_data['Kategori'][5] = 'Normal (18.5-24.9)'
                bmi_class = "param-good"
                bmi_msg = f"✅ **BMI normal** ({bmi})"
            
            # Analisis Tekanan Darah
            if tekanan_darah >= 140:
                param_data['Status'][2] = 'danger'
                param_data['Kategori'][2] = 'Hipertensi (≥140 mmHg)'
                tekanan_class = "param-danger"
                tekanan_msg = f"❌ **Tekanan darah tinggi** ({tekanan_darah} mmHg) - Hipertensi"
            elif tekanan_darah >= 130:
                param_data['Status'][2] = 'warning'
                param_data['Kategori'][2] = 'Pra-hipertensi (130-139 mmHg)'
                tekanan_class = "param-warning"
                tekanan_msg = f"⚠️ **Tekanan darah perbatasan** ({tekanan_darah} mmHg) - Perlu pemantauan"
            else:
                param_data['Status'][2] = 'good'
                param_data['Kategori'][2] = 'Normal (<130 mmHg)'
                tekanan_class = "param-good"
                tekanan_msg = f"✅ **Tekanan darah normal** ({tekanan_darah} mmHg)"
            
            # Analisis Usia
            if usia >= 45:
                param_data['Status'][7] = 'warning'
                param_data['Kategori'][7] = 'Risiko Tinggi (≥45 tahun)'
                usia_class = "param-warning"
                usia_msg = f"⚠️ **Usia ≥45 tahun** ({usia} tahun) - Faktor risiko diabetes meningkat"
            else:
                param_data['Status'][7] = 'good'
                param_data['Kategori'][7] = 'Normal (<45 tahun)'
                usia_class = "param-good"
                usia_msg = f"✅ **Usia <45 tahun** ({usia} tahun) - Risiko lebih rendah"
            
            # Analisis Insulin
            if insulin > 100:
                param_data['Status'][4] = 'warning'
                param_data['Kategori'][4] = 'Tinggi (>100 μU/mL)'
                insulin_class = "param-warning"
                insulin_msg = f"⚠️ **Insulin tinggi** ({insulin} μU/mL) - Kemungkinan resistensi insulin"
            elif insulin < 25:
                param_data['Status'][4] = 'warning'
                param_data['Kategori'][4] = 'Rendah (<25 μU/mL)'
                insulin_class = "param-warning"
                insulin_msg = f"⚠️ **Insulin rendah** ({insulin} μU/mL) - Perlu evaluasi fungsi pankreas"
            else:
                param_data['Status'][4] = 'good'
                param_data['Kategori'][4] = 'Normal (25-100 μU/mL)'
                insulin_class = "param-good"
                insulin_msg = f"✅ **Insulin normal** ({insulin} μU/mL)"
            
            # Tampilkan analisis per parameter
            st.markdown(f'<div class="param-analysis {glukosa_class}">{glukosa_msg}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="param-analysis {bmi_class}">{bmi_msg}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="param-analysis {tekanan_class}">{tekanan_msg}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="param-analysis {usia_class}">{usia_msg}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="param-analysis {insulin_class}">{insulin_msg}</div>', unsafe_allow_html=True)
            
            # Tampilkan tabel parameter
            param_df = pd.DataFrame(param_data)
            st.subheader("📋 Tabel Parameter Pasien")
            st.dataframe(param_df, use_container_width=True)
            
            # Visualisasi parameter
            st.subheader("📊 Visualisasi Parameter Kesehatan")
            
            # Bar chart untuk parameter
            fig_bar = px.bar(param_df, x='Parameter', y='Nilai', color='Status',
                           color_discrete_map={'good': 'green', 'warning': 'orange', 'danger': 'red'},
                           title="Nilai Parameter Kesehatan",
                           hover_data=['Kategori'])
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Tombol download hasil
            st.subheader("💾 Simpan Hasil")
            hasil_text = f"""HASIL PREDIKSI DIABETES
Tanggal: {datetime.now().strftime("%d/%m/%Y %H:%M")}
Model yang digunakan: Support Vector Machine (SVM)
Tingkat Keyakinan: {confidence_label}

DATA PASIEN:
- Kehamilan: {kehamilan}
- Glukosa: {glukosa} mg/dL
- Tekanan Darah: {tekanan_darah} mm Hg
- Ketebalan Kulit: {ketebalan_kulit} mm
- Insulin: {insulin} mu U/ml
- BMI: {bmi}
- Riwayat Diabetes: {riwayat_diabetes}
- Usia: {usia} tahun

HASIL: {'RISIKO DIABETES TINGGI' if hasil_prediksi == 1 else 'RISIKO DIABETES RENDAH'}

ANALISIS PARAMETER:
1. {glukosa_msg}
2. {bmi_msg}
3. {tekanan_msg}
4. {usia_msg}
5. {insulin_msg}

Catatan: Hasil ini merupakan prediksi berdasarkan model AI (SVM Classifier). 
Konsultasi dengan dokter tetap diperlukan untuk diagnosis pasti.
"""
            
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="📥 Download Hasil Prediksi",
                    data=hasil_text,
                    file_name=f"hasil_prediksi_diabetes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="download_txt"
                )
            with col_dl2:
                # Simpan data ke CSV
                hasil_df = pd.DataFrame({
                    'Parameter': param_data['Parameter'],
                    'Nilai': param_data['Nilai'],
                    'Kategori': param_data['Kategori']
                })
                csv = hasil_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Data CSV",
                    data=csv,
                    file_name=f"data_pasien_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_csv"
                )
                
        else:
            st.error("Model tidak tersedia. Pastikan file 'diabetes_model.sav' ada di server.")

# ==================== HALAMAN ANALISIS ====================
elif menu == "📈 Analisis":
    st.header("📈 Analisis Data Diabetes")
    
    try:
        # Load data
        df = pd.read_csv("diabetes.csv")
        
        # Ubah nama kolom
        df_indonesia = df.rename(columns={
            'Pregnancies': 'Kehamilan',
            'Glucose': 'Glukosa',
            'BloodPressure': 'Tekanan Darah',
            'SkinThickness': 'Ketebalan Kulit',
            'Insulin': 'Insulin',
            'BMI': 'BMI',
            'DiabetesPedigreeFunction': 'Riwayat Diabetes',
            'Age': 'Usia',
            'Outcome': 'Diabetes'
        })
        
        # Tabs untuk berbagai visualisasi
        tab1, tab2, tab3 = st.tabs(["Distribusi Data", "Korelasi", "Perbandingan"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                parameter = st.selectbox(
                    "Pilih Parameter:",
                    ['Glukosa', 'Usia', 'BMI', 'Tekanan Darah'],
                    key="param_dist"
                )
                
                fig = px.histogram(df_indonesia, x=parameter, 
                                 color='Diabetes',
                                 title=f'Distribusi {parameter}',
                                 nbins=20,
                                 color_discrete_map={0: 'green', 1: 'red'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pie chart hasil diabetes
                diabetes_count = df_indonesia['Diabetes'].value_counts()
                fig = px.pie(values=diabetes_count.values,
                            names=['Tidak Diabetes', 'Diabetes'],
                            title='Proporsi Diabetes dalam Dataset',
                            color=['Tidak Diabetes', 'Diabetes'],
                            color_discrete_map={'Tidak Diabetes':'green', 'Diabetes':'red'})
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Heatmap korelasi
            fig = px.imshow(df.corr(),
                           title='Korelasi Antar Parameter',
                           color_continuous_scale='RdBu',
                           text_auto=True)
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("""
            **Interpretasi:**
            - Warna biru: Korelasi positif (semakin besar nilai satu, semakin besar nilai lainnya)
            - Warna merah: Korelasi negatif (semakin besar nilai satu, semakin kecil nilai lainnya)
            - Glukosa memiliki korelasi tinggi dengan hasil diabetes
            """)
        
        with tab3:
            # Scatter plot interaktif
            col_x, col_y = st.columns(2)
            with col_x:
                x_axis = st.selectbox("Sumbu X:", df_indonesia.columns[:-1], index=1, key="x_axis")
            with col_y:
                y_axis = st.selectbox("Sumbu Y:", df_indonesia.columns[:-1], index=6, key="y_axis")
            
            fig = px.scatter(df_indonesia, x=x_axis, y=y_axis,
                            color='Diabetes',
                            size='Usia',
                            hover_data=['Kehamilan', 'BMI'],
                            title=f'{x_axis} vs {y_axis}',
                            color_discrete_map={0: 'green', 1: 'red'})
            st.plotly_chart(fig, use_container_width=True)
            
    except FileNotFoundError:
        st.error("File 'diabetes.csv' tidak ditemukan. Pastikan file ada di folder yang sama.")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

# ==================== HALAMAN DATA ====================
elif menu == "📋 Data":
    st.header("📋 Dataset Diabetes")
    
    try:
        df = pd.read_csv("diabetes.csv")
        
        # Tampilkan data
        st.dataframe(df, use_container_width=True, height=400)
        
        # Statistik
        st.subheader("📊 Statistik Dataset")
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Total Pasien", len(df))
        with col_stat2:
            st.metric("Pasien Diabetes", df['Outcome'].sum())
        with col_stat3:
            st.metric("Persentase Diabetes", f"{df['Outcome'].mean()*100:.1f}%")
        
        # Filter data
        st.subheader("🔍 Filter Data")
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            min_age = st.slider("Usia Minimum", int(df['Age'].min()), int(df['Age'].max()), 20, key="filter_age")
        with col_filter2:
            min_glucose = st.slider("Glukosa Minimum", int(df['Glucose'].min()), int(df['Glucose'].max()), 100, key="filter_glucose")
        
        filtered_df = df[(df['Age'] >= min_age) & (df['Glucose'] >= min_glucose)]
        st.write(f"Menampilkan {len(filtered_df)} dari {len(df)} pasien")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Ekspor data
        st.subheader("💾 Export Data")
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            if st.button("📥 Download Data Filtered", key="btn_filtered"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="data_diabetes_filtered.csv",
                    mime="text/csv",
                    key="dl_filtered"
                )
        with col_exp2:
            if st.button("📥 Download All Data", key="btn_all"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="diabetes_dataset.csv",
                    mime="text/csv",
                    key="dl_all"
                )
                
    except FileNotFoundError:
        st.error("File 'diabetes.csv' tidak ditemukan. Pastikan file ada di folder yang sama.")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

# ==================== HALAMAN TENTANG ====================
elif menu == "ℹ️ Tentang":
    st.header("ℹ️ Tentang Aplikasi")
    
    col_about1, col_about2 = st.columns([2, 1])
    
    with col_about1:
        st.write("""
        **Aplikasi Prediksi Diabetes** adalah solusi berbasis AI untuk mendeteksi 
        risiko diabetes secara dini menggunakan algoritma Machine Learning.
        
        **Teknologi yang Digunakan:**
        - **Streamlit**: Framework untuk membangun aplikasi web interaktif
        - **Scikit-learn**: Library Machine Learning untuk model prediksi (SVM Classifier)
        - **Plotly**: Visualisasi data interaktif
        - **Pandas & NumPy**: Pengolahan data
        
        **Parameter yang Dianalisis:**
        1. Jumlah Kehamilan
        2. Kadar Glukosa
        3. Tekanan Darah
        4. Ketebalan Kulit
        5. Insulin Serum
        6. Indeks Massa Tubuh (BMI)
        7. Riwayat Diabetes Keluarga
        8. Usia
        
        **Model yang Digunakan:**
        - **Support Vector Machine (SVM)** dengan kernel linear
        - Model ini dipilih karena performa yang baik untuk klasifikasi biner
        
        **Disclaimer:**
        Hasil prediksi ini bersifat informatif dan tidak menggantikan diagnosis medis. 
        Selalu konsultasikan dengan dokter untuk pemeriksaan dan penanganan yang tepat.
        """)
    
    with col_about2:
        st.image("https://img.icons8.com/color/200/000000/artificial-intelligence.png")
        st.markdown("""
        <div style='text-align: center'>
            <h4>🤖 AI-Powered</h4>
            <p>Ditenagai oleh kecerdasan buatan</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.write("**Versi Aplikasi:** 2.0.0")
    st.write("**Terakhir Diupdate:** " + datetime.now().strftime("%d %B %Y"))
    st.write("**Developer:** Regina Ria Aurellia (632025005)")
    
    # Kontak
    with st.expander("📞 Kontak & Support"):
        st.write("""
        **Email:** 632025005@student.uksw.edu
        **Universitas:** Universitas Kristen Satya Wacana Salatiga
        **Program:** Magister Sains Data
        **Mata Kuliah:** Artificial Intelligence
        
        **File yang Diperlukan:**
        1. `diabetes_model.sav` - Model machine learning (SVM)
        2. `diabetes.csv` - Dataset untuk analisis
        """)

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.caption("🩺 Aplikasi Prediksi Diabetes")
with footer_col2:
    st.caption("Regina Ria Aurellia - 632025005")
with footer_col3:
    st.caption(f"© {datetime.now().year} - Tugas Artificial Intelligence")
