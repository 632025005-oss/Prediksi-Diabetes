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

# Load model dengan debugging
@st.cache_resource
def load_model():
    try:
        with open('diabetes_model.sav', 'rb') as file:
            model = pickle.load(file)
        
        if DEBUG:
            st.sidebar.write("🔍 Model Info:")
            st.sidebar.write(f"Type: {type(model)}")
            if hasattr(model, '__class__'):
                st.sidebar.write(f"Class: {model.__class__.__name__}")
            if hasattr(model, 'feature_names_in_'):
                st.sidebar.write(f"Features: {model.feature_names_in_}")
        
        return model, True
    except Exception as e:
        if DEBUG:
            st.sidebar.error(f"Error detail: {str(e)}")
        return None, False

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
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Medis")
            kehamilan = st.slider('Jumlah Kehamilan', 0, 17, 3, 
                                 help="Total kehamilan yang pernah dialami")
            glukosa = st.number_input('Kadar Glukosa (mg/dL)', 0, 200, 117,
                                     help="Kadar glukosa darah puasa")
            tekanan_darah = st.number_input('Tekanan Darah (mm Hg)', 0, 130, 72)
            ketebalan_kulit = st.number_input('Ketebalan Kulit (mm)', 0, 100, 23)
        
        with col2:
            st.subheader("Data Fisik")
            insulin = st.number_input('Insulin Serum (mu U/ml)', 0, 900, 30)
            bmi = st.number_input('Indeks Massa Tubuh (BMI)', 0.0, 70.0, 32.0, step=0.1)
            riwayat_diabetes = st.number_input('Skor Riwayat Diabetes Keluarga', 0.0, 2.5, 0.3725, step=0.01)
            usia = st.number_input('Usia (tahun)', 1, 100, 29)
    
    with tab2:
        st.write("Pilih contoh data pasien:")
        contoh_data = st.selectbox(
            "Pilihan Contoh:",
            ["Pasien Standar", "Risiko Rendah", "Risiko Tinggi", "Lansia", "Ibu Hamil"]
        )
        
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
    
    st.markdown("---")
    
    # Tombol prediksi
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button('🚀 LAKUKAN PREDIKSI', type="primary", use_container_width=True):
            if model_loaded:
                # Format data
                data_input = np.array([[kehamilan, glukosa, tekanan_darah, ketebalan_kulit,
                                       insulin, bmi, riwayat_diabetes, usia]])
                
                # Prediksi
                hasil_prediksi = model_diabetes.predict(data_input)[0]
                
                # Simpan ke session state untuk halaman lain
                st.session_state['last_prediction'] = {
                    'data': [kehamilan, glukosa, tekanan_darah, ketebalan_kulit,
                            insulin, bmi, riwayat_diabetes, usia],
                    'hasil': hasil_prediksi,
                    'waktu': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'model': type(model_diabetes).__name__
                }
                
                # Tampilkan hasil
                st.balloons()  # Animasi
                
                if hasil_prediksi == 1:
                    st.markdown('<div class="result-box positive">', unsafe_allow_html=True)
                    st.error('## ⚠️ **HASIL: RISIKO DIABETES TINGGI**')
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
                
                # Tampilkan parameter dalam bentuk chart
                st.subheader("📊 Visualisasi Parameter Kesehatan")
                
                # Radar chart
                categories = ['Kehamilan', 'Glukosa', 'Tekanan Darah', 'Ketebalan Kulit',
                            'Insulin', 'BMI', 'Riwayat Diabetes', 'Usia']
                values = [kehamilan/17*100, glukosa/200*100, tekanan_darah/130*100,
                         ketebalan_kulit/100*100, insulin/900*100, bmi/70*100,
                         riwayat_diabetes/2.5*100, usia/100*100]
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    line_color='blue'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=False,
                    title="Profil Kesehatan Pasien"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tombol download hasil
                hasil_text = f"""
                HASIL PREDIKSI DIABETES
                Tanggal: {datetime.now().strftime("%d/%m/%Y %H:%M")}
                
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
                
                Catatan: Hasil ini merupakan prediksi berdasarkan model AI. 
                Konsultasi dengan dokter tetap diperlukan untuk diagnosis pasti.
                """

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
                
                st.download_button(
                    label="📥 Download Hasil Prediksi",
                    data=hasil_text,
                    file_name=f"hasil_prediksi_diabetes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
            else:
                st.error("Model tidak tersedia. Pastikan file model ada di server.")

# ==================== HALAMAN ANALISIS ====================
elif menu == "📈 Analisis":
    st.header("📈 Analisis Data Diabetes")
    
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
                ['Glukosa', 'Usia', 'BMI', 'Tekanan Darah']
            )
            
            fig = px.histogram(df_indonesia, x=parameter, 
                             color='Diabetes',
                             title=f'Distribusi {parameter}',
                             nbins=20)
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
                       color_continuous_scale='RdBu')
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("""
        **Interpretasi:**
        - Warna biru: Korelasi positif (semakin besar nilai satu, semakin besar nilai lainnya)
        - Warna merah: Korelasi negatif (semakin besar nilai satu, semakin kecil nilai lainnya)
        - Glukosa memiliki korelasi tinggi dengan hasil diabetes
        """)
    
    with tab3:
        # Scatter plot interaktif
        x_axis = st.selectbox("Sumbu X:", df_indonesia.columns[:-1], index=1)
        y_axis = st.selectbox("Sumbu Y:", df_indonesia.columns[:-1], index=6)
        
        fig = px.scatter(df_indonesia, x=x_axis, y=y_axis,
                        color='Diabetes',
                        size='Usia',
                        hover_data=['Kehamilan', 'BMI'],
                        title=f'{x_axis} vs {y_axis}')
        st.plotly_chart(fig, use_container_width=True)

# ==================== HALAMAN DATA ====================
elif menu == "📋 Data":
    st.header("📋 Dataset Diabetes")
    
    df = pd.read_csv("diabetes.csv")
    
    # Tampilkan data
    st.dataframe(df, use_container_width=True)
    
    # Statistik
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pasien", len(df))
    with col2:
        st.metric("Pasien Diabetes", df['Outcome'].sum())
    with col3:
        st.metric("Persentase Diabetes", f"{df['Outcome'].mean()*100:.1f}%")
    
    # Filter data
    st.subheader("Filter Data")
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        min_age = st.slider("Usia Minimum", int(df['Age'].min()), int(df['Age'].max()), 20)
    with col_filter2:
        min_glucose = st.slider("Glukosa Minimum", int(df['Glucose'].min()), int(df['Glucose'].max()), 100)
    
    filtered_df = df[(df['Age'] >= min_age) & (df['Glucose'] >= min_glucose)]
    st.write(f"Menampilkan {len(filtered_df)} dari {len(df)} pasien")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Ekspor data
    if st.button("📥 Download Data Filtered"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="data_diabetes_filtered.csv",
            mime="text/csv"
        )

# ==================== HALAMAN TENTANG ====================
else:
    st.header("ℹ️ Tentang Aplikasi")
    
    col_about1, col_about2 = st.columns([2, 1])
    
    with col_about1:
        st.write("""
        **Aplikasi Prediksi Diabetes** adalah solusi berbasis AI untuk mendeteksi 
        risiko diabetes secara dini menggunakan algoritma Machine Learning.
        
        **Teknologi yang Digunakan:**
        - **Streamlit**: Framework untuk membangun aplikasi web interaktif
        - **Scikit-learn**: Library Machine Learning untuk model prediksi
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
    st.write("**Developer:** Tim Prediksi Diabetes")
    
    # Kontak
    with st.expander("📞 Kontak & Support"):
        st.write("""
        **Email:** support@prediksidiabetes.com
        **Website:** www.prediksidiabetes.com
        **Hotline:** 1500-123
        
        **Jam Operasional:**
        Senin - Jumat: 08:00 - 17:00 WIB
        Sabtu: 08:00 - 12:00 WIB
        """)

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.caption("🩺 Aplikasi Prediksi Diabetes")
with footer_col2:
    st.caption("Regina Ria Aurellia-632025005")
with footer_col3:
    st.caption(f"© {datetime.now().year} - Hak Cipta Dilindungi")
