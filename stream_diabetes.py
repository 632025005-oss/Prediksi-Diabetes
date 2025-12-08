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
    # Tombol prediksi
    if st.button("🚀 LAKUKAN PREDIKSI", type="primary", use_container_width=True):
        if model_ok and model_diabetes is not None:
            # Format input
            input_data = np.array([[kehamilan, glukosa, tekanan_darah, ketebalan_kulit,
                                  insulin, bmi, riwayat_diabetes, usia]])
            
            # Cek tipe model
            model_type = type(model_diabetes).__name__
            st.info(f"Model: {model_type}")
            
            # Prediksi berdasarkan tipe model
            if model_type == 'SVC':
                # Untuk SVC, kita buat probabilitas manual
                hasil = model_diabetes.predict(input_data)[0]
                
                # Hitung skor risiko manual
                skor_risiko = 0
                
                # Aturan sederhana berdasarkan parameter medis
                if glukosa >= 126:  # Diabetes threshold
                    skor_risiko += 30
                elif glukosa >= 100:  # Pre-diabetes
                    skor_risiko += 15
                    
                if bmi >= 30:  # Obesitas
                    skor_risiko += 20
                elif bmi >= 25:  # Overweight
                    skor_risiko += 10
                    
                if usia >= 45:
                    skor_risiko += 15
                elif usia >= 35:
                    skor_risiko += 5
                    
                if riwayat_diabetes >= 0.5:
                    skor_risiko += 20
                    
                # Faktor lainnya
                if insulin == 0:
                    skor_risiko += 10
                    
                # Normalisasi skor 0-100
                skor_risiko = min(100, skor_risiko)
                
                # Tampilkan hasil berdasarkan skor
                st.balloons()
                
                if skor_risiko >= 50:
                    st.markdown('<div class="positive">', unsafe_allow_html=True)
                    st.error(f"## ⚠️ RISIKO DIABETES TINGGI")
                    st.write(f"**Skor Risiko:** {skor_risiko}/100")
                    st.write("""
                    **Rekomendasi:**
                    1. 🏥 Konsultasi dokter segera
                    2. 💉 Cek HbA1c untuk konfirmasi
                    3. 🍎 Diet rendah gula & karbo
                    4. 🏃 Olahraga 30 menit/hari
                    5. 📊 Pantau gula darah rutin
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                elif skor_risiko >= 30:
                    st.warning(f"## ⚠️ RISIKO DIABETES SEDANG")
                    st.write(f"**Skor Risiko:** {skor_risiko}/100")
                    st.write("""
                    **Saran:**
                    1. 🩺 Cek kesehatan rutin
                    2. ⚖️ Jaga berat badan ideal
                    3. 🥗 Perbaiki pola makan
                    4. 🚶 Aktif bergerak setiap hari
                    """)
                    
                else:
                    st.markdown('<div class="negative">', unsafe_allow_html=True)
                    st.success(f"## ✅ RISIKO DIABETES RENDAH")
                    st.write(f"**Skor Risiko:** {skor_risiko}/100")
                    st.write("""
                    **Pertahankan:**
                    1. ✅ Pola makan sehat
                    2. ✅ Aktivitas fisik teratur
                    3. ✅ Cek kesehatan 6 bulan sekali
                    4. ✅ Kelola stres dengan baik
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Tampilkan analisis parameter
                st.subheader("📊 Analisis Parameter:")
                
                analisis = []
                if glukosa >= 126:
                    analisis.append(f"❌ **Glukosa tinggi** ({glukosa} ≥ 126 mg/dL)")
                elif glukosa >= 100:
                    analisis.append(f"⚠️ **Glukosa perbatasan** ({glukosa} mg/dL)")
                else:
                    analisis.append(f"✅ **Glukosa normal** ({glukosa} mg/dL)")
                    
                if bmi >= 30:
                    analisis.append(f"❌ **BMI obesitas** ({bmi} ≥ 30)")
                elif bmi >= 25:
                    analisis.append(f"⚠️ **BMI overweight** ({bmi})")
                else:
                    analisis.append(f"✅ **BMI normal** ({bmi})")
                    
                if usia >= 45:
                    analisis.append(f"⚠️ **Usia risiko** ({usia} ≥ 45 tahun)")
                else:
                    analisis.append(f"✅ **Usia aman** ({usia} tahun)")
                    
                if riwayat_diabetes >= 0.8:
                    analisis.append(f"❌ **Riwayat keluarga kuat** ({riwayat_diabetes})")
                elif riwayat_diabetes >= 0.5:
                    analisis.append(f"⚠️ **Riwayat keluarga sedang** ({riwayat_diabetes})")
                else:
                    analisis.append(f"✅ **Riwayat keluarga rendah** ({riwayat_diabetes})")
                
                for a in analisis:
                    st.write(f"- {a}")
            
            else:
                # Untuk model selain SVC (RandomForest dll)
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
            
            # Tampilkan data input (untuk semua model)
            st.subheader("📋 Data yang Dimasukkan")
            data_dict = {
                'Parameter': ['Kehamilan', 'Glukosa', 'Tekanan Darah', 'Ketebalan Kulit',
                             'Insulin', 'BMI', 'Riwayat Diabetes', 'Usia'],
                'Nilai': [kehamilan, glukosa, tekanan_darah, ketebalan_kulit,
                         insulin, bmi, riwayat_diabetes, usia],
                'Status': [
                    'Normal' if kehamilan <= 5 else 'Tinggi',
                    'Normal' if glukosa < 100 else 'Perbatasan' if glukosa < 126 else 'Tinggi',
                    'Normal' if tekanan_darah < 80 else 'Perhatian',
                    'Normal' if ketebalan_kulit < 30 else 'Tebal',
                    'Normal' if insulin > 0 else 'Tidak Ada',
                    'Normal' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obesitas',
                    'Rendah' if riwayat_diabetes < 0.5 else 'Tinggi',
                    'Muda' if usia < 40 else 'Dewasa' if usia < 60 else 'Lansia'
                ]
            }
            st.dataframe(pd.DataFrame(data_dict), use_container_width=True)
            
        else:
            st.error("❌ Model tidak tersedia. Pastikan file diabetes_model.sav ada.")

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
    st.caption("Untuk tujuan edukasi dan penelitian")
with footer_col3:
    st.caption(f"© {datetime.now().year} - Hak Cipta Dilindungi")
