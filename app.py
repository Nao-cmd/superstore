# FILE 2: app.py (Jalankan dengan: streamlit run app.py)

import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import io

# --- SETUP (Memuat Model dan Fitur) ---

# Muat model Random Forest yang sudah disimpan
try:
    rf_model = joblib.load('rf_model.pkl')
    MODEL_FEATURES = joblib.load('model_features.pkl')
    st.success("Model Random Forest Klasifikasi berhasil dimuat.") # <--- PERBAIKAN!
except FileNotFoundError:

# --- FUNGSI PREDIKSI ---
def preprocess_and_predict(input_data):
    """
    Memproses input pengguna dan menghasilkan prediksi
    """
    # Mengubah input menjadi DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Identifikasi kolom kategorikal (harus sama dengan saat training)
    categorical_cols = ['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category']
    
    # 1. One-Hot Encoding pada input pengguna
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols)

    # 2. Reindex kolom agar urutan dan jumlahnya sama persis dengan MODEL_FEATURES
    # Membuat DataFrame dummy dengan semua kolom yang dibutuhkan (diisi 0)
    final_input = pd.DataFrame(0, index=[0], columns=MODEL_FEATURES)
    
    # Mengisi nilai dari input pengguna yang sudah di-encoded
    for col in input_encoded.columns:
        if col in final_input.columns:
            final_input[col] = input_encoded[col].iloc[0]
            
    # 3. Prediksi
    prediction = rf_model.predict(final_input)
    
    return "Profitable (Keuntungan)" if prediction[0] == 1 else "Not Profitable (Rugi/Impase)"


# --- FUNGSI EVALUASI (Menggunakan Data Uji dari Model Asli) ---
def get_evaluation_metrics(df_full):
    """
    Menghitung ulang metrik pada data uji (harus dilakukan di lingkungan yang sama)
    """
    # Ulangi langkah Feature Engineering dan Preprocessing
    df_full['Is_Profitable'] = (df_full['Profit'] > 0).astype(int)
    features = ['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category', 'Sales', 'Quantity', 'Discount']
    target = 'Is_Profitable'

    X = df_full[features]
    y = df_full[target]
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Splitting data (HARUS sama dengan saat training)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # Prediksi
    y_pred = rf_model.predict(X_test)
    
    # Hasil
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, conf_matrix, X.shape[0], X_encoded.shape[1]

# --- STREAMLIT UI START ---

st.set_page_config(
    page_title="Tugas Data Mining: Klasifikasi Ensemble (Random Forest)",
    layout="wide"
)

st.title("Tugas Data Mining: Klasifikasi Transaksi Superstore")
st.header("Metode Ensemble: Random Forest")

# --- 1. UPLOAD DATA (Sesuai Referensi Teman) ---
st.subheader("📂 Upload Dataset Superstore")
uploaded_file = st.file_uploader("Upload 'Sample - Superstore.csv' atau dataset Anda", type=["csv"])

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file, encoding='latin1')
    
    # --- 2. PRATINJAU DATASET ---
    st.subheader("📌 Pratinjau Dataset")
    st.dataframe(df_uploaded.head())
    
    # --- 3. INFORMASI DATA ---
    st.markdown(f"**Jumlah baris:** {df_uploaded.shape[0]}")
    st.markdown(f"**Jumlah kolom:** {df_uploaded.shape[1]}")
    st.markdown("**Nama kolom:**")
    st.write(df_uploaded.columns.tolist())
    
    # --- 4. DATA CLEANING SUMMARY (Sesuai Referensi) ---
    st.subheader("🧹 Data Cleaning")
    missing_values = df_uploaded.isnull().sum()
    st.write("Missing value per kolom:")
    st.dataframe(missing_values[missing_values > 0])
    
    st.markdown(f"**Ukuran data setelah cleaning (asumsi tidak ada drop):** ({df_uploaded.shape[0]}, {df_uploaded.shape[1]})")

    # --- BAGIAN A: KLASIFIKASI HARGA (RANDOM FOREST) ---
    st.markdown("---")
    st.header("🅰 Bagian A – Klasifikasi Profit (Random Forest)")
    
    accuracy, report, conf_matrix, total_rows, total_features = get_evaluation_metrics(df_uploaded)
    
    # --- 4a. EVALUASI RANDOM FOREST ---
    st.subheader("📊 Evaluasi Random Forest")
    st.markdown(f"**Accuracy:** `{accuracy:.4f}`")

    # Menampilkan Classification Report (dalam format tabel)
    st.write("Classification Report:")
    report_df = pd.DataFrame(report).transpose().round(2)
    st.dataframe(report_df)
    
    st.write("Confusion Matrix:")
    st.code(conf_matrix)

    # --- 4b. PREDIKSI INTERAKTIF ---
    st.subheader("🔍 Prediksi Profitabilitas Transaksi")
    
    # Mengumpulkan input pengguna untuk prediksi
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        # Kolom 1
        with col1:
            sales = st.number_input("Sales ($)", min_value=1.0, max_value=20000.0, value=250.0, step=10.0)
            quantity = st.slider("Quantity (Jumlah Barang)", min_value=1, max_value=14, value=3)
            discount = st.slider("Discount (%)", min_value=0.0, max_value=0.8, value=0.0, step=0.05)

        # Kolom 2
        with col2:
            ship_mode = st.selectbox("Ship Mode", df_uploaded['Ship Mode'].unique())
            segment = st.selectbox("Segment", df_uploaded['Segment'].unique())
            region = st.selectbox("Region", df_uploaded['Region'].unique())
        
        # Kolom 3
        with col3:
            category = st.selectbox("Category", df_uploaded['Category'].unique())
            sub_category = st.selectbox("Sub-Category", df_uploaded[df_uploaded['Category'] == category]['Sub-Category'].unique())

        submitted = st.form_submit_button("Prediksi Hasil Klasifikasi")

    if submitted:
        input_data = {
            'Sales': sales,
            'Quantity': quantity,
            'Discount': discount,
            'Ship Mode': ship_mode,
            'Segment': segment,
            'Region': region,
            'Category': category,
            'Sub-Category': sub_category
        }
        
        # Panggil fungsi prediksi
        result = preprocess_and_predict(input_data)
        
        st.success(f"Hasil Klasifikasi (Random Forest): Transaksi ini diprediksi **{result}**")
        
    st.markdown("---")
    
else:
    st.info("Silakan unggah dataset Anda untuk memulai analisis dan demo model.")

# --- BAGIAN B & SEGMENTASI (Placeholder) ---
# Anda dapat menambahkan bagian B (Regresi) dan Segmentasi di sini
st.header("🅱 Bagian B – Prediksi Konsumsi BBM Motor (Regresi) - [Placeholder]")
st.header("📊 Segmentasi Motor Berdasarkan Konsumsi BBM - [Placeholder]")
