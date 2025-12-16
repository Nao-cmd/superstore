# FILE: app.py (Jalankan dengan: streamlit run app.py)

import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- SETUP (Memuat Model dan Fitur) ---

# Muat model Random Forest yang sudah disimpan
try:
    rf_model = joblib.load('rf_model.pkl')
    MODEL_FEATURES = joblib.load('model_features.pkl')
    st.success("Model Random Forest Klasifikasi berhasil dimuat.") 
except FileNotFoundError:
    st.error("File model (rf_model.pkl atau model_features.pkl) tidak ditemukan. Pastikan sudah diunduh dari Colab dan berada di folder yang sama.")
    st.stop()

# Model Regresi (Random Forest Regressor - FINAL)
try:
    # Pastikan nama file ini adalah RFR, BUKAN xgb_ atau lgbm_
    rfr_model = joblib.load('rfr_model.pkl') 
    RFR_PREPROCESSOR = joblib.load('rfr_preprocessor.pkl')
    RFR_FEATURE_NAMES = joblib.load('rfr_feature_names.pkl')
    st.sidebar.success("Model Regresi (RFR, R²=0.6038) berhasil dimuat.") 
except FileNotFoundError:
    st.sidebar.error("File model Regresi (RFR) tidak ditemukan.")
    st.stop()

# --- FUNGSI PREDIKSI ---
def preprocess_and_predict(input_data):
    """
    Memproses input pengguna dan menghasilkan prediksi
    """
    input_df = pd.DataFrame([input_data])
    
    # Kolom kategorikal yang digunakan saat training
    categorical_cols = ['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category']
    
    # 1. One-Hot Encoding pada input pengguna
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols)

    # 2. Reindex kolom agar urutan dan jumlahnya sama persis dengan MODEL_FEATURES
    final_input = pd.DataFrame(0, index=[0], columns=MODEL_FEATURES)
    
    for col in input_encoded.columns:
        if col in final_input.columns:
            final_input[col] = input_encoded[col].iloc[0]
            
    # 3. Prediksi
    prediction = rf_model.predict(final_input)
    
    return "Profitable (Keuntungan)" if prediction[0] == 1 else "Not Profitable (Rugi/Impase)"


# --- FUNGSI EVALUASI & PLOTTING ---
@st.cache_data 
def get_evaluation_metrics(df_full):
    """Menghitung ulang metrik pada data uji."""
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

    y_pred = rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0) 
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, conf_matrix, X_encoded.columns


def plot_confusion_matrix(cm):
    """Membuat dan menampilkan diagram Confusion Matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar=False,
        xticklabels=['Rugi/Impase (0)', 'Profitable (1)'],
        yticklabels=['Rugi/Impase (0)', 'Profitable (1)']
    )
    ax.set_title('Confusion Matrix Random Forest')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    st.pyplot(fig) # Menampilkan plot di Streamlit
    


def plot_feature_importance(model, features):
    """Membuat dan menampilkan diagram Feature Importance."""
    importances = model.feature_importances_
    feature_series = pd.Series(importances, index=features)
    importance_df = feature_series.sort_values(ascending=False).head(10)
    
    fig, ax = plt.subplots()
    importance_df.plot(kind='barh', ax=ax, color='skyblue')
    ax.set_title("10 Fitur Paling Berpengaruh (Feature Importance)")
    ax.set_xlabel("Tingkat Kepentingan")
    ax.invert_yaxis() # Agar fitur terpenting ada di atas
    st.pyplot(fig) # Menampilkan plot di Streamlit
    


# --- STREAMLIT UI START ---

st.set_page_config(
    page_title="Tugas Data Mining: Klasifikasi Ensemble (Random Forest)",
    layout="wide"
)

st.title("Tugas Data Mining: Klasifikasi Profitabilitas Transaksi (Superstore)")
st.header("Metode Ensemble: Random Forest")

# --- 1. UPLOAD DATA ---
st.subheader("📂 Upload Dataset Superstore")
uploaded_file = st.file_uploader("Upload 'Sample - Superstore.csv' atau dataset Anda", type=["csv"])

if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file, encoding='latin1')
    except Exception as e:
        st.error(f"Error saat memuat file: {e}")
        st.stop()
    
    # --- 2. PRATINJAU DATASET ---
    st.subheader("📌 Pratinjau Dataset")
    st.dataframe(df_uploaded.head())
    
    # --- 3. INFORMASI DATA ---
    st.markdown(f"**Jumlah baris:** {df_uploaded.shape[0]}")
    st.markdown(f"**Jumlah kolom:** {df_uploaded.shape[1]}")
    st.markdown("**Nama kolom:**")
    st.write(df_uploaded.columns.tolist())
    
    # --- 4. DATA CLEANING SUMMARY ---
    st.subheader("🧹 Data Cleaning & Preprocessing")
    
    # Solusi untuk tabel kosong: Tunjukkan bahwa data bersih dan proses preprocessing
    missing_values = df_uploaded.isnull().sum()
    st.write("Hasil Cek Missing Value:")
    st.dataframe(missing_values[missing_values > 0].to_frame('Jumlah Missing Value'))
    
    if missing_values.sum() == 0:
        st.success("Dataset ini ditemukan sangat bersih (0 Missing Value).")
    
    st.info("Langkah Preprocessing: Encoding fitur kategorikal (One-Hot Encoding) dan pembuatan variabel target biner 'Is_Profitable'.")

    # --- BAGIAN KLASIFIKASI ---
    st.markdown("---")
    st.header("🅰 Klasifikasi Profitabilitas (Random Forest)")
    
    accuracy, report, conf_matrix, encoded_features = get_evaluation_metrics(df_uploaded)
    
    # --- 4a. EVALUASI RANDOM FOREST ---
    st.subheader("📊 Evaluasi Model")
    
    col_acc, col_rep = st.columns(2)
    
    with col_acc:
        st.metric(label="Akurasi Model", value=f"{accuracy:.4f}")

    with col_rep:
        st.write("Classification Report (Presisi, Recall, F1-Score):")
        report_df = pd.DataFrame(report).transpose().round(2)
        st.dataframe(report_df)
    
    st.markdown("---")
    
    # Visualisasi Confusion Matrix dan Feature Importance
    col_cm, col_fi = st.columns(2)
    
    with col_cm:
        st.subheader("Matriks Kebingungan")
        plot_confusion_matrix(conf_matrix)
    
    with col_fi:
        st.subheader("Feature Importance")
        plot_feature_importance(rf_model, encoded_features)


    # --- 4b. PREDIKSI INTERAKTIF ---
    st.markdown("---")
    st.subheader("🔍 Prediksi Profitabilitas Transaksi Baru")
    
    # Mengumpulkan input pengguna untuk prediksi
    ship_modes = df_uploaded['Ship Mode'].unique()
    segments = df_uploaded['Segment'].unique()
    regions = df_uploaded['Region'].unique()
    categories = df_uploaded['Category'].unique()
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        # Kolom 1: Numerik
        with col1:
            sales = st.number_input("Sales ($)", min_value=0.01, max_value=25000.0, value=250.0, step=10.0)
            quantity = st.slider("Quantity (Jumlah Barang)", min_value=1, max_value=14, value=3)
            discount = st.slider("Discount (%)", min_value=0.0, max_value=0.8, value=0.0, step=0.05)

        # Kolom 2 & 3: Kategorikal
        with col2:
            ship_mode = st.selectbox("Ship Mode", ship_modes)
            segment = st.selectbox("Segment", segments)
            region = st.selectbox("Region", regions)
        
        with col3:
            category = st.selectbox("Category", categories)
            # Filter Sub-Category berdasarkan Category yang dipilih
            sub_categories = df_uploaded[df_uploaded['Category'] == category]['Sub-Category'].unique()
            sub_category = st.selectbox("Sub-Category", sub_categories)

        submitted = st.form_submit_button("Prediksi Hasil Klasifikasi")

    if submitted:
        input_data = {
            'Sales': sales, 'Quantity': quantity, 'Discount': discount,
            'Ship Mode': ship_mode, 'Segment': segment, 'Region': region,
            'Category': category, 'Sub-Category': sub_category
        }
        
        # Panggil fungsi prediksi
        result = preprocess_and_predict(input_data)
        
        st.balloons()
        st.success(f"Hasil Klasifikasi (Random Forest): Transaksi ini diprediksi **{result}**")
        
else:
    st.info("Silakan unggah dataset Anda ('Sample - Superstore.csv') untuk memulai analisis dan demo model.")
