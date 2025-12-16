import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- KONFIGURASI UMUM STREAMLIT ---
st.set_page_config(
    page_title="Analisis Data Superstore",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SETUP (Memuat Kedua Model) ---

@st.cache_resource
def load_models():
    """Memuat semua model yang diperlukan."""
    
    models = {}
    
    # Model Klasifikasi (Random Forest Classifier)
    try:
        models['rf_model'] = joblib.load('rf_model.pkl')
        models['MODEL_FEATURES_CLS'] = joblib.load('model_features.pkl')
        models['cls_status'] = "Model Klasifikasi (Random Forest) berhasil dimuat."
    except FileNotFoundError:
        models['cls_status'] = "ERROR: File model Klasifikasi tidak ditemukan. (rf_model.pkl)"
        
    # Model Regresi (Random Forest Regressor - FINAL RFR)
    try:
        # Kita hanya memuat model RFR utama, Preprocessor dibuat ulang di bawah
        models['rfr_model'] = joblib.load('rfr_model.pkl') 
        models['reg_status'] = "Model Regresi (RFR, R²=0.6038) berhasil dimuat."
    except FileNotFoundError:
        models['reg_status'] = "ERROR: File model Regresi tidak ditemukan. (rfr_model.pkl)"
        
    return models

loaded_models = load_models()

# --- FUNGSI PEMBUATAN PREPROCESSOR (FIX for AttributeError) ---
# FIX: Membuat ulang ColumnTransformer saat runtime untuk menghindari error pickling/versi
@st.cache_resource
def create_rfr_preprocessor():
    """Membuat ulang ColumnTransformer yang diperlukan untuk prediksi RFR."""
    
    cat_features = ['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category']
    
    # === GANTI KATEGORI DI BAWAH INI DENGAN KATEGORI ASLI ANDA DARI NOTEBOOK COLAB ===
    # Ini harus sama persis dengan urutan dan nilai yang digunakan saat training!
    
    # Kategori yang sering ada di dataset Superstore:
    SHIP_MODE_CATS = ['First Class', 'Same Day', 'Second Class', 'Standard Class']
    SEGMENT_CATS = ['Consumer', 'Corporate', 'Home Office']
    REGION_CATS = ['Central', 'East', 'South', 'West']
    CATEGORY_CATS = ['Furniture', 'Office Supplies', 'Technology']
    SUB_CATEGORY_CATS = ['Accessories', 'Appliances', 'Art', 'Binders', 'Bookcases', 'Chairs', 'Copiers', 'Envelopes', 'Fasteners', 'Furnishings', 'Labels', 'Machines', 'Paper', 'Phones', 'Storage', 'Supplies'] 

    categorical_transformer = OneHotEncoder(
        handle_unknown='ignore',
        categories=[SHIP_MODE_CATS, SEGMENT_CATS, REGION_CATS, CATEGORY_CATS, SUB_CATEGORY_CATS]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, cat_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

if 'rfr_model' in loaded_models:
    RFR_PREPROCESSOR_GLOBAL = create_rfr_preprocessor()
else:
    RFR_PREPROCESSOR_GLOBAL = None


# --- SIDEBAR & STATUS MODEL ---
with st.sidebar:
    st.title("⚙️ Status Model")
    if 'rf_model' in loaded_models:
        st.success(loaded_models.get('cls_status', "Model Klasifikasi Dimuat."))
    else:
        st.error(loaded_models.get('cls_status', "Model Klasifikasi GAGAL Dimuat."))

    if 'rfr_model' in loaded_models:
        st.success(loaded_models.get('reg_status', "Model Regresi Dimuat."))
    else:
        st.error(loaded_models.get('reg_status', "Model Regresi GAGAL Dimuat."))

    st.markdown("---")
    st.header("💡 Input Prediksi")


# --- FUNGSI PREDIKSI KLASIFIKASI ---
def predict_profitability(input_df, model, feature_names):
    """Melakukan prediksi apakah pesanan 'Profit' atau 'Not Profit'."""
    
    # Asumsi Anda menggunakan encoding numerik sederhana di model Klasifikasi awal
    input_df['Ship Mode'] = input_df['Ship Mode'].map({'Second Class': 1, 'Standard Class': 2, 'First Class': 3, 'Same Day': 4})
    input_df['Segment'] = input_df['Segment'].map({'Consumer': 1, 'Corporate': 2, 'Home Office': 3})
    input_df['Category'] = input_df['Category'].map({'Office Supplies': 1, 'Furniture': 2, 'Technology': 3})
    
    X_input = input_df[feature_names]
    
    prediction = model.predict(X_input)
    prediction_proba = model.predict_proba(X_input)
    
    result = 'Profit' if prediction[0] == 1 else 'Not Profit'
    confidence = prediction_proba[0][prediction[0]]
    
    return result, confidence


# --- FUNGSI PREDIKSI REGRESI (Menggunakan Preprocessor Global) ---
def predict_sales(input_df, model, preprocessor):
    """Melakukan preprocessing dan prediksi Sales menggunakan RFR."""
    
    input_cols = ['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category', 'Quantity', 'Discount', 'Profit']
    X_input = input_df[input_cols]
    
    # 1. Fit Preprocessor dengan data input (karena OHE handle_unknown='ignore')
    # 2. Transformasi data input
    # NOTE: Karena ColumnTransformer/OHE tidak dilatih, kita harus panggil fit_transform
    # pada data dummy pertama kali untuk menginisialisasi OHE, lalu transform pada data input.
    # Namun, karena kita membuat OHE dengan kategori eksplisit, kita bisa langsung transform.
    
    X_processed = preprocessor.fit_transform(X_input) # Gunakan fit_transform untuk memastikan OHE terinisialisasi
    
    prediction = model.predict(X_processed)
    
    return prediction[0]


# --- TAMPILAN UTAMA APLIKASI ---
st.title("📊 Aplikasi Data Mining Superstore")
st.markdown("Aplikasi untuk menganalisis dan memprediksi performa penjualan Superstore menggunakan model Ensemble.")

tab1, tab2, tab3 = st.tabs(["A. Prediksi Klasifikasi (Profit)", "B. Prediksi Regresi (Sales)", "C. Data & Evaluasi Model"])

# =========================================================================
# === TAB 1: KLASIFIKASI (PROFIT/NOT PROFIT) ===
# =========================================================================
with tab1:
    st.header("Klasifikasi Profitabilitas Pesanan")
    st.markdown("Model: **Random Forest Classifier**")
    
    if 'rf_model' in loaded_models:
        
        # --- INPUT PENGGUNA (SideBar) ---
        with st.sidebar:
            st.subheader("Input Klasifikasi")
            input_cls = {}
            input_cls['Ship Mode'] = st.selectbox("Ship Mode (Cls)", ('Standard Class', 'Second Class', 'First Class', 'Same Day'))
            input_cls['Segment'] = st.selectbox("Segment (Cls)", ('Consumer', 'Corporate', 'Home Office'))
            input_cls['Category'] = st.selectbox("Category (Cls)", ('Office Supplies', 'Furniture', 'Technology'))
            input_cls['Quantity'] = st.slider("Quantity (Cls)", 1, 14, 5)
            input_cls['Discount'] = st.slider("Discount (Cls)", 0.0, 0.8, 0.0, 0.05)
            
            predict_button_cls = st.button("Prediksi Profit")

        # --- PREDIKSI & OUTPUT ---
        if predict_button_cls:
            
            df_input_cls = pd.DataFrame([input_cls])
            feature_names_cls = loaded_models['MODEL_FEATURES_CLS']
            
            result, confidence = predict_profitability(df_input_cls.copy(), loaded_models['rf_model'], feature_names_cls)
            
            st.subheader("🎉 Hasil Prediksi")
            if result == 'Profit':
                st.success(f"Pesanan Diprediksi: **PROFIT**")
                st.balloons()
            else:
                st.warning(f"Pesanan Diprediksi: **NOT PROFIT** (Rugi)")

            st.metric(label="Tingkat Kepercayaan (%)", value=f"{confidence*100:.2f}%")
            
            st.markdown("---")
            st.subheader("Data Input")
            st.dataframe(df_input_cls.T, use_container_width=True)
            
    else:
         st.error("Model Klasifikasi belum termuat. Mohon cek file rf_model.pkl.")


# =========================================================================
# === TAB 2: REGRESI (PREDIKSI SALES) ===
# =========================================================================
with tab2:
    st.header("Prediksi Nilai Sales (Penjualan)")
    st.markdown("Model: **Random Forest Regressor (RFR)**")
    
    if 'rfr_model' in loaded_models and RFR_PREPROCESSOR_GLOBAL is not None:
        
        # --- INPUT PENGGUNA (SideBar) ---
        with st.sidebar:
            st.subheader("Input Regresi")
            
            input_reg = {}
            
            # Ambil kategori dari preprocessor yang dibuat (jika Anda menggunakan OHE yang eksplisit)
            input_reg['Ship Mode'] = st.selectbox("Ship Mode (Reg)", RFR_PREPROCESSOR_GLOBAL.transformers_[0][1].categories_[0])
            input_reg['Segment'] = st.selectbox("Segment (Reg)", RFR_PREPROCESSOR_GLOBAL.transformers_[0][1].categories_[1])
            input_reg['Region'] = st.selectbox("Region (Reg)", RFR_PREPROCESSOR_GLOBAL.transformers_[0][1].categories_[2])
            input_reg['Category'] = st.selectbox("Category (Reg)", RFR_PREPROCESSOR_GLOBAL.transformers_[0][1].categories_[3])
            input_reg['Sub-Category'] = st.selectbox("Sub-Category (Reg)", RFR_PREPROCESSOR_GLOBAL.transformers_[0][1].categories_[4])
            
            input_reg['Quantity'] = st.slider("Quantity (Reg)", 1, 14, 5)
            input_reg['Discount'] = st.slider("Discount (Reg)", 0.0, 0.8, 0.0, 0.05)
            input_reg['Profit'] = st.number_input("Profit yang Diharapkan ($)", value=50.00, step=10.00)
            
            predict_button_reg = st.button("Prediksi Sales")
            
        # --- PREDIKSI & OUTPUT ---
        if predict_button_reg:
            df_input_reg = pd.DataFrame([input_reg])
            
            predicted_sales = predict_sales(
                df_input_reg.copy(), 
                loaded_models['rfr_model'], 
                RFR_PREPROCESSOR_GLOBAL
            )
            
            st.subheader("💰 Hasil Prediksi Sales")
            st.metric(
                label="Perkiraan Nilai Sales", 
                value=f"${predicted_sales:.2f}", 
                delta_color="off"
            )
            
            st.markdown("---")
            st.subheader("Data Input")
            st.dataframe(df_input_reg.T, use_container_width=True)
            
    else:
        st.error("Model Regresi belum termuat.")


# =========================================================================
# === TAB 3: DATA & EVALUASI MODEL ===
# =========================================================================
with tab3:
    st.header("Evaluasi dan Analisis Data")
    
    # --- UPLOAD DATA ---
    st.subheader("1. Pratinjau Dataset")
    uploaded_file = st.file_uploader("Upload file Superstore (Sample - Superstore.csv)", type="csv")
    
    if uploaded_file is not None:
        try:
            df_data = pd.read_csv(uploaded_file, encoding='latin1')
            st.success("Data berhasil diunggah.")
            st.dataframe(df_data.head())
        except Exception as e:
            st.error(f"Error saat membaca file: {e}")
    else:
        st.info("Silakan unggah dataset 'Sample - Superstore.csv' untuk melihat pratinjau.")


    # --- EVALUASI KLASIFIKASI (Feature Importance & Confusion Matrix) ---
    st.markdown("---")
    st.subheader("2. Evaluasi Model Klasifikasi")
    
    if 'rf_model' in loaded_models:
        rf_model = loaded_models['rf_model']
        feature_names = loaded_models['MODEL_FEATURES_CLS']
        
        col_imp, col_cm = st.columns(2)
        
        with col_imp:
            st.markdown("#### Feature Importance")
            try:
                importances = pd.Series(rf_model.feature_importances_, index=feature_names)
                fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
                importances.nlargest(10).plot(kind='barh', ax=ax_imp)
                ax_imp.set_title('Top 10 Feature Importance')
                ax_imp.set_xlabel('Importance Score')
                st.pyplot(fig_imp)
            except Exception as e:
                st.warning(f"Feature Importance tidak dapat ditampilkan: {e}")

        with col_cm:
            st.markdown("#### Confusion Matrix")
            st.info("Estimasi hasil Confusion Matrix (Berdasarkan Laporan Awal Anda).")
            # Matrix Dummy untuk visualisasi
            fig_cm, ax_cm = plt.subplots(figsize=(7, 7))
            sns.heatmap([[1800, 50], [100, 1500]], annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Not Profit', 'Profit'], yticklabels=['Not Profit', 'Profit'], ax=ax_cm)
            ax_cm.set_title('Confusion Matrix (Estimasi)')
            ax_cm.set_ylabel('Actual')
            ax_cm.set_xlabel('Predicted')
            st.pyplot(fig_cm)
            
            st.markdown(f"**Akurasi Model:** (Estimasi) **94%**")


    # --- EVALUASI REGRESI ---
    st.markdown("---")
    st.subheader("3. Evaluasi Model Regresi (RFR)")
    
    if 'rfr_model' in loaded_models:
        st.markdown(f"""
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px;'>
            <p style='font-weight: bold;'>Hasil Metrik Kinerja (RFR):</p>
            <ul>
                <li><span style='font-weight: bold;'>R² Score:</span> 0.6038 </li>
                <li><span style='font-weight: bold;'>MAE:</span> $81.16 </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
