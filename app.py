import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="PureWater-Predict | Water Quality Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e293b;
        border-radius: 4px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 16px;
        padding: 0px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 40px;
        font-weight: 700;
        color: #3b82f6;
    }
    .prediction-card-potable {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        padding: 30px;
        border-radius: 12px;
        border: 2px solid #059669;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    .prediction-card-non-potable {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        padding: 30px;
        border-radius: 12px;
        border: 2px solid #dc2626;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

st.title("💧 PureWater-Predict Dashboard")
st.write("An Interactive Machine Learning Web Application for Water Quality analysis & Potability prediction.")

# Load resources
@st.cache_resource
def load_model_scaler():
    try:
        with open("potability_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_model_scaler()

# Load dataset for statistics/EDA
@st.cache_data
def load_dataset():
    if os.path.exists("water_potability.csv"):
        return pd.read_csv("water_potability.csv")
    return None

df = load_dataset()

# Check if model trained
if model is None or scaler is None:
    st.warning("⚠️ Machine learning models are not found or training is in progress. Please wait for model files or run training.")
    if st.button("Train Model Now", type="primary"):
        with st.spinner("Training model on dataset..."):
            os.system("python train_model.py")
            st.rerun()
else:
    tab1, tab2, tab3 = st.tabs(["🎯 Potability Predictor", "📊 Exploratory Data Analysis", "📈 Model Performance"])
    
    with tab1:
        st.subheader("Predict Water Potability in Real-Time")
        st.write("Adjust the physicochemical parameters below to evaluate if the water is potable or non-potable.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Physicochemical Parameters")
            
            # Input controls with default means from dataset description
            ph = st.slider("pH Level", 0.0, 14.0, 7.08, 0.01, help="Acidity or alkalinity level. Standard range is 6.5 - 8.5.")
            hardness = st.slider("Hardness (mg/L)", 40.0, 325.0, 196.37, 0.1, help="Concentration of calcium and magnesium.")
            solids = st.slider("Solids (TDS in ppm)", 320.0, 61227.0, 22014.06, 1.0, help="Total dissolved solids.")
            chloramines = st.slider("Chloramines (ppm)", 0.35, 13.13, 7.12, 0.01, help="Chloramine concentration.")
            sulfate = st.slider("Sulfate (mg/L)", 129.0, 481.0, 333.78, 0.1, help="Sulfate level.")
            conductivity = st.slider("Conductivity (μS/cm)", 181.0, 753.34, 426.21, 0.1, help="Electrical conductivity.")
            organic_carbon = st.slider("Organic Carbon (ppm)", 2.20, 28.30, 14.28, 0.01, help="Total Organic Carbon level.")
            trihalomethanes = st.slider("Trihalomethanes (μg/L)", 0.73, 124.0, 66.40, 0.01, help="Byproducts of chlorine disinfection.")
            turbidity = st.slider("Turbidity (NTU)", 1.45, 6.74, 3.97, 0.01, help="Clarity measurement based on suspended particles.")

        with col2:
            st.markdown("### Prediction Result")
            st.write("Click below to run the XGBoost prediction pipeline.")
            
            if st.button("Analyze Water Sample", type="primary", use_container_width=True):
                # 1. Transform skewed features (using square root)
                solids_trans = np.sqrt(solids)
                cond_trans = np.sqrt(conductivity)
                
                # 2. Arrange features into array
                features = np.array([[ph, hardness, solids_trans, chloramines, sulfate, cond_trans, organic_carbon, trihalomethanes, turbidity]])
                
                # 3. Scale features
                features_scaled = scaler.transform(features)
                
                # 4. Predict
                prediction = model.predict(features_scaled)[0]
                probability = model.predict_proba(features_scaled)[0]
                
                # 5. Render gorgeous prediction cards
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-card-potable">
                        <h2 style="color: white; margin: 0;">✅ POTABLE</h2>
                        <p style="color: #a7f3d0; font-size: 18px; margin-top: 10px; margin-bottom: 0;">
                            This water sample is <b>SAFE</b> for human consumption.
                        </p>
                        <h3 style="color: white; margin-top: 15px; margin-bottom: 0;">
                            Confidence: {probability[1]*100:.1f}%
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-card-non-potable">
                        <h2 style="color: white; margin: 0;">❌ NON-POTABLE</h2>
                        <p style="color: #fca5a5; font-size: 18px; margin-top: 10px; margin-bottom: 0;">
                            This water sample is <b>UNSAFE</b> for human consumption.
                        </p>
                        <h3 style="color: white; margin-top: 15px; margin-bottom: 0;">
                            Confidence: {probability[0]*100:.1f}%
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)

    with tab2:
        st.subheader("Interactive Exploratory Data Analysis")
        if df is not None:
            eda_option = st.selectbox("Select Analysis Feature", ["Dataset Summary", "Feature Distribution", "Feature vs Feature Scatter Plot", "Correlation Matrix"])
            
            if eda_option == "Dataset Summary":
                st.write("Preview of the raw dataset:")
                st.dataframe(df.head(10), use_container_width=True)
                st.write("Statistical Description:")
                st.dataframe(df.describe(), use_container_width=True)
                
            elif eda_option == "Feature Distribution":
                col_dist = st.selectbox("Select column to inspect", df.drop(columns=['Potability']).columns)
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.histplot(data=df, x=col_dist, hue='Potability', kde=True, bins=30, palette={0: 'red', 1: 'green'}, alpha=0.5, ax=ax)
                ax.set_title(f"Distribution of {col_dist} by Potability")
                st.pyplot(fig)
                
            elif eda_option == "Feature vs Feature Scatter Plot":
                col_x = st.selectbox("Select X axis", df.drop(columns=['Potability']).columns, index=0)
                col_y = st.selectbox("Select Y axis", df.drop(columns=['Potability']).columns, index=1)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df, x=col_x, y=col_y, hue='Potability', palette={0: 'red', 1: 'green'}, alpha=0.6, ax=ax)
                ax.set_title(f"{col_x} vs {col_y}")
                st.pyplot(fig)
                
            elif eda_option == "Correlation Matrix":
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="Blues", vmin=-1, vmax=1, ax=ax)
                ax.set_title("Pearson Correlation Heatmap")
                st.pyplot(fig)
        else:
            st.error("Dataset not found. Please place `water_potability.csv` in the root directory.")

    with tab3:
        st.subheader("Model Metrics & Performance")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="Model Architecture", value="XGBoost Classifier")
            st.metric(label="Mean CV Accuracy", value="81.0%")
            st.metric(label="Weighted F1-Score", value="80.0%")
        with col_m2:
            st.markdown("""
            ### XGBoost Hyperparameters
            - **random_state**: 42
            - **reg_lambda (L2 Regularization)**: 1.3
            - **Data Preprocessing Pipeline**:
              1. **Imputation**: Group-mean values for pH, Sulfate, THMs.
              2. **Outlier Filtering**: Z-score thresholding (>3 standard deviations).
              3. **Transformations**: Square root transforms on `Solids` and `Conductivity` to reduce skew.
              4. **Resampling**: SMOTE applied to minority Potable class.
            """)
        
        st.markdown("### Feature Importance")
        if df is not None:
            # Recreate feature importance
            importances = model.feature_importances_
            feature_names = df.drop(columns=['Potability']).columns
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=importance_df, x='Importance', y='Feature', hue='Feature', palette='Blues_r', legend=False, ax=ax)
            ax.set_title("XGBoost Feature Importance Plot")
            st.pyplot(fig)
