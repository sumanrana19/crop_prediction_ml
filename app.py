
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="🌾 Smart Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f8f0;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 20px 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('crop_prediction_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load dataset for visualization
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Crop_recommendation.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    # Main header
    st.markdown('<h1 class="main-header">🌾 Smart Crop Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">Precision Agriculture for Better Farming Decisions</p>', unsafe_allow_html=True)

    # Load model and data
    model, scaler = load_model()
    df = load_data()

    if model is None or scaler is None or df is None:
        st.error("Failed to load required files. Please ensure all files are present.")
        return

    # Sidebar for navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["Crop Prediction", "Data Visualization", "Model Info"])

    if page == "Crop Prediction":
        crop_prediction_page(model, scaler, df)
    elif page == "Data Visualization":
        data_visualization_page(df)
    else:
        model_info_page(df)

def crop_prediction_page(model, scaler, df):
    st.markdown('<h2 class="sub-header">🔮 Crop Prediction</h2>', unsafe_allow_html=True)

    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧪 Soil Nutrients")
        nitrogen = st.slider("Nitrogen (N)", 0, 140, 50, help="Nitrogen content in soil")
        phosphorus = st.slider("Phosphorus (P)", 5, 145, 53, help="Phosphorus content in soil")
        potassium = st.slider("Potassium (K)", 5, 205, 48, help="Potassium content in soil")
        ph = st.slider("pH Level", 3.5, 10.0, 6.5, 0.1, help="pH value of the soil")

    with col2:
        st.subheader("🌡️ Environmental Conditions")
        temperature = st.slider("Temperature (°C)", 8.0, 44.0, 25.0, 0.1, help="Temperature in Celsius")
        humidity = st.slider("Humidity (%)", 14.0, 100.0, 71.0, 0.1, help="Relative humidity percentage")
        rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 103.0, 0.1, help="Rainfall in millimeters")

    # Prediction button
    if st.button("🎯 Predict Best Crop", type="primary"):
        # Prepare input data
        input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        # Get top 3 predictions
        top_3_indices = prediction_proba.argsort()[-3:][::-1]
        top_3_crops = model.classes_[top_3_indices]
        top_3_probabilities = prediction_proba[top_3_indices]

        # Display prediction
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.success(f"🎉 **Recommended Crop: {prediction.upper()}**")
        st.write(f"**Confidence: {max(prediction_proba)*100:.2f}%**")
        st.markdown('</div>', unsafe_allow_html=True)

        # Display top 3 recommendations
        st.subheader("📊 Top 3 Recommendations")
        for i, (crop, prob) in enumerate(zip(top_3_crops, top_3_probabilities)):
            st.write(f"{i+1}. **{crop.title()}** - {prob*100:.2f}%")

        # Display input summary
        st.subheader("📋 Input Summary")
        input_df = pd.DataFrame({
            'Parameter': ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 'Temperature (°C)', 
                         'Humidity (%)', 'pH Level', 'Rainfall (mm)'],
            'Value': [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
        })
        st.dataframe(input_df, use_container_width=True)

        # Compare with optimal ranges
        st.subheader("🎯 Comparison with Optimal Ranges")
        crop_data = df[df['label'] == prediction]
        if not crop_data.empty:
            comparison_data = {
                'Parameter': ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall'],
                'Your Input': [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall],
                'Optimal Min': [crop_data['N'].min(), crop_data['P'].min(), crop_data['K'].min(),
                               crop_data['temperature'].min(), crop_data['humidity'].min(),
                               crop_data['ph'].min(), crop_data['rainfall'].min()],
                'Optimal Max': [crop_data['N'].max(), crop_data['P'].max(), crop_data['K'].max(),
                               crop_data['temperature'].max(), crop_data['humidity'].max(),
                               crop_data['ph'].max(), crop_data['rainfall'].max()]
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)

def data_visualization_page(df):
    st.markdown('<h2 class="sub-header">📊 Data Visualization</h2>', unsafe_allow_html=True)

    # Dataset overview
    st.subheader("📈 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Number of Crops", df['label'].nunique())
    with col3:
        st.metric("Features", len(df.columns) - 1)
    with col4:
        st.metric("Samples per Crop", len(df) // df['label'].nunique())

    # Crop distribution
    st.subheader("🌾 Crop Distribution")
    crop_counts = df['label'].value_counts()
    fig_pie = px.pie(values=crop_counts.values, names=crop_counts.index, 
                     title="Distribution of Crops in Dataset")
    st.plotly_chart(fig_pie, use_container_width=True)

    # Feature distributions
    st.subheader("📊 Feature Distributions")
    feature_option = st.selectbox("Select feature to visualize:", 
                                  ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

    fig_dist = px.histogram(df, x=feature_option, color='label', 
                           title=f'Distribution of {feature_option} across different crops',
                           marginal="box")
    st.plotly_chart(fig_dist, use_container_width=True)

    # Correlation matrix
    st.subheader("🔗 Feature Correlation Matrix")
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        title="Correlation Matrix of Features")
    st.plotly_chart(fig_corr, use_container_width=True)

    # Box plots for selected crops
    st.subheader("📦 Feature Comparison by Crop")
    selected_crops = st.multiselect("Select crops to compare:", 
                                   df['label'].unique(), 
                                   default=df['label'].unique()[:5])

    if selected_crops:
        filtered_df = df[df['label'].isin(selected_crops)]
        feature_to_plot = st.selectbox("Select feature for comparison:", 
                                      ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'],
                                      key="box_feature")

        fig_box = px.box(filtered_df, x='label', y=feature_to_plot,
                        title=f'{feature_to_plot} distribution across selected crops')
        fig_box.update_xaxes(tickangle=45)
        st.plotly_chart(fig_box, use_container_width=True)

def model_info_page(df):
    st.markdown('<h2 class="sub-header">🤖 Model Information</h2>', unsafe_allow_html=True)

    # Model details
    st.subheader("📋 Model Details")
    st.write("""
    - **Algorithm**: Random Forest Classifier
    - **Number of Trees**: 100
    - **Features Used**: N, P, K, Temperature, Humidity, pH, Rainfall
    - **Target**: Crop Type (22 different crops)
    - **Accuracy**: 99.55%
    """)

    # Feature importance
    st.subheader("⭐ Feature Importance")
    feature_importance = {
        'Feature': ['Rainfall', 'Humidity', 'K (Potassium)', 'P (Phosphorus)', 'N (Nitrogen)', 'Temperature', 'pH'],
        'Importance': [0.2302, 0.2242, 0.1754, 0.1508, 0.0964, 0.0724, 0.0506]
    }

    fig_importance = px.bar(x=feature_importance['Importance'], 
                           y=feature_importance['Feature'],
                           orientation='h',
                           title="Feature Importance in Crop Prediction",
                           labels={'x': 'Importance Score', 'y': 'Features'})
    st.plotly_chart(fig_importance, use_container_width=True)

    # Dataset statistics
    st.subheader("📊 Dataset Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    # About the project
    st.subheader("ℹ️ About This Project")
    st.write("""
    This Smart Crop Recommendation System uses machine learning to help farmers make informed decisions 
    about crop selection based on soil and environmental conditions. The system analyzes:

    1. **Soil Nutrients**: Nitrogen (N), Phosphorus (P), Potassium (K)
    2. **Environmental Factors**: Temperature, Humidity, Rainfall
    3. **Soil Properties**: pH level

    The model provides recommendations for 22 different crops including cereals, pulses, fruits, 
    and cash crops, helping optimize agricultural productivity and sustainability.
    """)

if __name__ == "__main__":
    main()
