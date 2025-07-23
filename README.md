# 🌾 Smart Crop Recommendation System

A machine learning-powered web application that helps farmers make informed decisions about crop selection based on soil and environmental conditions.

## 🎯 Features

- **Crop Prediction**: Get personalized crop recommendations based on soil nutrients and environmental factors
- **Interactive Visualizations**: Explore data patterns and feature relationships
- **Model Insights**: Understand feature importance and model performance
- **User-Friendly Interface**: Clean and intuitive Streamlit interface

## 📊 Dataset

The system uses a comprehensive dataset with:
- **2,200 samples** across **22 different crops**
- **7 key features**: N, P, K, Temperature, Humidity, pH, Rainfall
- **Balanced dataset**: 100 samples per crop type

### Supported Crops
Rice, Maize, Chickpea, Kidneybeans, Pigeonpeas, Mothbeans, Mungbean, Blackgram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📁 Project Structure

```
crop-recommendation-system/
│
├── app.py                          # Main Streamlit application
├── Crop_recommendation.csv         # Dataset file
├── crop_prediction_model.pkl       # Trained ML model
├── scaler.pkl                      # Feature scaler
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── models/                         # Additional model files
    ├── model_training.py           # Model training script
    └── data_analysis.py            # Data analysis utilities
```

## 🤖 Model Information

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 99.55%
- **Features**: 7 input features (N, P, K, Temperature, Humidity, pH, Rainfall)
- **Output**: Crop recommendation with confidence score

### Feature Importance
1. **Rainfall** (23.02%)
2. **Humidity** (22.42%)
3. **Potassium (K)** (17.54%)
4. **Phosphorus (P)** (15.08%)
5. **Nitrogen (N)** (9.64%)
6. **Temperature** (7.24%)
7. **pH** (5.06%)

## 💻 Usage

### Crop Prediction
1. Navigate to the "Crop Prediction" page
2. Adjust the sliders for soil nutrients (N, P, K, pH)
3. Set environmental conditions (Temperature, Humidity, Rainfall)
4. Click "Predict Best Crop" to get recommendations
5. View top 3 crop suggestions with confidence scores

### Data Visualization
- Explore crop distribution in the dataset
- Analyze feature distributions across different crops
- View correlation matrices and feature relationships
- Compare specific crops using box plots

### Model Information
- View detailed model performance metrics
- Understand feature importance
- Explore dataset statistics

## 🔧 Customization

### Adding New Features
1. Update the dataset with new features
2. Retrain the model using `models/model_training.py`
3. Update the Streamlit interface in `app.py`

### Model Improvements
- Experiment with different algorithms
- Tune hyperparameters
- Add feature engineering
- Implement cross-validation

## 📈 Technical Details

### Data Preprocessing
- Feature scaling using StandardScaler
- No missing values in the dataset
- Balanced classes (100 samples per crop)

### Model Training
- Train-test split: 80-20
- Stratified sampling to maintain class balance
- Random Forest with 100 estimators
- Model persistence using joblib

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For questions or support, please:
- Check the documentation
- Review existing issues
- Create a new issue with detailed description

## 🙏 Acknowledgments

- Dataset source: Kaggle Crop Recommendation Dataset
- Built with Streamlit, scikit-learn, and Plotly
- Inspired by precision agriculture initiatives

---

**Happy Farming! 🌱**
