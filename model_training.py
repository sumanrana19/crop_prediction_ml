
"""
Model Training Script for Crop Recommendation System
This script handles data preprocessing, model training, and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class CropPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

    def load_data(self, filepath='Crop_recommendation.csv'):
        """Load and prepare the dataset"""
        try:
            df = pd.read_csv(filepath)
            print(f"Dataset loaded successfully: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def preprocess_data(self, df):
        """Preprocess the data for training"""
        X = df[self.feature_names]
        y = df['label']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_model(self, X_train, y_train, hyperparameter_tuning=False):
        """Train the Random Forest model"""
        if hyperparameter_tuning:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        else:
            # Use default parameters
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)

        return self.model

    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return accuracy, y_pred

    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return cv_scores

    def plot_feature_importance(self):
        """Plot feature importance"""
        if self.model is None:
            print("Model not trained yet!")
            return

        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [self.feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_model(self, model_path='crop_prediction_model.pkl', scaler_path='scaler.pkl'):
        """Save the trained model and scaler"""
        if self.model is None or self.scaler is None:
            print("Model or scaler not available!")
            return

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

    def load_model(self, model_path='crop_prediction_model.pkl', scaler_path='scaler.pkl'):
        """Load a pre-trained model and scaler"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print("Model and scaler loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")

    def predict(self, input_data):
        """Make predictions on new data"""
        if self.model is None or self.scaler is None:
            print("Model or scaler not available!")
            return None

        input_scaled = self.scaler.transform(input_data)
        prediction = self.model.predict(input_scaled)
        prediction_proba = self.model.predict_proba(input_scaled)

        return prediction, prediction_proba

def main():
    """Main training pipeline"""
    # Initialize the predictor
    predictor = CropPredictor()

    # Load data
    df = predictor.load_data()
    if df is None:
        return

    # Preprocess data
    X_train, X_test, y_train, y_test = predictor.preprocess_data(df)

    # Train model
    print("Training model...")
    model = predictor.train_model(X_train, y_train, hyperparameter_tuning=False)

    # Evaluate model
    print("\nEvaluating model...")
    accuracy, y_pred = predictor.evaluate_model(X_test, y_test)

    # Cross-validation
    print("\nPerforming cross-validation...")
    X_full = df[predictor.feature_names]
    y_full = df['label']
    X_full_scaled = predictor.scaler.transform(X_full)
    predictor.cross_validate(X_full_scaled, y_full)

    # Plot feature importance
    predictor.plot_feature_importance()

    # Save model
    predictor.save_model()

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
