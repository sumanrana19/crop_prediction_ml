
"""
Data Analysis Utilities for Crop Recommendation System
This script provides comprehensive data analysis and visualization functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class CropDataAnalyzer:
    def __init__(self, data_path='Crop_recommendation.csv'):
        """Initialize the analyzer with the dataset"""
        self.df = pd.read_csv(data_path)
        self.feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.target_column = 'label'

    def basic_info(self):
        """Display basic information about the dataset"""
        print("=== DATASET OVERVIEW ===")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"\nData types:")
        print(self.df.dtypes)
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        print(f"\nUnique crops: {self.df[self.target_column].nunique()}")
        print(f"Crops: {sorted(self.df[self.target_column].unique())}")

    def statistical_summary(self):
        """Generate statistical summary"""
        print("\n=== STATISTICAL SUMMARY ===")
        print(self.df[self.feature_columns].describe())

        # Crop distribution
        print(f"\n=== CROP DISTRIBUTION ===")
        crop_counts = self.df[self.target_column].value_counts()
        print(crop_counts)

        return crop_counts

    def plot_feature_distributions(self, save_plots=False):
        """Plot distributions of all features"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.ravel()

        for i, feature in enumerate(self.feature_columns):
            axes[i].hist(self.df[feature], bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)

        # Remove empty subplot
        axes[-2].remove()
        axes[-1].remove()

        plt.tight_layout()
        if save_plots:
            plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

    def correlation_analysis(self, save_plots=False):
        """Analyze correlations between features"""
        correlation_matrix = self.df[self.feature_columns].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        if save_plots:
            plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        return correlation_matrix

    def crop_wise_analysis(self, crops=None):
        """Analyze features for specific crops"""
        if crops is None:
            crops = self.df[self.target_column].unique()[:5]  # First 5 crops

        crop_stats = {}
        for crop in crops:
            crop_data = self.df[self.df[self.target_column] == crop]
            crop_stats[crop] = crop_data[self.feature_columns].describe()

        return crop_stats

    def plot_crop_comparison(self, feature, crops=None, save_plots=False):
        """Compare a specific feature across different crops"""
        if crops is None:
            crops = self.df[self.target_column].unique()

        crop_data = self.df[self.df[self.target_column].isin(crops)]

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=crop_data, x=self.target_column, y=feature)
        plt.xticks(rotation=45)
        plt.title(f'{feature} Distribution Across Crops')
        plt.tight_layout()
        if save_plots:
            plt.savefig(f'{feature}_crop_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def outlier_detection(self):
        """Detect outliers in the dataset"""
        outliers = {}

        for feature in self.feature_columns:
            Q1 = self.df[feature].quantile(0.25)
            Q3 = self.df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_indices = self.df[(self.df[feature] < lower_bound) | 
                                    (self.df[feature] > upper_bound)].index
            outliers[feature] = len(outlier_indices)

        print("\n=== OUTLIER ANALYSIS ===")
        for feature, count in outliers.items():
            print(f"{feature}: {count} outliers")

        return outliers

    def pca_analysis(self, n_components=2):
        """Perform PCA analysis"""
        from sklearn.preprocessing import StandardScaler

        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.df[self.feature_columns])

        # PCA
        pca = PCA(n_components=n_components)
        pca_components = pca.fit_transform(scaled_features)

        # Plot PCA
        if n_components == 2:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(pca_components[:, 0], pca_components[:, 1], 
                                c=pd.Categorical(self.df[self.target_column]).codes, 
                                cmap='tab20', alpha=0.6)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('PCA: Crop Data Visualization')
            plt.colorbar(scatter)
            plt.grid(True, alpha=0.3)
            plt.show()

        print(f"\nPCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
        print(f"Total Variance Explained: {pca.explained_variance_ratio_.sum():.2%}")

        return pca, pca_components

    def feature_importance_by_crop(self):
        """Analyze which features are most important for each crop"""
        crop_profiles = {}

        for crop in self.df[self.target_column].unique():
            crop_data = self.df[self.df[self.target_column] == crop]
            profile = {}

            for feature in self.feature_columns:
                profile[feature] = {
                    'mean': crop_data[feature].mean(),
                    'std': crop_data[feature].std(),
                    'min': crop_data[feature].min(),
                    'max': crop_data[feature].max()
                }

            crop_profiles[crop] = profile

        return crop_profiles

    def generate_insights(self):
        """Generate key insights from the data"""
        insights = []

        # Basic insights
        insights.append(f"Dataset contains {len(self.df)} samples across {self.df[self.target_column].nunique()} crops")
        insights.append(f"Each crop has exactly {len(self.df) // self.df[self.target_column].nunique()} samples (balanced dataset)")

        # Feature insights
        for feature in self.feature_columns:
            mean_val = self.df[feature].mean()
            std_val = self.df[feature].std()
            insights.append(f"{feature}: Mean = {mean_val:.2f}, Std = {std_val:.2f}")

        # Correlation insights
        corr_matrix = self.df[self.feature_columns].corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.5:  # High correlation threshold
                    high_corr_pairs.append((corr_matrix.columns[i], 
                                          corr_matrix.columns[j], corr_val))

        if high_corr_pairs:
            insights.append("High correlations found:")
            for pair in high_corr_pairs:
                insights.append(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")

        return insights

    def export_analysis_report(self, filename='crop_analysis_report.txt'):
        """Export comprehensive analysis report"""
        with open(filename, 'w') as f:
            f.write("CROP RECOMMENDATION DATASET ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Basic info
            f.write(f"Dataset Shape: {self.df.shape}\n")
            f.write(f"Features: {self.feature_columns}\n")
            f.write(f"Target: {self.target_column}\n")
            f.write(f"Number of Crops: {self.df[self.target_column].nunique()}\n\n")

            # Statistical summary
            f.write("STATISTICAL SUMMARY:\n")
            f.write(str(self.df[self.feature_columns].describe()))
            f.write("\n\n")

            # Insights
            insights = self.generate_insights()
            f.write("KEY INSIGHTS:\n")
            for insight in insights:
                f.write(f"- {insight}\n")

        print(f"Analysis report exported to {filename}")

def main():
    """Main analysis pipeline"""
    # Initialize analyzer
    analyzer = CropDataAnalyzer()

    # Basic analysis
    analyzer.basic_info()
    crop_counts = analyzer.statistical_summary()

    # Plot distributions
    analyzer.plot_feature_distributions(save_plots=True)

    # Correlation analysis
    corr_matrix = analyzer.correlation_analysis(save_plots=True)

    # Crop comparison for key features
    key_features = ['N', 'P', 'K', 'rainfall']
    for feature in key_features:
        analyzer.plot_crop_comparison(feature, save_plots=True)

    # Outlier detection
    outliers = analyzer.outlier_detection()

    # PCA analysis
    pca, pca_components = analyzer.pca_analysis()

    # Generate insights
    insights = analyzer.generate_insights()
    print("\n=== KEY INSIGHTS ===")
    for insight in insights:
        print(f"- {insight}")

    # Export report
    analyzer.export_analysis_report()

    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()
