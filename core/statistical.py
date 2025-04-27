"""
Advanced statistical analysis module for the dataset analyzer
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.helpers import get_logger

logger = get_logger()

class StatisticalAnalyzer:
    """Provides advanced statistical analysis capabilities"""
    
    def __init__(self, data=None):
        self.data = data
        
    def set_data(self, data):
        """Set the data to analyze"""
        self.data = data
        
    def get_summary_statistics(self):
        """Get comprehensive summary statistics for numeric columns"""
        if self.data is None:
            logger.warning("No data available for summary statistics")
            return None
            
        numeric_data = self.data.select_dtypes(include=np.number)
        if numeric_data.empty:
            logger.warning("No numeric columns found in dataset")
            return pd.DataFrame()
            
        # Extended statistics beyond .describe()
        stats_df = pd.DataFrame({
            'mean': numeric_data.mean(),
            'median': numeric_data.median(),
            'std': numeric_data.std(),
            'var': numeric_data.var(),
            'min': numeric_data.min(),
            'max': numeric_data.max(),
            'skew': numeric_data.skew(),
            'kurtosis': numeric_data.kurtosis(),
            'iqr': numeric_data.quantile(0.75) - numeric_data.quantile(0.25),
        })
        
        return stats_df
        
    def test_normality(self, alpha=0.05):
        """Test for normality using Shapiro-Wilk test"""
        if self.data is None:
            logger.warning("No data available for normality test")
            return None
            
        numeric_data = self.data.select_dtypes(include=np.number)
        if numeric_data.empty:
            logger.warning("No numeric columns found in dataset")
            return pd.DataFrame()
            
        results = {}
        for column in numeric_data.columns:
            # Skip if too many values (Shapiro-Wilk limited to 5000 samples)
            if len(numeric_data[column].dropna()) > 5000:
                sample = numeric_data[column].dropna().sample(5000, random_state=42)
            else:
                sample = numeric_data[column].dropna()
                
            if len(sample) < 3:  # Need at least 3 values for test
                continue
                
            stat, p_value = stats.shapiro(sample)
            results[column] = {
                'statistic': stat,
                'p_value': p_value,
                'normal': p_value > alpha
            }
            
        return pd.DataFrame(results).T
        
    def perform_pca(self, n_components=2, scale=True):
        """Perform Principal Component Analysis"""
        if self.data is None:
            logger.warning("No data available for PCA")
            return None, None
            
        numeric_data = self.data.select_dtypes(include=np.number)
        if numeric_data.empty:
            logger.warning("No numeric columns found in dataset")
            return None, None
            
        # Handle missing values
        numeric_data = numeric_data.dropna()
        
        if scale:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
        else:
            scaled_data = numeric_data.values
            
        # Perform PCA
        pca = PCA(n_components=min(n_components, min(scaled_data.shape)))
        principal_components = pca.fit_transform(scaled_data)
        
        # Create DataFrame with principal components
        columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
        pca_df = pd.DataFrame(data=principal_components, columns=columns)
        
        # Return both the PCA results and the PCA object for explained variance
        return pca_df, pca
        
    def detect_outliers(self, method='zscore', threshold=3.0):
        """Detect outliers in numeric columns
        
        Methods:
            - 'zscore': Use Z-score method
            - 'iqr': Use Interquartile Range method
        """
        if self.data is None:
            logger.warning("No data available for outlier detection")
            return None
            
        numeric_data = self.data.select_dtypes(include=np.number)
        if numeric_data.empty:
            logger.warning("No numeric columns found in dataset")
            return pd.DataFrame()
            
        results = {}
        
        for column in numeric_data.columns:
            column_data = numeric_data[column].dropna()
            
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(column_data))
                outliers = column_data[z_scores > threshold]
            elif method == 'iqr':
                q1 = column_data.quantile(0.25)
                q3 = column_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (threshold * iqr)
                upper_bound = q3 + (threshold * iqr)
                outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]
            else:
                logger.error(f"Invalid outlier detection method: {method}")
                continue
                
            results[column] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(column_data)) * 100 if len(column_data) > 0 else 0,
                'min_outlier': outliers.min() if not outliers.empty else None,
                'max_outlier': outliers.max() if not outliers.empty else None
            }
            
        return pd.DataFrame(results).T 