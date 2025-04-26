import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from .preprocessor import DataPreprocessor

class DataAnalyzer:
    def __init__(self):
        self.data = None
        self.preprocessor = DataPreprocessor()
        
    def set_data(self, data):
        self.data = self.preprocessor.clean_column_names(data)
        
    def generate_profile(self):
        return ProfileReport(
            self.data,
            title="Dataset Report",
            explorative=True,
            correlations={
                "auto": {"calculate": True},
                "pearson": {"calculate": True},
                "spearman": {"calculate": True},
                "kendall": {"calculate": True}
            }
        )
    
    def calculate_correlations(self, method='pearson'):
        """Enhanced correlation analysis"""
        return self.data.select_dtypes(include=np.number).corr(method=method)
    
    def identify_issues(self):
        """Advanced data quality checks"""
        issues = {
            'high_missing': self.data.columns[self.data.isna().mean() > 0.5].tolist(),
            'low_variance': self._find_low_variance_features(),
            'duplicate_rows': self.data.duplicated().sum(),
            'constant_features': self._find_constant_features()
        }
        return issues
    
    def _find_low_variance_features(self, threshold=0.01):
        numeric = self.data.select_dtypes(include=np.number)
        return numeric.columns[
            (numeric.std() / numeric.mean()).abs() < threshold
        ].tolist()
    
    def _find_constant_features(self):
        return self.data.columns[self.data.nunique() == 1].tolist()