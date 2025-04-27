import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from .preprocessor import DataPreprocessor
import matplotlib
import threading

class DataAnalyzer:
    def __init__(self):
        self.data = None
        self.preprocessor = DataPreprocessor()
        
    def set_data(self, data):
        self.data = self.preprocessor.clean_column_names(data)
        
    def generate_profile(self):
        # Check if we're in a background thread and switch matplotlib backend if needed
        if threading.current_thread() != threading.main_thread():
            # Save original backend
            original_backend = matplotlib.get_backend()
            # Switch to non-interactive backend for background thread
            matplotlib.use('Agg')
            
        try:
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
        finally:
            # Restore original backend if we changed it
            if threading.current_thread() != threading.main_thread():
                matplotlib.use(original_backend)
    
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