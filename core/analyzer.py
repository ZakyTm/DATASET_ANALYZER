import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from .preprocessor import DataPreprocessor
import matplotlib
import threading
import os

class DataAnalyzer:
    def __init__(self):
        self.data = None
        self.preprocessor = DataPreprocessor()
        
    def set_data(self, data):
        self.data = self.preprocessor.clean_column_names(data)
        
    def generate_profile(self):
        # Force non-interactive mode more aggressively when in a background thread
        is_background_thread = threading.current_thread() != threading.main_thread()
        
        if is_background_thread:
            # Save original backend
            original_backend = matplotlib.get_backend()
            # Force Agg backend for non-interactive plots
            matplotlib.use('Agg', force=True)
            # Set environment variable to ensure no GUI is used
            os.environ['DISPLAY'] = ''
            
        try:
            # Create the report with minimal correlations to avoid plotting issues
            return ProfileReport(
                self.data,
                title="Dataset Report",
                explorative=True,
                minimal=is_background_thread,  # Use minimal mode in background threads
                correlations={
                    "auto": {"calculate": not is_background_thread},
                    "pearson": {"calculate": not is_background_thread},
                    "spearman": {"calculate": not is_background_thread},
                    "kendall": {"calculate": not is_background_thread},
                    "phi_k": {"calculate": False},
                    "cramers": {"calculate": False}
                },
                plot={
                    "correlation": {
                        "cmap": "RdBu",
                        "bad": "#000000"
                    },
                    "missing": {
                        "cmap": "RdBu"
                    }
                },
                interactions={
                    "continuous": False,
                },
                samples={"head": 5, "tail": 5}
            )
        finally:
            # Restore original backend if we changed it
            if is_background_thread:
                matplotlib.use(original_backend)
                # Remove environment variable
                if 'DISPLAY' in os.environ:
                    del os.environ['DISPLAY']
    
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