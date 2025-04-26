import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataPreprocessor:
    def __init__(self):
        self.imputer = None
        self.scaler = None
        self.encoder = None
        
    def clean_column_names(self, df):
        """Sanitize DataFrame column names"""
        df.columns = [
            col.strip().replace(' ', '_').lower() 
            for col in df.columns
        ]
        return df
    
    def handle_missing_values(self, df, strategy='median'):
        """Advanced missing value handling"""
        numeric_cols = df.select_dtypes(include=np.number).columns
        self.imputer = SimpleImputer(strategy=strategy)
        df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        return df
    
    def encode_categorical(self, df, max_categories=10):
        """Smart categorical encoding"""
        cat_cols = df.select_dtypes(exclude=np.number).columns
        self.encoder = OneHotEncoder(
            max_categories=max_categories, 
            handle_unknown='ignore'
        )
        encoded = self.encoder.fit_transform(df[cat_cols])
        return pd.concat([
            df.drop(cat_cols, axis=1),
            pd.DataFrame(encoded.toarray(), 
                       columns=self.encoder.get_feature_names_out())
        ], axis=1)
    
    def normalize_data(self, df):
        """Feature scaling"""
        numeric_cols = df.select_dtypes(include=np.number).columns
        self.scaler = StandardScaler()
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df