"""
Machine Learning modeling module for dataset analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from utils.helpers import get_logger
from datetime import datetime

logger = get_logger()

class ModelingTask:
    REGRESSION = 'regression'
    CLASSIFICATION = 'classification'
    CLUSTERING = 'clustering'

class MLModel:
    """Machine Learning modeling capabilities"""
    
    def __init__(self, data=None):
        self.data = data
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.pipeline = None
        self.model_type = None
        self.results = {}
        self.feature_importance = None
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
    def set_data(self, data):
        """Set the dataframe for modeling"""
        self.data = data
        
    def prepare_data(self, target_column, feature_columns=None, test_size=0.25, random_state=42):
        """Prepare data for modeling by splitting into features and target"""
        if self.data is None:
            logger.error("No data available for modeling")
            return False
            
        if target_column not in self.data.columns:
            logger.error(f"Target column '{target_column}' not found in data")
            return False
            
        # If feature columns not specified, use all columns except target
        if feature_columns is None:
            feature_columns = [col for col in self.data.columns if col != target_column]
        
        # Get feature matrix and target vector
        self.X = self.data[feature_columns]
        self.y = self.data[target_column]
        
        # Determine task type based on target variable
        if self.y.dtype in ['int64', 'float64'] and self.y.nunique() > 10:
            self.model_type = ModelingTask.REGRESSION
            logger.info(f"Target '{target_column}' has {self.y.nunique()} unique values, treating as regression task")
        else:
            self.model_type = ModelingTask.CLASSIFICATION
            logger.info(f"Target '{target_column}' has {self.y.nunique()} unique values, treating as classification task")
        
        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        return True
    
    def create_preprocessor(self):
        """Create a preprocessor for numerical and categorical columns"""
        if self.X is None:
            logger.error("Features not prepared. Call prepare_data first.")
            return None
            
        # Identify numeric and categorical columns
        numeric_features = self.X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.X.select_dtypes(include=['object', 'category']).columns
        
        # Create transformers for each type
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor
    
    def train_model(self, model_name='auto', hyperparams=None):
        """Train a model on the prepared data"""
        if self.X_train is None or self.y_train is None:
            logger.error("Data not prepared. Call prepare_data first.")
            return False
            
        # Create the preprocessor
        preprocessor = self.create_preprocessor()
        
        # Choose model based on task type
        if model_name == 'auto':
            if self.model_type == ModelingTask.REGRESSION:
                model_name = 'random_forest'
            else:
                model_name = 'random_forest'
        
        # Regression models
        regression_models = {
            'linear': LinearRegression(),
            'decision_tree': DecisionTreeRegressor(random_state=42),
            'random_forest': RandomForestRegressor(random_state=42),
            'svr': SVR()
        }
        
        # Classification models
        classification_models = {
            'logistic': LogisticRegression(random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'svc': SVC(probability=True, random_state=42)
        }
        
        # Select model based on task type
        if self.model_type == ModelingTask.REGRESSION:
            if model_name not in regression_models:
                logger.error(f"Unknown regression model: {model_name}")
                return False
            model = regression_models[model_name]
        else:
            if model_name not in classification_models:
                logger.error(f"Unknown classification model: {model_name}")
                return False
            model = classification_models[model_name]
        
        # Apply hyperparameters if provided
        if hyperparams and isinstance(hyperparams, dict):
            model.set_params(**hyperparams)
        
        # Create pipeline with preprocessor and model
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Fit the model
        logger.info(f"Training {model_name} model")
        self.pipeline.fit(self.X_train, self.y_train)
        self.model = model
        
        # Evaluate the model
        self._evaluate_model()
        
        # Extract feature importance if available
        self._extract_feature_importance()
        
        return True
    
    def _evaluate_model(self):
        """Evaluate the trained model on test data"""
        if self.model is None or self.pipeline is None:
            logger.error("No trained model available")
            return
            
        # Make predictions
        y_pred = self.pipeline.predict(self.X_test)
        
        # Regression metrics
        if self.model_type == ModelingTask.REGRESSION:
            self.results = {
                'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                'mae': mean_absolute_error(self.y_test, y_pred),
                'r2': r2_score(self.y_test, y_pred)
            }
            logger.info(f"Model performance: RMSE={self.results['rmse']:.4f}, MAE={self.results['mae']:.4f}, R²={self.results['r2']:.4f}")
        
        # Classification metrics
        else:
            # Binary classification
            if len(np.unique(self.y)) <= 2:
                self.results = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred, average='binary'),
                    'recall': recall_score(self.y_test, y_pred, average='binary'),
                    'f1': f1_score(self.y_test, y_pred, average='binary'),
                    'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
                }
            # Multiclass classification
            else:
                self.results = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred, average='weighted'),
                    'recall': recall_score(self.y_test, y_pred, average='weighted'),
                    'f1': f1_score(self.y_test, y_pred, average='weighted'),
                    'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
                }
            
            logger.info(f"Model performance: Accuracy={self.results['accuracy']:.4f}, F1={self.results['f1']:.4f}")
            
    def _extract_feature_importance(self):
        """Extract feature importance from the model if available"""
        if hasattr(self.model, 'feature_importances_'):
            # Get feature importance from model
            importance = self.model.feature_importances_
            
            # Get feature names after preprocessing
            if hasattr(self.pipeline, 'named_steps') and 'preprocessor' in self.pipeline.named_steps:
                preprocessor = self.pipeline.named_steps['preprocessor']
                if hasattr(preprocessor, 'transformers_'):
                    # This is complicated because of one-hot encoding
                    # For simplicity, we'll use the original feature names
                    feature_names = self.X.columns
                    self.feature_importance = dict(zip(feature_names, importance))
                    return
            
            # Fallback if preprocessing makes it hard to get feature names
            feature_names = self.X.columns
            self.feature_importance = dict(zip(feature_names, importance))
    
    def cross_validate(self, cv=5):
        """Perform cross-validation on the model"""
        if self.pipeline is None:
            logger.error("No model pipeline created. Call train_model first.")
            return None
            
        if self.X is None or self.y is None:
            logger.error("Data not prepared. Call prepare_data first.")
            return None
            
        logger.info(f"Performing {cv}-fold cross-validation")
        
        if self.model_type == ModelingTask.REGRESSION:
            cv_results = cross_val_score(self.pipeline, self.X, self.y, cv=cv, scoring='r2')
            metric_name = 'r2'
        else:
            cv_results = cross_val_score(self.pipeline, self.X, self.y, cv=cv, scoring='f1_weighted')
            metric_name = 'f1_weighted'
            
        cv_mean = cv_results.mean()
        cv_std = cv_results.std()
        
        logger.info(f"Cross-validation {metric_name}: {cv_mean:.4f} ± {cv_std:.4f}")
        
        return {
            'cv_scores': cv_results.tolist(),
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'metric': metric_name
        }
    
    def optimize_hyperparameters(self, param_grid=None):
        """Optimize model hyperparameters using grid search"""
        if self.pipeline is None:
            logger.error("No model pipeline created. Call train_model first.")
            return None
            
        if param_grid is None:
            # Default parameter grids for different models
            if isinstance(self.model, RandomForestRegressor) or isinstance(self.model, RandomForestClassifier):
                param_grid = {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [None, 10, 20, 30]
                }
            elif isinstance(self.model, DecisionTreeRegressor) or isinstance(self.model, DecisionTreeClassifier):
                param_grid = {
                    'model__max_depth': [None, 5, 10, 15, 20],
                    'model__min_samples_split': [2, 5, 10]
                }
            elif isinstance(self.model, LinearRegression):
                # Linear regression doesn't have hyperparameters to tune
                logger.info("LinearRegression doesn't have hyperparameters to tune")
                return None
            elif isinstance(self.model, LogisticRegression):
                param_grid = {
                    'model__C': [0.01, 0.1, 1, 10, 100],
                    'model__solver': ['liblinear', 'saga']
                }
            else:
                logger.warning("No default parameter grid for this model type")
                return None
        
        logger.info("Optimizing hyperparameters using grid search")
        
        # Set up grid search
        if self.model_type == ModelingTask.REGRESSION:
            scoring = 'r2'
        else:
            scoring = 'f1_weighted'
            
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=5,
            scoring=scoring,
            n_jobs=-1
        )
        
        # Fit grid search
        grid_search.fit(self.X, self.y)
        
        # Get best parameters and score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best {scoring} score: {best_score:.4f}")
        
        # Train model with best parameters
        self.train_model(hyperparams={k.replace('model__', ''): v for k, v in best_params.items()})
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'scoring': scoring
        }
    
    def save_model(self, filename=None):
        """Save the trained model to disk"""
        if self.pipeline is None:
            logger.error("No trained model to save")
            return None
            
        if filename is None:
            # Create default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_type = self.model.__class__.__name__
            filename = f"{model_type}_{timestamp}.joblib"
            
        model_path = self.models_dir / filename
        
        # Save the model
        joblib.dump(self.pipeline, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path):
        """Load a saved model from disk"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
                
            self.pipeline = joblib.load(model_path)
            
            # Extract the model from the pipeline
            if hasattr(self.pipeline, 'named_steps') and 'model' in self.pipeline.named_steps:
                self.model = self.pipeline.named_steps['model']
                
                # Determine model type
                if isinstance(self.model, (LinearRegression, DecisionTreeRegressor, RandomForestRegressor, SVR)):
                    self.model_type = ModelingTask.REGRESSION
                else:
                    self.model_type = ModelingTask.CLASSIFICATION
                    
                logger.info(f"Loaded {self.model.__class__.__name__} model from {model_path}")
                return True
            else:
                logger.error(f"Invalid model pipeline structure in {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, data):
        """Make predictions using the trained model"""
        if self.pipeline is None:
            logger.error("No trained model available for predictions")
            return None
            
        # Convert to DataFrame if it's not already
        if not isinstance(data, pd.DataFrame):
            logger.warning("Input is not a DataFrame. Converting to DataFrame.")
            data = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
            
        # Check if data has required columns
        missing_columns = set(self.X.columns) - set(data.columns)
        if missing_columns:
            logger.error(f"Input data is missing required columns: {missing_columns}")
            return None
            
        # Make predictions
        try:
            predictions = self.pipeline.predict(data[self.X.columns])
            
            # For classification, also get probabilities if available
            if self.model_type == ModelingTask.CLASSIFICATION and hasattr(self.pipeline, 'predict_proba'):
                probabilities = self.pipeline.predict_proba(data[self.X.columns])
                return {
                    'predictions': predictions,
                    'probabilities': probabilities
                }
                
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None
    
    def plot_feature_importance(self, top_n=10):
        """Plot feature importance if available"""
        if self.feature_importance is None:
            logger.error("No feature importance available. Train a model that supports feature importance.")
            return None
            
        # Sort feature importance
        sorted_importance = dict(sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        # Limit to top_n features
        if len(sorted_importance) > top_n:
            sorted_importance = dict(list(sorted_importance.items())[:top_n])
            
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create bar plot
        plt.barh(list(sorted_importance.keys()), list(sorted_importance.values()))
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = self.models_dir / f"feature_importance_{timestamp}.png"
        plt.savefig(fig_path)
        
        return fig_path
    
    def plot_regression_results(self):
        """Plot actual vs predicted values for regression models"""
        if self.model_type != ModelingTask.REGRESSION:
            logger.error("This method is only for regression models")
            return None
            
        if self.X_test is None or self.y_test is None:
            logger.error("No test data available")
            return None
            
        # Make predictions
        y_pred = self.pipeline.predict(self.X_test)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot actual vs predicted
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(min(self.y_test), min(y_pred))
        max_val = max(max(self.y_test), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Values')
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = self.models_dir / f"regression_results_{timestamp}.png"
        plt.savefig(fig_path)
        
        return fig_path
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix for classification models"""
        if self.model_type != ModelingTask.CLASSIFICATION:
            logger.error("This method is only for classification models")
            return None
            
        if 'confusion_matrix' not in self.results:
            logger.error("No confusion matrix available")
            return None
            
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Plot confusion matrix
        cm = np.array(self.results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = self.models_dir / f"confusion_matrix_{timestamp}.png"
        plt.savefig(fig_path)
        
        return fig_path 