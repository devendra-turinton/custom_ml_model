#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Regression ML Pipeline
-----------------------------------
This script provides an end-to-end solution for regression machine learning tasks.
Users only need to provide data and specify the target variable.

Usage:
    python regression_pipeline.py --data_path "path/to/data.csv" --target "target_column_name"
"""

import argparse
import os
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import BaseEstimator, TransformerMixin

# Suppress common warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Custom transformer for outlier detection and handling."""
    
    def __init__(self, method='iqr', threshold=1.5, strategy='clip'):
        """
        Initialize outlier handler.
        
        Args:
            method (str): Method for outlier detection ('iqr' or 'zscore')
            threshold (float): Threshold for outlier detection
            strategy (str): Strategy for handling outliers ('clip', 'remove', or 'none')
        """
        self.method = method
        self.threshold = threshold
        self.strategy = strategy
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
    
    def fit(self, X, y=None):
        """
        Compute the outlier thresholds for each feature.
        
        Args:
            X (pd.DataFrame): Features
            y: Ignored
            
        Returns:
            self
        """
        for col in X.columns:
            if np.issubdtype(X[col].dtype, np.number):
                if self.method == 'iqr':
                    q1 = X[col].quantile(0.25)
                    q3 = X[col].quantile(0.75)
                    iqr = q3 - q1
                    self.lower_bounds_[col] = q1 - (self.threshold * iqr)
                    self.upper_bounds_[col] = q3 + (self.threshold * iqr)
                elif self.method == 'zscore':
                    mean = X[col].mean()
                    std = X[col].std()
                    self.lower_bounds_[col] = mean - (self.threshold * std)
                    self.upper_bounds_[col] = mean + (self.threshold * std)
        return self
    
    def transform(self, X):
        """
        Handle outliers in the data.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            pd.DataFrame: Transformed features
        """
        X_transformed = X.copy()
        
        if self.strategy == 'none':
            return X_transformed
        
        for col in self.lower_bounds_.keys():
            if col in X_transformed.columns:
                if self.strategy == 'clip':
                    X_transformed[col] = X_transformed[col].clip(
                        lower=self.lower_bounds_[col],
                        upper=self.upper_bounds_[col]
                    )
                elif self.strategy == 'remove':
                    mask = (
                        (X_transformed[col] >= self.lower_bounds_[col]) & 
                        (X_transformed[col] <= self.upper_bounds_[col])
                    )
                    X_transformed = X_transformed[mask]
        
        return X_transformed


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer for datetime feature extraction."""
    
    def __init__(self, date_features=True, cyclic_features=True, custom_ref_dates=None):
        """
        Initialize datetime feature extractor.
        
        Args:
            date_features (bool): Whether to extract date components
            cyclic_features (bool): Whether to extract cyclic features
            custom_ref_dates (dict): Custom reference dates for distance calculation
        """
        self.date_features = date_features
        self.cyclic_features = cyclic_features
        self.custom_ref_dates = custom_ref_dates or {}
        self.datetime_columns_ = []
    
    def fit(self, X, y=None):
        """
        Identify datetime columns.
        
        Args:
            X (pd.DataFrame): Features
            y: Ignored
            
        Returns:
            self
        """
        self.datetime_columns_ = []
        
        for col in X.columns:
            # Check if it's already a datetime type
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                self.datetime_columns_.append(col)
            # Try to convert string to datetime
            elif X[col].dtype == 'object':
                try:
                    pd.to_datetime(X[col], errors='raise')
                    self.datetime_columns_.append(col)
                except (ValueError, TypeError):
                    continue
        
        return self
    
    def transform(self, X):
        """
        Extract datetime features.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            pd.DataFrame: Transformed features with datetime features added
        """
        X_transformed = X.copy()
        
        for col in self.datetime_columns_:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(X_transformed[col]):
                try:
                    X_transformed[col] = pd.to_datetime(X_transformed[col], errors='coerce')
                except:
                    # If conversion fails, skip this column
                    continue
            
            # Basic date components
            if self.date_features:
                X_transformed[f'{col}_year'] = X_transformed[col].dt.year
                X_transformed[f'{col}_month'] = X_transformed[col].dt.month
                X_transformed[f'{col}_day'] = X_transformed[col].dt.day
                X_transformed[f'{col}_dayofweek'] = X_transformed[col].dt.dayofweek
                X_transformed[f'{col}_dayofyear'] = X_transformed[col].dt.dayofyear
                X_transformed[f'{col}_quarter'] = X_transformed[col].dt.quarter
                
                # Hour components if time data is present
                if (X_transformed[col].dt.hour > 0).any():
                    X_transformed[f'{col}_hour'] = X_transformed[col].dt.hour
                    X_transformed[f'{col}_minute'] = X_transformed[col].dt.minute
                
                # Flag features
                X_transformed[f'{col}_is_weekend'] = X_transformed[col].dt.dayofweek >= 5
                X_transformed[f'{col}_is_month_end'] = X_transformed[col].dt.is_month_end
            
            # Cyclic features to handle periodicity
            if self.cyclic_features:
                # Month has cycle of 12
                X_transformed[f'{col}_month_sin'] = np.sin(2 * np.pi * X_transformed[col].dt.month / 12)
                X_transformed[f'{col}_month_cos'] = np.cos(2 * np.pi * X_transformed[col].dt.month / 12)
                
                # Day of week has cycle of 7
                X_transformed[f'{col}_dayofweek_sin'] = np.sin(2 * np.pi * X_transformed[col].dt.dayofweek / 7)
                X_transformed[f'{col}_dayofweek_cos'] = np.cos(2 * np.pi * X_transformed[col].dt.dayofweek / 7)
                
                # Hour has cycle of 24 (if time data exists)
                if (X_transformed[col].dt.hour > 0).any():
                    X_transformed[f'{col}_hour_sin'] = np.sin(2 * np.pi * X_transformed[col].dt.hour / 24)
                    X_transformed[f'{col}_hour_cos'] = np.cos(2 * np.pi * X_transformed[col].dt.hour / 24)
            
            # Distance from reference dates
            for ref_name, ref_date in self.custom_ref_dates.items():
                ref_date = pd.to_datetime(ref_date)
                X_transformed[f'{col}_days_from_{ref_name}'] = (X_transformed[col] - ref_date).dt.days
            
            # Default time since epoch
            X_transformed[f'{col}_days_from_epoch'] = (X_transformed[col] - pd.Timestamp('1970-01-01')).dt.days
            
        return X_transformed


class RegressionPipeline:
    """End-to-end pipeline for regression tasks."""
    
    def __init__(self, data_path=None, target=None, df=None, test_size=0.2, random_state=42):
        """
        Initialize the pipeline.
        
        Args:
            data_path (str): Path to the data file
            target (str): Name of the target column
            df (pd.DataFrame): Data as DataFrame (alternative to data_path)
            test_size (float): Proportion of the dataset to be used as test set
            random_state (int): Random seed for reproducibility
        """
        self.data_path = data_path
        self.target = target
        self.df = df
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """
        Load data from file or use provided DataFrame.
        
        Returns:
            pd.DataFrame: Loaded data
        """
        if self.df is not None:
            return self.df
        
        if self.data_path is None:
            raise ValueError("Either data_path or df must be provided")
        
        file_ext = os.path.splitext(self.data_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                self.df = pd.read_csv(self.data_path)
            elif file_ext in ['.xls', '.xlsx']:
                self.df = pd.read_excel(self.data_path)
            elif file_ext == '.json':
                self.df = pd.read_json(self.data_path)
            elif file_ext == '.parquet':
                self.df = pd.read_parquet(self.data_path)
            else:
                raise ValueError(f"Unsupported file extension: {file_ext}")
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")
        
        return self.df
    
    def validate_data(self):
        """
        Validate the input data and target.
        
        Returns:
            bool: True if validation passes
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if self.target is None:
            raise ValueError("Target column name must be provided")
        
        if self.target not in self.df.columns:
            raise ValueError(f"Target column '{self.target}' not found in data")
        
        if len(self.df) == 0:
            raise ValueError("Dataset is empty")
        
        # Check if target has valid numeric values
        if not pd.api.types.is_numeric_dtype(self.df[self.target]):
            raise ValueError(f"Target column '{self.target}' must contain numeric values")
        
        # Check for at least some valid values in target
        if self.df[self.target].isna().all():
            raise ValueError(f"Target column '{self.target}' contains only NaN values")
        
        return True
    
    def preprocess_data(self):
        """
        Preprocess the data for machine learning.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Split into features and target
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Identify column types
        numeric_features = []
        categorical_features = []
        datetime_features = []
        
        for col in X.columns:
            # Check for datetime
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                datetime_features.append(col)
            # Try to convert to datetime
            elif X[col].dtype == 'object':
                try:
                    pd.to_datetime(X[col], errors='raise')
                    datetime_features.append(col)
                    continue
                except (ValueError, TypeError):
                    pass
            
            # Check numeric vs categorical
            if pd.api.types.is_numeric_dtype(X[col]):
                if X[col].nunique() < 10 and X[col].nunique() / len(X[col]) < 0.05:
                    categorical_features.append(col)  # Treat as categorical if few unique values
                else:
                    numeric_features.append(col)
            else:
                categorical_features.append(col)
        
        # Create preprocessing steps for different column types
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('outlier', OutlierHandler(method='iqr', threshold=1.5, strategy='clip')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        datetime_transformer = DatetimeFeatureExtractor(
            date_features=True,
            cyclic_features=True,
            custom_ref_dates={'today': datetime.now().strftime('%Y-%m-%d')}
        )
        
        # Combine all preprocessing steps
        preprocessor_steps = []
        
        if numeric_features:
            preprocessor_steps.append(('numeric', numeric_transformer, numeric_features))
        
        if categorical_features:
            preprocessor_steps.append(('categorical', categorical_transformer, categorical_features))
        
        if datetime_features:
            preprocessor_steps.append(('datetime', datetime_transformer, datetime_features))
        
        self.preprocessor = ColumnTransformer(
            transformers=preprocessor_steps,
            remainder='drop'  # Drop any other columns
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """
        Train multiple regression models.
        
        Returns:
            dict: Trained models dictionary
        """
        if self.preprocessor is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        # Define models to train
        models = {
            'linear_regression': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', LinearRegression())
            ]),
            'ridge': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', Ridge(alpha=1.0))
            ]),
            'lasso': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', Lasso(alpha=0.1))
            ]),
            'elastic_net': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', ElasticNet(alpha=0.1, l1_ratio=0.5))
            ]),
            'decision_tree': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', DecisionTreeRegressor(random_state=self.random_state))
            ]),
            'random_forest': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=self.random_state))
            ]),
            'gradient_boosting': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=self.random_state))
            ])
        }
        
        # Try to add SVR and MLPRegressor models if data is not too large
        if len(self.X_train) < 10000:  # Avoid training complex models on large datasets
            models['svr'] = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', SVR(kernel='rbf', gamma='scale'))
            ])
            
            models['mlp'] = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=self.random_state))
            ])
        
        # Train all models
        print("Training models...")
        self.models = {}
        
        for name, model in models.items():
            try:
                print(f"  Training {name}...")
                model.fit(self.X_train, self.y_train)
                self.models[name] = model
            except Exception as e:
                print(f"  Failed to train {name}: {str(e)}")
        
        return self.models
    
    def evaluate_models(self):
        """
        Evaluate trained models.
        
        Returns:
            pd.DataFrame: Evaluation results
        """
        if not self.models:
            raise ValueError("No trained models. Call train_models() first.")
        
        print("Evaluating models...")
        results = []
        
        for name, model in self.models.items():
            try:
                # Make predictions
                y_train_pred = model.predict(self.X_train)
                y_test_pred = model.predict(self.X_test)
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
                train_mae = mean_absolute_error(self.y_train, y_train_pred)
                test_mae = mean_absolute_error(self.y_test, y_test_pred)
                train_r2 = r2_score(self.y_train, y_train_pred)
                test_r2 = r2_score(self.y_test, y_test_pred)
                
                # Store results
                results.append({
                    'model': name,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_r2': train_r2,
                    'test_r2': test_r2
                })
                
                print(f"  {name}:")
                print(f"    Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
                print(f"    Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
                
            except Exception as e:
                print(f"  Failed to evaluate {name}: {str(e)}")
        
        # Convert to DataFrame
        self.results = pd.DataFrame(results)
        
        # Identify best model based on test R²
        if not self.results.empty:
            best_model_name = self.results.loc[self.results['test_r2'].idxmax(), 'model']
            self.best_model = self.models[best_model_name]
            print(f"\nBest model: {best_model_name} (Test R² = {self.results.loc[self.results['model'] == best_model_name, 'test_r2'].values[0]:.4f})")
        
        return self.results
    
    def save_model(self, output_dir='models'):
        """
        Save the best model and preprocessing pipeline.
        
        Args:
            output_dir (str): Directory to save the model
            
        Returns:
            str: Path to the saved model
        """
        if self.best_model is None:
            raise ValueError("No best model selected. Call evaluate_models() first.")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = os.path.join(output_dir, f"regression_model_{timestamp}.pkl")
        
        # Save model
        with open(model_filename, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        print(f"Model saved to {model_filename}")
        
        # Save model info
        results_filename = os.path.join(output_dir, f"model_results_{timestamp}.csv")
        self.results.to_csv(results_filename, index=False)
        
        print(f"Results saved to {results_filename}")
        
        return model_filename
    
    def run_pipeline(self):
        """
        Run the complete pipeline.
        
        Returns:
            tuple: (best_model, evaluation_results)
        """
        print("Starting regression ML pipeline...")
        
        # Load and validate data
        print("\nLoading data...")
        self.load_data()
        self.validate_data()
        print(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        # Preprocess data
        print("\nPreprocessing data...")
        self.preprocess_data()
        print(f"Train set: {self.X_train.shape[0]} rows")
        print(f"Test set: {self.X_test.shape[0]} rows")
        
        # Train models
        print("\nTraining models...")
        self.train_models()
        print(f"Trained {len(self.models)} models")
        
        # Evaluate models
        print("\nEvaluating models...")
        self.evaluate_models()
        
        # Save the best model
        print("\nSaving model...")
        self.save_model()
        
        print("\nPipeline complete!")
        
        return self.best_model, self.results


def main():
    """Main function to run the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Regression ML Pipeline')
    parser.add_argument('--data_path', type=str, help='Path to data file')
    parser.add_argument('--target', type=str, help='Name of target column')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of test set')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='models', help='Output directory for saved models')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = RegressionPipeline(
        data_path=args.data_path,
        target=args.target,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    try:
        best_model, results = pipeline.run_pipeline()
        print("\nTop 3 models by test R²:")
        print(results.sort_values('test_r2', ascending=False).head(3)[['model', 'test_rmse', 'test_r2']])
    except Exception as e:
        print(f"\nError running pipeline: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    main()