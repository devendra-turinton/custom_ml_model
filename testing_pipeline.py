#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Testing pipeline for machine learning models.
This script loads a trained model and makes predictions on new data.
Supports both regression and classification models.
"""

import os
import sys
import argparse
import pickle
import json
import logging
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, roc_auc_score
)

# Suppress common warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def get_next_version_dir(base_output_dir, model_id, max_versions=5):
    """
    Get the next version directory for model outputs.
    
    Args:
        base_output_dir (str): Base output directory
        model_id (str): Model ID
        max_versions (int): Maximum number of versions to keep
        
    Returns:
        tuple: (next_version_dir, next_version_number)
    """
    # Create model directory if it doesn't exist
    model_dir = os.path.join(base_output_dir, model_id)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, "v1"), 1
    
    # Get existing version directories
    version_dirs = []
    for item in os.listdir(model_dir):
        if os.path.isdir(os.path.join(model_dir, item)) and item.startswith('v'):
            try:
                version_num = int(item[1:])  # Extract number part from 'v1', 'v2', etc.
                version_dirs.append((item, version_num))
            except ValueError:
                # Skip directories that don't follow the vN pattern
                continue
    
    # Sort by version number
    version_dirs.sort(key=lambda x: x[1])
    
    # If no valid versions exist, start with v1
    if not version_dirs:
        return os.path.join(model_dir, "v1"), 1
    
    # Get the next version number
    next_version_num = version_dirs[-1][1] + 1
    next_version = f"v{next_version_num}"
    
    # Check if we need to delete old versions
    if len(version_dirs) >= max_versions:
        # Keep the newest (max_versions-1) to make room for the new one
        versions_to_delete = version_dirs[:-(max_versions-1)]
        for version, _ in versions_to_delete:
            dir_to_delete = os.path.join(model_dir, version)
            try:
                import shutil
                shutil.rmtree(dir_to_delete)
                print(f"Deleted old version directory: {dir_to_delete}")
            except Exception as e:
                print(f"Error deleting {dir_to_delete}: {str(e)}")
    
    return os.path.join(model_dir, next_version), next_version_num

class ModelPredictor:
    """Base class for model prediction."""

    def __init__(self, model_dir=None, output_dir=None, model_id=None):
        """
        Initialize predictor with model ID or directory.
        
        Args:
            model_dir (str): Directory containing the trained model and metadata
            output_dir (str): Directory to save prediction results and logs
            model_id (str): Unique identifier for the model
        """
        # Initialize timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use home directory 
        home_dir = os.path.expanduser("~")
        
        # Store model_id
        self.model_id = model_id
        
        # Set up paths based on project structure
        base_data_dir = os.path.join(home_dir, "custom_ml_model", "custom_ml_data")
        
        # Determine model_dir if model_id is provided but model_dir isn't
        if model_id and not model_dir:
            # Look for model in training output directory
            train_out_dir = os.path.join(base_data_dir, "training", "output", model_id)
            
            # Find the latest version folder
            if os.path.exists(train_out_dir):
                version_dirs = []
                for item in os.listdir(train_out_dir):
                    if os.path.isdir(os.path.join(train_out_dir, item)) and item.startswith('v'):
                        try:
                            version_num = int(item[1:])
                            version_dirs.append((item, version_num))
                        except ValueError:
                            continue
                
                if version_dirs:
                    # Sort and use the latest version
                    version_dirs.sort(key=lambda x: x[1], reverse=True)
                    latest_version = version_dirs[0][0]
                    model_dir = os.path.join(train_out_dir, latest_version)
        
        # Set model directory
        if not model_dir:
            raise ValueError("Either model_dir or model_id must be provided")
        self.model_dir = model_dir
        
        # Set up versioned output directory
        if model_id:
            # Base output directory for testing results
            test_out_base = os.path.join(base_data_dir, "testing", "output")
            
            # Get next version directory
            self.output_dir, version_num = get_next_version_dir(test_out_base, model_id)
            self.version = f"v{version_num}"
        else:
            # Default output dir if no model_id
            if output_dir:
                self.output_dir = output_dir
            else:
                self.output_dir = os.path.join(os.path.dirname(model_dir), f"predictions_{self.timestamp}")
            self.version = "v1"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logger BEFORE using it
        self.logger = self._setup_logging()
        
        # Initialize attributes
        self.model = None
        self.metadata = None
        self.problem_type = None
        self.prediction_metadata = {
            'timestamp': self.timestamp,
            'model_dir': model_dir,
            'model_id': model_id,
            'version': self.version,
            'prediction_data': {},
            'preprocessing': {},
            'results': {}
        }

    def _setup_logging(self):
        """Set up logging configuration."""
        # Configure logging
        logger = logging.getLogger('model_predictor')
        logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        if logger.handlers:
            logger.handlers.clear()
        
        # Log file path inside the output directory
        log_file = os.path.join(self.output_dir, "prediction.log")
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Console handler for important info
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info(f"Logging initialized. Log file: {log_file}")
        return logger

    def auto_detect_test_data(self):
        """
        Auto-detect test data file path based on model_id.
        
        Returns:
            str: Path to the test data file
        """
        if not self.model_id:
            self.logger.warning("No model_id provided, cannot auto-detect test data")
            return None
        
        # Use home directory 
        home_dir = os.path.expanduser("~")
        
        # Construct the expected test data path
        test_data_path = os.path.join(
            home_dir,
            "custom_ml_model",
            "custom_ml_data", 
            "testing", 
            "input", 
            self.model_id, 
            "v1", 
            "input_data.csv"
        )
        
        self.logger.info(f"Looking for test data at: {test_data_path}")
        
        # Check if the file exists
        if os.path.exists(test_data_path):
            self.logger.info(f"Auto-detected test data at: {test_data_path}")
            return test_data_path
        
        # If the exact file doesn't exist, try the training data path
        alt_test_data_path = os.path.join(
            home_dir,
            "custom_ml_model",
            "custom_ml_data", 
            "training", 
            "input", 
            self.model_id, 
            "v1", 
            "input_data.csv"
        )
        
        if os.path.exists(alt_test_data_path):
            self.logger.info(f"Found test data in training input: {alt_test_data_path}")
            return alt_test_data_path
        
        self.logger.warning(f"Could not find test data for model_id {self.model_id}")
        self.logger.info(f"Tried paths: {test_data_path} and {alt_test_data_path}")
        
        return None

    def load_model(self):
        """
        Load the trained model and metadata.
        
        Returns:
            tuple: (model, metadata)
        """
        self.logger.info("Looking for model files...")
        
        # Find the model file (.pkl) and metadata file (.json)
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
        metadata_files = [f for f in os.listdir(self.model_dir) if f.endswith('.json')]
        
        if not model_files:
            error_msg = f"No model files (.pkl) found in {self.model_dir}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        if not metadata_files:
            error_msg = f"No metadata files (.json) found in {self.model_dir}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Sort by timestamp (newest first) if multiple files exist
        model_files.sort(reverse=True)
        metadata_files.sort(reverse=True)
        
        model_path = os.path.join(self.model_dir, model_files[0])
        metadata_path = os.path.join(self.model_dir, metadata_files[0])
        
        self.logger.info(f"Loading model from {model_path}")
        self.logger.info(f"Loading metadata from {metadata_path}")
        
        try:
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Determine problem type from metadata
            self.problem_type = self.metadata.get('parameters', {}).get('problem_type')
            if not self.problem_type:
                raise ValueError("Could not determine problem type from metadata")
            
            self.logger.info(f"Problem type: {self.problem_type}")
            self.logger.info("Model and metadata loaded successfully")
            
            # Update prediction metadata
            self.prediction_metadata['model_info'] = {
                'model_file': model_path,
                'metadata_file': metadata_path,
                'problem_type': self.problem_type,
                'best_model': self.metadata.get('best_model', {}).get('name', 'unknown')
            }
            
            return self.model, self.metadata
            
        except Exception as e:
            error_msg = f"Error loading model or metadata: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    def load_data(self, data_path=None, df=None):
        """
        Load prediction data from file or DataFrame.
        
        Args:
            data_path (str): Path to data file
            df (pd.DataFrame): DataFrame to use for prediction
            
        Returns:
            pd.DataFrame: Loaded data
        """
        self.logger.info("Loading prediction data...")
        
        if df is not None:
            self.logger.info("Using provided DataFrame")
            self.prediction_metadata['prediction_data']['source'] = 'provided_dataframe'
            self.prediction_metadata['prediction_data']['shape'] = df.shape
            self.prediction_df = df
            return df
        
        if data_path is None:
            error_msg = "Either data_path or df must be provided"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        file_ext = os.path.splitext(data_path)[1].lower()
        self.prediction_metadata['prediction_data']['source'] = data_path
        self.prediction_metadata['prediction_data']['file_extension'] = file_ext
        
        try:
            start_time = datetime.now()
            
            if file_ext == '.csv':
                self.logger.info(f"Loading CSV file: {data_path}")
                df = pd.read_csv(data_path)
            elif file_ext in ['.xls', '.xlsx']:
                self.logger.info(f"Loading Excel file: {data_path}")
                df = pd.read_excel(data_path)
            elif file_ext == '.json':
                self.logger.info(f"Loading JSON file: {data_path}")
                df = pd.read_json(data_path)
            elif file_ext == '.parquet':
                self.logger.info(f"Loading Parquet file: {data_path}")
                df = pd.read_parquet(data_path)
            else:
                error_msg = f"Unsupported file extension: {file_ext}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            load_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Data loaded in {load_time:.2f} seconds")
            
            # Update metadata
            self.prediction_metadata['prediction_data']['shape'] = df.shape
            self.prediction_metadata['prediction_data']['loading_time_seconds'] = load_time
            self.prediction_metadata['prediction_data']['columns'] = list(df.columns)
            
            self.prediction_df = df
            return df
            
        except Exception as e:
            error_msg = f"Error loading prediction data: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    def validate_data(self):
        """
        Validate the prediction data against training metadata.
        
        Returns:
            bool: True if validation passes
        """
        self.logger.info("Validating prediction data against training metadata...")
        
        validation_results = {
            'passed': True,
            'warnings': [],
            'errors': []
        }
        
        # Check if model and metadata are loaded
        if self.model is None or self.metadata is None:
            error_msg = "Model and metadata must be loaded first. Call load_model() before validate_data()."
            self.logger.error(error_msg)
            validation_results['passed'] = False
            validation_results['errors'].append(error_msg)
            raise ValueError(error_msg)
        
        # Check if prediction data is loaded
        if not hasattr(self, 'prediction_df') or self.prediction_df is None:
            error_msg = "Prediction data not loaded. Call load_data() first."
            self.logger.error(error_msg)
            validation_results['passed'] = False
            validation_results['errors'].append(error_msg)
            raise ValueError(error_msg)
        
        # Get target column from metadata
        target_column = self.metadata.get('parameters', {}).get('target_column')
        if not target_column:
            # Try to find target column in preprocessing metadata
            target_column = self.metadata.get('preprocessing', {}).get('target_column')
            
        # Check if target column is present in prediction data
        target_present = target_column in self.prediction_df.columns
        if not target_present:
            warning_msg = f"Target column '{target_column}' not found in prediction data. Will only generate predictions."
            self.logger.warning(warning_msg)
            validation_results['warnings'].append(warning_msg)
        else:
            # If target is present, validate target type
            if self.problem_type == 'regression' and not pd.api.types.is_numeric_dtype(self.prediction_df[target_column]):
                warning_msg = f"Target column '{target_column}' should be numeric for regression, but found {self.prediction_df[target_column].dtype}."
                self.logger.warning(warning_msg)
                validation_results['warnings'].append(warning_msg)
            
            if self.problem_type == 'classification':
                # For classification, check if values are within expected classes
                class_encoding = self.metadata.get('preprocessing', {}).get('class_encoding', {})
                if class_encoding:
                    valid_classes = list(class_encoding.keys())
                    invalid_classes = [c for c in self.prediction_df[target_column].unique() if c not in valid_classes]
                    if invalid_classes:
                        warning_msg = f"Found {len(invalid_classes)} unexpected classes in target column: {invalid_classes[:5]}..."
                        self.logger.warning(warning_msg)
                        validation_results['warnings'].append(warning_msg)
        
        # Check expected columns against columns in the pipeline
        # In scikit-learn pipelines, feature names might be embedded in the preprocessor
        try:
            # Extract feature names from model pipeline if available
            if hasattr(self.model, 'feature_names_in_'):
                expected_features = list(self.model.feature_names_in_)
            elif hasattr(self.model, 'named_steps') and 'preprocessor' in self.model.named_steps:
                preprocessor = self.model.named_steps['preprocessor']
                if hasattr(preprocessor, 'feature_names_in_'):
                    expected_features = list(preprocessor.feature_names_in_)
                else:
                    # Fallback to metadata
                    expected_features = self.metadata.get('preprocessing', {}).get('feature_types', {}).get('numeric_features', []) + \
                                        self.metadata.get('preprocessing', {}).get('feature_types', {}).get('categorical_features', [])
            else:
                # Fallback to metadata
                expected_features = self.metadata.get('preprocessing', {}).get('feature_types', {}).get('numeric_features', []) + \
                                    self.metadata.get('preprocessing', {}).get('feature_types', {}).get('categorical_features', [])
            
            # Filter out target column
            if target_column in expected_features:
                expected_features.remove(target_column)
            
            # Check for missing features
            prediction_features = [col for col in self.prediction_df.columns if col != target_column]
            missing_features = [f for f in expected_features if f not in prediction_features]
            
            if missing_features:
                error_msg = f"Missing required features: {missing_features}"
                self.logger.error(error_msg)
                validation_results['passed'] = False
                validation_results['errors'].append(error_msg)
            
            # Check for extra features - these are ok but should be noted
            extra_features = [f for f in prediction_features if f not in expected_features]
            if extra_features:
                warning_msg = f"Found {len(extra_features)} extra features not in training data: {extra_features[:10]}..."
                self.logger.warning(warning_msg)
                validation_results['warnings'].append(warning_msg)
            
        except Exception as e:
            warning_msg = f"Could not validate feature names: {str(e)}"
            self.logger.warning(warning_msg)
            validation_results['warnings'].append(warning_msg)
        
        # Check for missing values
        missing_values = self.prediction_df.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0]
        if not columns_with_missing.empty:
            warning_msg = f"Found missing values in {len(columns_with_missing)} columns. Imputation will be applied based on training data."
            self.logger.warning(warning_msg)
            for col, count in columns_with_missing.items():
                self.logger.warning(f"  {col}: {count} missing values ({count/len(self.prediction_df)*100:.2f}%)")
            validation_results['warnings'].append(warning_msg)
            validation_results['missing_values'] = columns_with_missing.to_dict()
        
        # Data type validation
        if 'data' in self.metadata and 'dtypes' in self.metadata['data']:
            dtype_issues = []
            train_dtypes = self.metadata['data']['dtypes']
            
            for col, dtype_str in train_dtypes.items():
                if col in self.prediction_df.columns:
                    train_dtype = dtype_str
                    pred_dtype = str(self.prediction_df[col].dtype)
                    
                    # Check for fundamental type mismatches (e.g., number vs. string)
                    numeric_dtypes = ['int', 'float', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                    train_is_numeric = any(num_type in train_dtype.lower() for num_type in numeric_dtypes)
                    pred_is_numeric = any(num_type in pred_dtype.lower() for num_type in numeric_dtypes)
                    
                    if train_is_numeric != pred_is_numeric:
                        issue_msg = f"Column '{col}' has type mismatch: training={train_dtype}, prediction={pred_dtype}"
                        dtype_issues.append(issue_msg)
            
            if dtype_issues:
                warning_msg = f"Found {len(dtype_issues)} data type issues:"
                self.logger.warning(warning_msg)
                for issue in dtype_issues:
                    self.logger.warning(f"  {issue}")
                validation_results['warnings'].append(warning_msg)
                validation_results['dtype_issues'] = dtype_issues
        
        # Log validation results
        if validation_results['passed']:
            self.logger.info("Data validation passed with %d warnings", len(validation_results['warnings']))
        else:
            self.logger.error("Data validation failed with %d errors and %d warnings", 
                             len(validation_results['errors']), len(validation_results['warnings']))
        
        # Store validation results in metadata
        self.prediction_metadata['validation'] = validation_results
        
        return validation_results['passed']

    def prepare_prediction_data(self):
        """
        Prepare data for prediction (e.g., handle target column if present).
        
        Returns:
            pd.DataFrame: Prepared data ready for model prediction
        """
        self.logger.info("Preparing data for prediction...")
        
        if not hasattr(self, 'prediction_df') or self.prediction_df is None:
            error_msg = "Prediction data not loaded. Call load_data() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Get target column from metadata
        target_column = self.metadata.get('parameters', {}).get('target_column')
        if not target_column:
            # Try to find target column in preprocessing metadata
            target_column = self.metadata.get('preprocessing', {}).get('target_column')
        
        # Create a copy of prediction data
        prediction_data = self.prediction_df.copy()
        
        # Extract target column if present
        if target_column and target_column in prediction_data.columns:
            self.logger.info(f"Target column '{target_column}' found in prediction data")
            self.y_true = prediction_data[target_column]
            X_pred = prediction_data.drop(columns=[target_column])
            self.prediction_metadata['preprocessing']['target_present'] = True
            
            # For classification, store original class values for later comparison
            if self.problem_type == 'classification':
                self.original_y_true = self.y_true.copy()
                
                # If label encoder is stored in metadata and we're doing classification
                class_encoding = self.metadata.get('preprocessing', {}).get('class_encoding', {})
                if class_encoding:
                    self.logger.info("Using class encoding from metadata")
                    self.class_names = list(class_encoding.keys())
                    self.prediction_metadata['preprocessing']['class_names'] = self.class_names
                    self.prediction_metadata['preprocessing']['class_encoding'] = class_encoding
                    
                    # We don't encode here - the model pipeline will handle it
                    # Just keep the original for later decoding
        else:
            self.logger.info("No target column found in prediction data")
            X_pred = prediction_data
            self.y_true = None
            self.prediction_metadata['preprocessing']['target_present'] = False
        
        self.X_pred = X_pred
        self.prediction_metadata['preprocessing']['prediction_data_shape'] = X_pred.shape
        
        return X_pred

    def make_predictions(self):
        """
        Make predictions using the loaded model.
        
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        self.logger.info("Making predictions...")
        
        if self.model is None:
            error_msg = "Model not loaded. Call load_model() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not hasattr(self, 'X_pred') or self.X_pred is None:
            error_msg = "Prediction data not prepared. Call prepare_prediction_data() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            start_time = datetime.now()
            
            # Make predictions
            self.predictions = self.model.predict(self.X_pred)
            
            # For classification, also get probability estimates if available
            if self.problem_type == 'classification' and hasattr(self.model, 'predict_proba'):
                try:
                    self.prediction_probs = self.model.predict_proba(self.X_pred)
                    self.logger.info("Probability estimates generated")
                    self.prediction_metadata['results']['probability_estimates_available'] = True
                except Exception as e:
                    self.logger.warning(f"Could not generate probability estimates: {str(e)}")
                    self.prediction_probs = None
                    self.prediction_metadata['results']['probability_estimates_available'] = False
            
            prediction_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Predictions generated in {prediction_time:.2f} seconds")
            
            # Create result DataFrame
            result_df = pd.DataFrame(self.X_pred.copy())
            
            # For classification, decode predictions if needed
            if self.problem_type == 'classification':
                # Get class encoding from metadata
                class_encoding = self.metadata.get('preprocessing', {}).get('class_encoding', {})
                
                if class_encoding:
                    # Create reverse mapping: encoded value -> original class
                    label_decoder = {str(v): k for k, v in class_encoding.items()}
                    
                    # Try to decode predictions
                    try:
                        decoded_predictions = np.array([label_decoder.get(str(int(p)), str(p)) for p in self.predictions])
                        self.logger.info("Decoded predictions to original classes")
                        
                        # Store both encoded and decoded predictions
                        result_df['prediction_encoded'] = self.predictions
                        result_df['prediction'] = decoded_predictions
                    except Exception as e:
                        self.logger.warning(f"Could not decode predictions: {str(e)}")
                        result_df['prediction'] = self.predictions
                else:
                    # No class encoding found in metadata
                    self.logger.warning("No class encoding found in metadata, using raw predictions")
                    result_df['prediction'] = self.predictions
                
                # For classification with probabilities, add top probability
                if hasattr(self, 'prediction_probs') and self.prediction_probs is not None:
                    result_df['prediction_probability'] = np.max(self.prediction_probs, axis=1)
            else:
                # For regression
                result_df['prediction'] = self.predictions
            
            # Add true values if available
            if self.y_true is not None:
                if self.problem_type == 'classification' and hasattr(self, 'original_y_true'):
                    result_df['actual'] = self.original_y_true
                else:
                    result_df['actual'] = self.y_true
                
                # For regression, add residual
                if self.problem_type == 'regression':
                    result_df['residual'] = self.y_true - self.predictions
            
            # Update metadata
            self.prediction_metadata['results']['prediction_time_seconds'] = prediction_time
            self.prediction_metadata['results']['num_predictions'] = len(self.predictions)
            
            # Set result_df BEFORE calculating metrics
            self.result_df = result_df
            
            # Calculate appropriate metrics
            if self.y_true is not None:
                self._calculate_metrics(result_df)
            
            return result_df
            
        except Exception as e:
            error_msg = f"Error making predictions: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    def _calculate_metrics(self, result_df):
        """
        Calculate performance metrics based on problem type.
        
        Args:
            result_df: DataFrame with predictions and actual values
        """
        if self.problem_type == 'regression':
            self._calculate_regression_metrics()
        elif self.problem_type == 'classification':
            self._calculate_classification_metrics()
        else:
            self.logger.warning(f"Metrics calculation not implemented for problem type: {self.problem_type}")

    def _calculate_regression_metrics(self):
        """Calculate regression metrics."""
        self.logger.info("Calculating regression metrics...")
        
        # Basic prediction stats
        self.prediction_metadata['results']['predictions_stats'] = {
            'min': float(np.min(self.predictions)),
            'max': float(np.max(self.predictions)),
            'mean': float(np.mean(self.predictions)),
            'std': float(np.std(self.predictions))
        }
        
        # Calculate performance metrics
        mse = mean_squared_error(self.y_true, self.predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_true, self.predictions)
        r2 = r2_score(self.y_true, self.predictions)
        
        # Add MAPE if no zeros in actual values
        try:
            mape = mean_absolute_percentage_error(self.y_true, self.predictions)
            has_mape = True
        except:
            self.logger.warning("Could not calculate MAPE (possible division by zero)")
            mape = None
            has_mape = False
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
        
        if has_mape:
            metrics['mape'] = float(mape)
        
        self.prediction_metadata['results']['performance_metrics'] = metrics
        
        self.logger.info(f"Regression metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        if has_mape:
            self.logger.info(f"MAPE={mape:.4f}")

    def _calculate_classification_metrics(self):
        """Calculate classification metrics."""
        self.logger.info("Calculating classification metrics...")
        
        # For classification with string labels, we need special handling
        try:
            # Get predictions - consistently use string format for both
            if 'prediction' in self.result_df.columns:
                predictions_for_metrics = self.result_df['prediction'].astype(str).values
            else:
                predictions_for_metrics = [str(p) for p in self.predictions]
            
            # Get true values - also as strings for consistency
            if hasattr(self, 'original_y_true'):
                true_values_for_metrics = self.original_y_true.astype(str).values
            elif hasattr(self, 'y_true'):
                true_values_for_metrics = [str(y) for y in self.y_true]
            else:
                self.logger.warning("No true values available for metrics calculation")
                return
            
            # Calculate confusion matrix with string labels
            cm = confusion_matrix(true_values_for_metrics, predictions_for_metrics)
            cm_list = cm.tolist()
            
            # Calculate accuracy
            accuracy = accuracy_score(true_values_for_metrics, predictions_for_metrics)
            
            # Store basic metrics
            metrics = {
                'accuracy': float(accuracy),
                'confusion_matrix': cm_list
            }
            
            # Add classification report
            try:
                report = classification_report(true_values_for_metrics, predictions_for_metrics, output_dict=True)
                metrics['classification_report'] = report
            except Exception as e:
                self.logger.warning(f"Could not generate classification report: {str(e)}")
            
            # For binary classification, calculate more metrics
            unique_classes = np.unique(true_values_for_metrics)
            is_binary = len(unique_classes) == 2
            
            if is_binary:
                try:
                    precision = precision_score(true_values_for_metrics, predictions_for_metrics, average='binary', zero_division=0)
                    recall = recall_score(true_values_for_metrics, predictions_for_metrics, average='binary', zero_division=0)
                    f1 = f1_score(true_values_for_metrics, predictions_for_metrics, average='binary', zero_division=0)
                    
                    metrics.update({
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1)
                    })
                    
                    self.logger.info(f"Binary classification metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                except Exception as e:
                    self.logger.warning(f"Could not calculate some binary classification metrics: {str(e)}")
            else:
                # Multiclass metrics
                try:
                    precision = precision_score(true_values_for_metrics, predictions_for_metrics, average='macro', zero_division=0)
                    recall = recall_score(true_values_for_metrics, predictions_for_metrics, average='macro', zero_division=0)
                    f1 = f1_score(true_values_for_metrics, predictions_for_metrics, average='macro', zero_division=0)
                    
                    metrics.update({
                        'precision_macro': float(precision),
                        'recall_macro': float(recall),
                        'f1_macro': float(f1)
                    })
                    
                    self.logger.info(f"Multiclass metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                except Exception as e:
                    self.logger.warning(f"Could not calculate multiclass metrics: {str(e)}")
            
            # Store class distribution
            try:
                true_class_counts = pd.Series(true_values_for_metrics).value_counts().to_dict()
                pred_class_counts = pd.Series(predictions_for_metrics).value_counts().to_dict()
                
                metrics['true_class_distribution'] = {str(k): int(v) for k, v in true_class_counts.items()}
                metrics['predicted_class_distribution'] = {str(k): int(v) for k, v in pred_class_counts.items()}
            except Exception as e:
                self.logger.warning(f"Could not calculate class distribution: {str(e)}")
            
            # Store in metadata
            self.prediction_metadata['results']['performance_metrics'] = metrics
        
        except Exception as e:
            self.logger.error(f"Error calculating classification metrics: {str(e)}", exc_info=True)
            self.prediction_metadata['results']['metrics_error'] = str(e)

    def _store_prediction_samples(self, max_samples=100):
        """
        Store prediction samples in metadata.
        
        Args:
            max_samples: Maximum number of predictions to store
        """
        if not hasattr(self, 'result_df') or self.result_df is None:
            self.logger.warning("No prediction results available")
            return
        
        # Create a sample of predictions for metadata
        sample_size = min(max_samples, len(self.result_df))
        
        # For classification, try to include samples from all classes
        if self.problem_type == 'classification' and 'actual' in self.result_df.columns:
            # Create a balanced sample with both correct and incorrect predictions
            correct_mask = self.result_df['prediction'] == self.result_df['actual']
            correct_samples = self.result_df[correct_mask]
            incorrect_samples = self.result_df[~correct_mask]
            
            # Include more incorrect samples as they're more interesting
            incorrect_sample_size = min(max_samples // 2, len(incorrect_samples))
            correct_sample_size = min(max_samples - incorrect_sample_size, len(correct_samples))
            
            incorrect_subset = incorrect_samples.sample(incorrect_sample_size) if len(incorrect_samples) > 0 else pd.DataFrame()
            correct_subset = correct_samples.sample(correct_sample_size) if len(correct_samples) > 0 else pd.DataFrame()
            
            samples_df = pd.concat([incorrect_subset, correct_subset])
            
            # Also store some misclassified examples specifically
            if len(incorrect_samples) > 0:
                misclassified_sample = incorrect_samples.head(min(10, len(incorrect_samples)))
                misclassified_list = []
                
                for _, row in misclassified_sample.iterrows():
                    sample = {
                        'actual': str(row['actual']),
                        'predicted': str(row['prediction'])
                    }
                    
                    if 'prediction_probability' in row:
                        sample['probability'] = float(row['prediction_probability'])
                    
                    misclassified_list.append(sample)
                
                self.prediction_metadata['results']['misclassified_examples'] = misclassified_list
        else:
            # For regression or if no actual values, just take a random sample
            samples_df = self.result_df.sample(sample_size)
        
        # Convert to list of dictionaries for storage in metadata
        samples_list = []
        include_cols = ['actual', 'prediction']
        
        # Additional columns for classification
        if self.problem_type == 'classification':
            if 'prediction_probability' in self.result_df.columns:
                include_cols.append('prediction_probability')
        
        # Additional columns for regression
        if self.problem_type == 'regression':
            if 'residual' in self.result_df.columns:
                include_cols.append('residual')
        
        # Filter to only use columns that exist in result_df
        cols_to_use = [col for col in include_cols if col in self.result_df.columns]
        
        for _, row in samples_df[cols_to_use].iterrows():
            sample_dict = {}
            for col in cols_to_use:
                # Convert values to appropriate types for JSON
                if pd.api.types.is_numeric_dtype(row[col]):
                    sample_dict[col] = float(row[col])
                else:
                    sample_dict[col] = str(row[col])
            
            samples_list.append(sample_dict)
        
        self.prediction_metadata['results']['prediction_samples'] = samples_list
        self.logger.info(f"Stored {len(samples_list)} prediction samples in metadata")

    def save_results(self):
        """
        Save prediction results and metadata.
        
        Returns:
            str: metadata_path
        """
        self.logger.info("Saving prediction results and metadata...")
        
        if not hasattr(self, 'result_df') or self.result_df is None:
            error_msg = "No prediction results available. Call make_predictions() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Store prediction samples in metadata
        self._store_prediction_samples()
        
        # Make metadata serializable
        def make_serializable(obj):
            if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                if len(obj) > 100:  # Limit large arrays
                    return obj[:100].tolist() + ["... truncated"]
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                if len(obj) > 5:  # Sample for large DataFrames
                    return f"DataFrame: {obj.shape}, sample: {obj.head(5).to_dict()}"
                return obj.to_dict()
            elif isinstance(obj, pd.Series):
                if len(obj) > 5:  # Sample for large Series
                    return f"Series: {obj.shape}, sample: {obj.head(5).to_dict()}"
                return obj.to_dict()
            elif hasattr(obj, '__dict__'):
                return str(type(obj))
            else:
                return str(obj)

        def process_dict(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    process_dict(v)
                elif isinstance(v, (list, tuple)):
                    d[k] = [make_serializable(item) for item in v]
                else:
                    d[k] = make_serializable(v)
            return d
        
        # Process metadata to ensure it's serializable
        processed_metadata = process_dict(self.prediction_metadata.copy())
        
        # Use simpler filename without timestamp
        metadata_path = os.path.join(self.output_dir, "prediction_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(processed_metadata, f, indent=2)
        
        self.logger.info(f"Prediction metadata saved to {metadata_path}")
        
        return metadata_path

    def run_prediction_pipeline(self, data_path=None, df=None):
        """
        Run the complete prediction pipeline.
        
        Args:
            data_path (str): Path to data file
            df (pd.DataFrame): DataFrame to use for prediction
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        self.logger.info("Starting prediction pipeline...")
        pipeline_start = datetime.now()
        
        try:
            # Load model
            self.logger.info("\nLoading model and metadata...")
            self.load_model()
            
            # Load prediction data
            self.logger.info("\nLoading prediction data...")
            self.load_data(data_path, df)
            self.logger.info(f"Loaded prediction data: {self.prediction_df.shape}")
            
            # Validate data
            self.logger.info("\nValidating prediction data...")
            try:
                validation_result = self.validate_data()
                if not validation_result:
                    self.logger.warning("Validation found issues but will attempt prediction anyway")
            except Exception as e:
                self.logger.warning(f"Validation error: {str(e)}. Will attempt prediction anyway.")
            
            # Prepare data
            self.logger.info("\nPreparing prediction data...")
            self.prepare_prediction_data()
            
            # Make predictions
            self.logger.info("\nMaking predictions...")
            result_df = self.make_predictions()
            
            # Save results
            self.logger.info("\nSaving results...")
            metadata_path = self.save_results()
            
            # Calculate total runtime
            pipeline_runtime = (datetime.now() - pipeline_start).total_seconds()
            self.logger.info(f"\nPrediction pipeline completed in {pipeline_runtime:.2f} seconds!")
            
            # Final metadata updates
            self.prediction_metadata['runtime_seconds'] = pipeline_runtime
            self.prediction_metadata['status'] = 'completed'
            
            # Print summary statistics based on problem type
            if self.problem_type == 'regression':
                metrics = self.prediction_metadata['results'].get('performance_metrics', {})
                if metrics:
                    self.logger.info("\nRegression metrics:")
                    self.logger.info(f"  RMSE: {metrics.get('rmse', 'N/A')}")
                    self.logger.info(f"  MAE: {metrics.get('mae', 'N/A')}")
                    self.logger.info(f"  R²: {metrics.get('r2', 'N/A')}")
                    if 'mape' in metrics:
                        self.logger.info(f"  MAPE: {metrics.get('mape', 'N/A')}")
            
            elif self.problem_type == 'classification':
                metrics = self.prediction_metadata['results'].get('performance_metrics', {})
                if metrics:
                    self.logger.info("\nClassification metrics:")
                    self.logger.info(f"  Accuracy: {metrics.get('accuracy', 'N/A')}")
                    
                    # For binary classification
                    if 'precision' in metrics:
                        self.logger.info(f"  Precision: {metrics.get('precision', 'N/A')}")
                        self.logger.info(f"  Recall: {metrics.get('recall', 'N/A')}")
                        self.logger.info(f"  F1: {metrics.get('f1', 'N/A')}")
                        if 'auc' in metrics:
                            self.logger.info(f"  AUC: {metrics.get('auc', 'N/A')}")
                    
                    # For multiclass
                    elif 'precision_macro' in metrics:
                        self.logger.info(f"  Precision (macro): {metrics.get('precision_macro', 'N/A')}")
                        self.logger.info(f"  Recall (macro): {metrics.get('recall_macro', 'N/A')}")
                        self.logger.info(f"  F1 (macro): {metrics.get('f1_macro', 'N/A')}")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error in prediction pipeline: {str(e)}", exc_info=True)
            
            # Update metadata with error information
            self.prediction_metadata['status'] = 'failed'
            self.prediction_metadata['error'] = str(e)
            
            # Try to save metadata even if pipeline failed
            try:
                metadata_path = os.path.join(self.output_dir, f"prediction_metadata_failed_{self.timestamp}.json")
                with open(metadata_path, 'w') as f:
                    json.dump(self.prediction_metadata, f, indent=2)
                self.logger.info(f"Error metadata saved to {metadata_path}")
            except:
                self.logger.error("Failed to save error metadata")
            
            raise

def main():
    """Main function to run the prediction script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ML Model Prediction Pipeline')
    parser.add_argument('--model_dir', type=str, help='Directory containing the trained model and metadata')
    parser.add_argument('--model_id', type=str, help='Unique identifier for the model')
    parser.add_argument('--data_path', type=str, help='Path to prediction data file (auto-detected if model_id is provided)')
    parser.add_argument('--output_dir', type=str, help='Base directory to save prediction results and logs')
    
    args = parser.parse_args()
    
    # Ensure either model_dir or model_id is provided
    if not args.model_dir and not args.model_id:
        print("Error: Either --model_dir or --model_id must be provided")
        parser.print_help()
        return 1
    
    try:
        # Create predictor
        predictor = ModelPredictor(model_dir=args.model_dir, output_dir=args.output_dir, model_id=args.model_id)
        
        # Auto-detect test data if no data_path is provided
        data_path = args.data_path
        if not data_path and predictor.model_id:
            data_path = predictor.auto_detect_test_data()
            
        if not data_path:
            print("Error: Could not determine data path. Provide --data_path or ensure test data exists.")
            return 1
        
        # Run prediction pipeline
        result_df = predictor.run_prediction_pipeline(data_path=data_path)
        return 0
    except Exception as e:
        print(f"\nError running prediction pipeline: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main())