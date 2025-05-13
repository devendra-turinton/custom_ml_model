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
    classification_report
)

# Suppress common warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger(__name__).setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


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
    logger.debug(f"Getting next version directory for model '{model_id}' in '{base_output_dir}'")
    
    # Create model directory if it doesn't exist
    model_dir = os.path.join(base_output_dir, model_id)
    if not os.path.exists(model_dir):
        logger.debug(f"Model directory does not exist, creating new directory: {model_dir}")
        os.makedirs(model_dir, exist_ok=True)
        logger.debug(f"Created new model directory, returning version v1")
        return os.path.join(model_dir, "v1"), 1
    
    # Get existing version directories
    version_dirs = []
    for item in os.listdir(model_dir):
        if os.path.isdir(os.path.join(model_dir, item)) and item.startswith('v'):
            try:
                version_num = int(item[1:])  # Extract number part from 'v1', 'v2', etc.
                version_dirs.append((item, version_num))
                logger.debug(f"Found existing version: {item}")
            except ValueError:
                # Skip directories that don't follow the vN pattern
                logger.debug(f"Skipping directory that doesn't follow version pattern: {item}")
                continue
    
    # Sort by version number
    version_dirs.sort(key=lambda x: x[1])
    
    # If no valid versions exist, start with v1
    if not version_dirs:
        logger.debug("No valid version directories found, starting with v1")
        return os.path.join(model_dir, "v1"), 1
    
    # Get the next version number
    next_version_num = version_dirs[-1][1] + 1
    next_version = f"v{next_version_num}"
    logger.debug(f"Current highest version is v{version_dirs[-1][1]}, next version will be {next_version}")
    
    # Check if we need to delete old versions
    if len(version_dirs) >= max_versions:
        # Keep the newest (max_versions-1) to make room for the new one
        versions_to_delete = version_dirs[:-(max_versions-1)]
        logger.info(f"Reached max versions limit ({max_versions}), will delete {len(versions_to_delete)} old version(s)")
        for version, _ in versions_to_delete:
            dir_to_delete = os.path.join(model_dir, version)
            try:
                import shutil
                logger.debug(f"Deleting old version directory: {dir_to_delete}")
                shutil.rmtree(dir_to_delete)
                logger.info(f"Deleted old version directory: {dir_to_delete}")
            except Exception as e:
                logger.error(f"Error deleting {dir_to_delete}: {str(e)}")
                print(f"Error deleting {dir_to_delete}: {str(e)}")
    
    logger.debug(f"Returning next version directory: {os.path.join(model_dir, next_version)}")
    return os.path.join(model_dir, next_version), next_version_num

class ModelPredictor:
    """Base class for model prediction."""

    def __init__(self, model_dir=None, output_dir=None, model_id=None, base_data_dir=None):
        """
        Initialize predictor with model ID or directory.
        
        Args:
            model_dir (str): Directory containing the trained model and metadata
            output_dir (str): Directory to save prediction results and logs
            model_id (str): Unique identifier for the model
            base_data_dir (str): Base data directory (optional, overrides default)
        """
        # Log initialization start
        print(f"Initializing ModelPredictor...")
        
        # Initialize timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Store model_id
        self.model_id = model_id
        
        # Set up paths based on project structure
        if base_data_dir is None:
            # Use default path based on home directory
            home_dir = os.path.expanduser("~")
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
                    print(f"Auto-detected latest model version: {latest_version}")
        
        # Set model directory
        if not model_dir:
            raise ValueError("Either model_dir or model_id must be provided")
        self.model_dir = model_dir
        
        # Set up versioned output directory
        if output_dir:
            # Use the provided output directory directly
            self.output_dir = output_dir
            # Extract version from the path
            basename = os.path.basename(output_dir)
            self.version = basename if basename.startswith('v') else "v1"
            print(f"Using provided output directory: {self.output_dir} (version: {self.version})")
        elif model_id:
            # Base output directory for testing results
            test_out_base = os.path.join(base_data_dir, "testing", "output")
            
            # Get next version directory
            self.output_dir, version_num = get_next_version_dir(test_out_base, model_id)
            self.version = f"v{version_num}"
            print(f"Created versioned output directory: {self.output_dir} (version: {self.version})")
        else:
            # Default output dir if no model_id or output_dir
            self.output_dir = os.path.join(os.path.dirname(model_dir), f"predictions_{self.timestamp}")
            self.version = "v1"
            print(f"Using default output directory: {self.output_dir}")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logger BEFORE using it
        self.logger = self._setup_logging()
        
        # Log system information
        self._log_system_info()
        
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
        
        self.logger.info(f"ModelPredictor initialized with model_id={model_id}, output_dir={self.output_dir}")

    def _log_system_info(self):
        """Log system information for debugging purposes."""
        try:
            import platform
            import psutil
            
            self.logger.debug("System Information:")
            self.logger.debug(f"  - OS: {platform.system()} {platform.release()}")
            self.logger.debug(f"  - Python: {platform.python_version()}")
            
            # CPU info
            cpu_count = psutil.cpu_count(logical=False)
            cpu_logical_count = psutil.cpu_count(logical=True)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.logger.debug(f"  - CPU: {cpu_count} physical cores, {cpu_logical_count} logical cores, {cpu_percent}% used")
            
            # Memory info
            mem = psutil.virtual_memory()
            self.logger.debug(f"  - Memory: {mem.total / (1024**3):.2f} GB total, {mem.available / (1024**3):.2f} GB available ({mem.percent}% used)")
            
            # Disk info
            disk = psutil.disk_usage('/')
            self.logger.debug(f"  - Disk: {disk.total / (1024**3):.2f} GB total, {disk.free / (1024**3):.2f} GB free ({disk.percent}% used)")
            
        except ImportError:
            self.logger.debug("System information logging skipped (psutil not available)")
        except Exception as e:
            self.logger.debug(f"Error collecting system information: {str(e)}")

    def _setup_logging(self):
        """Set up logging configuration."""
        # Create a job-specific logger name using model_id and version
        logger_name = f"model_predictor.{self.model_id}.{self.version}"
        logger = logging.getLogger(logger_name)
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
        
        # IMPORTANT: Prevent log propagation to parent loggers
        logger.propagate = False
        
        # Log startup message
        print(f"Logging initialized. Log file: {log_file}")
        
        return logger

    def auto_detect_test_data(self):
        """
        Auto-detect test data file path based on model_id.
        
        Returns:
            str: Path to the test data file
        """
        self.logger.info("Attempting to auto-detect test data file...")
        
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
            file_size_mb = os.path.getsize(test_data_path) / (1024 * 1024)
            self.logger.info(f"Auto-detected test data at: {test_data_path} (size: {file_size_mb:.2f} MB)")
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
        
        self.logger.debug(f"Test data not found at primary location, checking alternative: {alt_test_data_path}")
        
        if os.path.exists(alt_test_data_path):
            file_size_mb = os.path.getsize(alt_test_data_path) / (1024 * 1024)
            self.logger.info(f"Found test data in training input: {alt_test_data_path} (size: {file_size_mb:.2f} MB)")
            return alt_test_data_path
        
        # Check for other file extensions
        for ext in ['.xlsx', '.json', '.parquet']:
            alt_path = alt_test_data_path.replace('.csv', ext)
            if os.path.exists(alt_path):
                file_size_mb = os.path.getsize(alt_path) / (1024 * 1024)
                self.logger.info(f"Found alternative test data format: {alt_path} (size: {file_size_mb:.2f} MB)")
                return alt_path
        
        self.logger.warning(f"Could not find test data for model_id {self.model_id}")
        self.logger.debug(f"Tried paths: {test_data_path} and {alt_test_data_path}")
        
        return None

    def load_model(self):
        """
        Load the trained model and metadata.
        
        Returns:
            tuple: (model, metadata)
        """
        self.logger.info(f"Looking for model files in: {self.model_dir}")
        start_time = datetime.now()
        
        # Find the model file (.pkl) and metadata file (.json)
        try:
            model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
            metadata_files = [f for f in os.listdir(self.model_dir) if f.endswith('.json')]
            
            self.logger.debug(f"Found {len(model_files)} model files and {len(metadata_files)} metadata files")
            
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
            
            # Get file sizes
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            metadata_size_kb = os.path.getsize(metadata_path) / 1024
            
            self.logger.info(f"Loading model from {model_path} (size: {model_size_mb:.2f} MB)")
            self.logger.info(f"Loading metadata from {metadata_path} (size: {metadata_size_kb:.2f} KB)")
        except Exception as e:
            error_msg = f"Error finding model files: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
        
        try:
            # Load model
            model_load_start = datetime.now()
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            model_load_time = (datetime.now() - model_load_start).total_seconds()
            self.logger.debug(f"Model loaded in {model_load_time:.2f} seconds")
            
            # Log model info
            self._log_model_info()
            
            # Load metadata
            metadata_load_start = datetime.now()
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            metadata_load_time = (datetime.now() - metadata_load_start).total_seconds()
            self.logger.debug(f"Metadata loaded in {metadata_load_time:.2f} seconds")
            
            # Determine problem type from metadata
            self.problem_type = self.metadata.get('parameters', {}).get('problem_type')
            if not self.problem_type:
                self.logger.warning("Could not find problem_type in parameters, checking elsewhere in metadata")
                # Try alternative locations
                if 'best_model' in self.metadata and 'problem_type' in self.metadata['best_model']:
                    self.problem_type = self.metadata['best_model']['problem_type']
                    self.logger.debug(f"Found problem_type in best_model: {self.problem_type}")
                else:
                    raise ValueError("Could not determine problem type from metadata")
            
            self.logger.info(f"Problem type: {self.problem_type}")
            
            # Log metadata summary
            self._log_metadata_summary()
            
            total_load_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Model and metadata loaded successfully in {total_load_time:.2f} seconds")
            
            # Update prediction metadata
            self.prediction_metadata['model_info'] = {
                'model_file': model_path,
                'model_size_mb': model_size_mb,
                'metadata_file': metadata_path,
                'problem_type': self.problem_type,
                'best_model': self.metadata.get('best_model', {}).get('name', 'unknown'),
                'loading_time_seconds': total_load_time
            }
            
            return self.model, self.metadata
            
        except Exception as e:
            error_msg = f"Error loading model or metadata: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    def _log_model_info(self):
        """Log information about the loaded model."""
        try:
            self.logger.debug("Model information:")
            
            # Check if it's a sklearn Pipeline
            if hasattr(self.model, 'named_steps'):
                steps = list(self.model.named_steps.keys())
                self.logger.debug(f"  - Type: Pipeline with steps: {steps}")
                
                # Get information about the final estimator
                if 'model' in self.model.named_steps:
                    estimator = self.model.named_steps['model']
                    estimator_type = type(estimator).__name__
                    self.logger.debug(f"  - Final estimator: {estimator_type}")
                    
                    # Log hyperparameters for the estimator
                    self.logger.debug("  - Estimator parameters:")
                    for param, value in estimator.get_params().items():
                        # Only log direct parameters, not nested ones
                        if '__' not in param:
                            self.logger.debug(f"    - {param}: {value}")
                
                # Get information about the preprocessor if available
                if 'preprocessor' in self.model.named_steps:
                    preprocessor = self.model.named_steps['preprocessor']
                    preprocessor_type = type(preprocessor).__name__
                    self.logger.debug(f"  - Preprocessor: {preprocessor_type}")
                    
                    # If it's a column transformer, get more details
                    if hasattr(preprocessor, 'transformers_'):
                        self.logger.debug("  - Transformer pipelines:")
                        for name, _, cols in preprocessor.transformers_:
                            if cols:
                                num_cols = len(cols) if isinstance(cols, list) else 'all'
                                self.logger.debug(f"    - {name}: {num_cols} columns")
            else:
                # Not a pipeline, just a model
                model_type = type(self.model).__name__
                self.logger.debug(f"  - Type: {model_type} (not a Pipeline)")
                
                # Try to log some basic parameters
                if hasattr(self.model, 'get_params'):
                    main_params = {k: v for k, v in self.model.get_params().items() if '__' not in k}
                    self.logger.debug("  - Parameters:")
                    for param, value in main_params.items():
                        self.logger.debug(f"    - {param}: {value}")
                        
        except Exception as e:
            self.logger.debug(f"Error logging model info: {str(e)}")

    def _log_metadata_summary(self):
        """Log a summary of the loaded metadata."""
        try:
            self.logger.debug("Metadata summary:")
            
            # Log training parameters
            if 'parameters' in self.metadata:
                params = self.metadata['parameters']
                self.logger.debug("  - Training parameters:")
                for key, value in params.items():
                    self.logger.debug(f"    - {key}: {value}")
            
            # Log data information
            if 'data' in self.metadata:
                data_info = self.metadata['data']
                shape = data_info.get('shape', ['unknown', 'unknown'])
                self.logger.debug(f"  - Training data: {shape[0]} rows, {shape[1]} columns")
                
                if 'missing_values' in data_info:
                    missing = sum(data_info['missing_values'].values())
                    self.logger.debug(f"    - Missing values in training: {missing}")
            
            # Log best model information
            if 'best_model' in self.metadata:
                best_model = self.metadata['best_model']
                best_name = best_model.get('name', 'unknown')
                metrics = []
                for k, v in best_model.items():
                    if 'metric' in k and 'value' in k:
                        metrics.append(f"{k}={v}")
                
                metric_str = ", ".join(metrics) if metrics else "no metrics found"
                self.logger.debug(f"  - Best model: {best_name} ({metric_str})")
                
            # Log preprocessing information
            if 'preprocessing' in self.metadata:
                preproc = self.metadata['preprocessing']
                feature_types = preproc.get('feature_types', {})
                num_features = len(feature_types.get('numeric_features', []))
                cat_features = len(feature_types.get('categorical_features', []))
                self.logger.debug(f"  - Features: {num_features} numeric, {cat_features} categorical")
            
        except Exception as e:
            self.logger.debug(f"Error logging metadata summary: {str(e)}")

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
        start_time = datetime.now()
        
        if df is not None:
            memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            self.logger.info(f"Using provided DataFrame: {df.shape[0]:,} rows, {df.shape[1]:,} columns, {memory_usage_mb:.2f} MB")
            self.logger.debug(f"DataFrame dtypes: {df.dtypes.value_counts().to_dict()}")
            self.logger.debug(f"DataFrame columns: {list(df.columns)}")
            
            # Log some basic stats
            self._log_dataframe_stats(df)
            
            self.prediction_metadata['prediction_data']['source'] = 'provided_dataframe'
            self.prediction_metadata['prediction_data']['shape'] = df.shape
            self.prediction_metadata['prediction_data']['memory_usage_mb'] = memory_usage_mb
            self.prediction_df = df
            return df
        
        if data_path is None:
            error_msg = "Either data_path or df must be provided"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        file_ext = os.path.splitext(data_path)[1].lower()
        file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
        
        self.logger.info(f"Loading data from file: {data_path} (size: {file_size_mb:.2f} MB)")
        self.prediction_metadata['prediction_data']['source'] = data_path
        self.prediction_metadata['prediction_data']['file_extension'] = file_ext
        self.prediction_metadata['prediction_data']['file_size_mb'] = file_size_mb
        
        try:
            file_load_start = datetime.now()
            
            if file_ext == '.csv':
                self.logger.info(f"Loading CSV file: {data_path}")
                # First peek at the CSV to get column count and delimiter
                with open(data_path, 'r') as f:
                    first_lines = [next(f) for _ in range(5) if f]
                
                delimiter = ',' if ',' in first_lines[0] else ('\t' if '\t' in first_lines[0] else None)
                approx_columns = len(first_lines[0].split(delimiter)) if delimiter else "unknown"
                self.logger.debug(f"CSV preview: approx. {approx_columns} columns with delimiter '{delimiter}'")
                
                df = pd.read_csv(data_path)
                self.logger.debug(f"CSV file loaded: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
            elif file_ext in ['.xls', '.xlsx']:
                self.logger.info(f"Loading Excel file: {data_path}")
                
                # First check sheet names
                import openpyxl
                wb = openpyxl.load_workbook(data_path, read_only=True)
                sheet_names = wb.sheetnames
                self.logger.debug(f"Excel file contains sheets: {sheet_names}")
                wb.close()
                
                # Load the first sheet by default
                df = pd.read_excel(data_path)
                self.logger.debug(f"Loaded first sheet from Excel: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
            elif file_ext == '.json':
                self.logger.info(f"Loading JSON file: {data_path}")
                
                # Peek at JSON structure
                with open(data_path, 'r') as f:
                    first_char = f.read(1)
                expected_structure = "array" if first_char == '[' else "object" if first_char == '{' else "unknown"
                self.logger.debug(f"JSON appears to be {expected_structure} structure")
                
                df = pd.read_json(data_path)
                self.logger.debug(f"JSON file loaded: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
            elif file_ext == '.parquet':
                self.logger.info(f"Loading Parquet file: {data_path}")
                df = pd.read_parquet(data_path)
                self.logger.debug(f"Parquet file loaded: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
            else:
                error_msg = f"Unsupported file extension: {file_ext}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            file_load_time = (datetime.now() - file_load_start).total_seconds()
            memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            
            self.logger.info(f"Data loaded in {file_load_time:.2f} seconds - {df.shape[0]:,} rows, {df.shape[1]:,} columns, {memory_usage_mb:.2f} MB")
            
            # Log dataframe statistics
            self._log_dataframe_stats(df)
            
            # Update metadata
            self.prediction_metadata['prediction_data']['shape'] = df.shape
            self.prediction_metadata['prediction_data']['loading_time_seconds'] = file_load_time
            self.prediction_metadata['prediction_data']['memory_usage_mb'] = memory_usage_mb
            self.prediction_metadata['prediction_data']['columns'] = list(df.columns)
            self.prediction_metadata['prediction_data']['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            
            # Track missing values
            missing_counts = df.isnull().sum()
            missing_cols = missing_counts[missing_counts > 0]
            if not missing_cols.empty:
                missing_percent = (missing_cols / len(df) * 100).round(2)
                missing_data = {
                    'total_missing_values': int(missing_counts.sum()),
                    'columns_with_missing': len(missing_cols),
                    'missing_by_column': {str(col): int(count) for col, count in missing_cols.items()},
                    'missing_percent_by_column': {str(col): float(pct) for col, pct in missing_percent.items()}
                }
                self.prediction_metadata['prediction_data']['missing_values'] = missing_data
                
                self.logger.info(f"Found {missing_data['total_missing_values']:,} missing values in {missing_data['columns_with_missing']:,} columns")
                for col, count in missing_cols.items():
                    pct = missing_percent[col]
                    self.logger.debug(f"  - '{col}': {count:,} missing values ({pct:.2f}%)")
            else:
                self.logger.info("No missing values found in the data")
            
            self.prediction_df = df
            return df
            
        except Exception as e:
            error_msg = f"Error loading prediction data from {data_path}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    def _log_dataframe_stats(self, df):
        """Log statistics about the dataframe."""
        try:
            # Data types summary
            dtype_counts = df.dtypes.value_counts().to_dict()
            self.logger.debug("DataFrame dtypes summary:")
            for dtype, count in dtype_counts.items():
                self.logger.debug(f"  - {dtype}: {count:,} columns")
            
            # Sample of column names
            if len(df.columns) > 20:
                self.logger.debug(f"First 20 columns: {list(df.columns[:20])}")
            else:
                self.logger.debug(f"Columns: {list(df.columns)}")
            
            # Missing value summary
            missing_summary = df.isnull().sum()
            cols_with_missing = missing_summary[missing_summary > 0]
            if not cols_with_missing.empty:
                self.logger.debug(f"Found {len(cols_with_missing):,} columns with missing values")
                self.logger.debug(f"Total missing cells: {missing_summary.sum():,} out of {df.size:,} ({missing_summary.sum()/df.size*100:.2f}%)")
            
            # Numeric column summary (first 5)
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                cols_to_show = numeric_cols[:5]
                stats = df[cols_to_show].describe().loc[['min', 'mean', 'max']]
                self.logger.debug(f"Sample numeric column statistics (first 5):")
                for col in cols_to_show:
                    self.logger.debug(f"  - '{col}': min={stats.loc['min', col]:.4g}, mean={stats.loc['mean', col]:.4g}, max={stats.loc['max', col]:.4g}")
            
            # Categorical column summary (first 5)
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                self.logger.debug(f"Sample categorical column statistics (first 5):")
                for col in cat_cols[:5]:
                    nunique = df[col].nunique()
                    top_val = df[col].value_counts().index[0] if not df[col].isna().all() else "ALL NULL"
                    top_count = df[col].value_counts().iloc[0] if not df[col].isna().all() else 0
                    self.logger.debug(f"  - '{col}': {nunique:,} unique values, most common: '{top_val}' ({top_count:,} occurrences)")
            
        except Exception as e:
            self.logger.debug(f"Error logging DataFrame stats: {str(e)}")

    def validate_data(self):
        """
        Validate the prediction data against training metadata.
        
        Returns:
            bool: True if validation passes
        """
        self.logger.info("Validating prediction data against training metadata...")
        start_time = datetime.now()
        
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
            if target_column:
                self.logger.debug(f"Found target column '{target_column}' in preprocessing metadata")
            else:
                self.logger.warning("Could not find target column in metadata")
        else:
            self.logger.debug(f"Found target column '{target_column}' in parameters metadata")
            
        # Check if target column is present in prediction data
        target_present = target_column and target_column in self.prediction_df.columns
        if not target_present:
            if target_column:
                warning_msg = f"Target column '{target_column}' not found in prediction data. Will only generate predictions."
                self.logger.warning(warning_msg)
                validation_results['warnings'].append(warning_msg)
            else:
                warning_msg = "No target column identified. Will only generate predictions."
                self.logger.warning(warning_msg)
                validation_results['warnings'].append(warning_msg)
        else:
            self.logger.info(f"Target column '{target_column}' found in prediction data")
            
            # If target is present, validate target type
            if self.problem_type == 'regression' and not pd.api.types.is_numeric_dtype(self.prediction_df[target_column]):
                warning_msg = f"Target column '{target_column}' should be numeric for regression, but found {self.prediction_df[target_column].dtype}."
                self.logger.warning(warning_msg)
                validation_results['warnings'].append(warning_msg)
                self.logger.debug(f"First 5 values of non-numeric target: {self.prediction_df[target_column].head(5).tolist()}")
            
            if self.problem_type == 'classification':
                # For classification, check if values are within expected classes
                self.logger.debug("Checking classification target values against expected classes")
                
                class_encoding = self.metadata.get('preprocessing', {}).get('class_encoding', {})
                if class_encoding:
                    valid_classes = list(class_encoding.keys())
                    self.logger.debug(f"Expected classes from training data: {valid_classes}")
                    
                    prediction_classes = self.prediction_df[target_column].unique().tolist()
                    self.logger.debug(f"Classes in prediction data: {prediction_classes}")
                    
                    invalid_classes = [c for c in prediction_classes if str(c) not in map(str, valid_classes)]
                    if invalid_classes:
                        warning_msg = f"Found {len(invalid_classes)} unexpected classes in target column: {invalid_classes[:5]}..."
                        self.logger.warning(warning_msg)
                        validation_results['warnings'].append(warning_msg)
                        validation_results['unexpected_classes'] = invalid_classes
                    else:
                        self.logger.info("All classes in prediction data match expected classes from training")
                else:
                    self.logger.debug("No class encoding found in metadata, skipping class validation")
        
        # Check expected columns against columns in the pipeline
        # In scikit-learn pipelines, feature names might be embedded in the preprocessor
        try:
            self.logger.debug("Checking feature columns against model requirements")
            
            # Extract feature names from model pipeline if available
            if hasattr(self.model, 'feature_names_in_'):
                expected_features = list(self.model.feature_names_in_)
                self.logger.debug(f"Found feature_names_in_ directly in model: {len(expected_features)} features")
            elif hasattr(self.model, 'named_steps') and 'preprocessor' in self.model.named_steps:
                preprocessor = self.model.named_steps['preprocessor']
                if hasattr(preprocessor, 'feature_names_in_'):
                    expected_features = list(preprocessor.feature_names_in_)
                    self.logger.debug(f"Found feature_names_in_ in preprocessor: {len(expected_features)} features")
                else:
                    # Fallback to metadata
                    self.logger.debug("No feature_names_in_ found in preprocessor, using feature types from metadata")
                    expected_features = self.metadata.get('preprocessing', {}).get('feature_types', {}).get('numeric_features', []) + \
                                        self.metadata.get('preprocessing', {}).get('feature_types', {}).get('categorical_features', [])
                    self.logger.debug(f"Features from metadata: {len(expected_features)} features")
            else:
                # Fallback to metadata
                self.logger.debug("Model doesn't have named_steps or preprocessor, using feature types from metadata")
                expected_features = self.metadata.get('preprocessing', {}).get('feature_types', {}).get('numeric_features', []) + \
                                    self.metadata.get('preprocessing', {}).get('feature_types', {}).get('categorical_features', [])
                self.logger.debug(f"Features from metadata: {len(expected_features)} features")
            
            # Filter out target column
            if target_column in expected_features:
                self.logger.debug(f"Removing target column '{target_column}' from expected features")
                expected_features.remove(target_column)
            
            # Check for missing features
            prediction_features = [col for col in self.prediction_df.columns if col != target_column]
            missing_features = [f for f in expected_features if f not in prediction_features]
            
            if missing_features:
                error_msg = f"Missing {len(missing_features)} required features: {missing_features[:10]}"
                if len(missing_features) > 10:
                    error_msg += f"... and {len(missing_features) - 10} more"
                self.logger.error(error_msg)
                validation_results['passed'] = False
                validation_results['errors'].append(error_msg)
                validation_results['missing_features'] = missing_features
            else:
                self.logger.info("All required features are present in prediction data")
            
            # Check for extra features - these are ok but should be noted
            extra_features = [f for f in prediction_features if f not in expected_features]
            if extra_features:
                warning_msg = f"Found {len(extra_features)} extra features not in training data: {extra_features[:10]}"
                if len(extra_features) > 10:
                    warning_msg += f"... and {len(extra_features) - 10} more"
                self.logger.warning(warning_msg)
                validation_results['warnings'].append(warning_msg)
                validation_results['extra_features'] = extra_features
            else:
                self.logger.info("No extra features in prediction data")
            
        except Exception as e:
            warning_msg = f"Could not validate feature names: {str(e)}"
            self.logger.warning(warning_msg, exc_info=True)
            validation_results['warnings'].append(warning_msg)
        
        # Check for missing values
        self.logger.debug("Checking for missing values in prediction data")
        missing_values = self.prediction_df.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0]
        if not columns_with_missing.empty:
            warning_msg = f"Found missing values in {len(columns_with_missing)} columns. Imputation will be applied based on training data."
            self.logger.warning(warning_msg)
            for col, count in columns_with_missing.items():
                pct_missing = count/len(self.prediction_df)*100
                self.logger.warning(f"  {col}: {count:,} missing values ({pct_missing:.2f}%)")
            validation_results['warnings'].append(warning_msg)
            validation_results['missing_values'] = columns_with_missing.to_dict()
        else:
            self.logger.info("No missing values found in prediction data")
        
        # Data type validation
        if 'data' in self.metadata and 'dtypes' in self.metadata['data']:
            self.logger.debug("Checking data types against training data")
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
                        
                        # Add sample values to help with debugging
                        try:
                            sample_values = self.prediction_df[col].head(3).tolist()
                            self.logger.debug(f"  Sample values for '{col}': {sample_values}")
                        except:
                            pass
            
            if dtype_issues:
                warning_msg = f"Found {len(dtype_issues)} data type issues that may affect model performance:"
                self.logger.warning(warning_msg)
                for issue in dtype_issues:
                    self.logger.warning(f"  {issue}")
                validation_results['warnings'].append(warning_msg)
                validation_results['dtype_issues'] = dtype_issues
            else:
                self.logger.info("Data types are compatible with training data")
        
        # Log validation results
        validation_time = (datetime.now() - start_time).total_seconds()
        
        if validation_results['passed']:
            self.logger.info(f"Data validation passed with {len(validation_results['warnings'])} warnings in {validation_time:.2f} seconds")
        else:
            self.logger.error(f"Data validation failed with {len(validation_results['errors'])} errors and {len(validation_results['warnings'])} warnings in {validation_time:.2f} seconds")
        
        # Store validation results in metadata
        validation_results['validation_time_seconds'] = validation_time
        self.prediction_metadata['validation'] = validation_results
        
        return validation_results['passed']

    def prepare_prediction_data(self):
        """
        Prepare data for prediction (e.g., handle target column if present).
        
        Returns:
            pd.DataFrame: Prepared data ready for model prediction
        """
        self.logger.info("Preparing data for prediction...")
        start_time = datetime.now()
        
        if not hasattr(self, 'prediction_df') or self.prediction_df is None:
            error_msg = "Prediction data not loaded. Call load_data() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Get target column from metadata
        target_column = self.metadata.get('parameters', {}).get('target_column')
        if not target_column:
            # Try to find target column in preprocessing metadata
            target_column = self.metadata.get('preprocessing', {}).get('target_column')
            if target_column:
                self.logger.debug(f"Using target column '{target_column}' from preprocessing metadata")
            else:
                self.logger.warning("Target column not found in metadata")
        
        # Create a copy of prediction data
        self.logger.debug("Creating a copy of prediction data")
        prediction_data = self.prediction_df.copy()
        
        # Extract target column if present
        if target_column and target_column in prediction_data.columns:
            self.logger.info(f"Target column '{target_column}' found in prediction data")
            self.y_true = prediction_data[target_column]
            self.logger.debug(f"Extracted target column with {len(self.y_true)} values")
            
            # Log target summary
            if self.problem_type == 'regression' and pd.api.types.is_numeric_dtype(self.y_true):
                target_stats = self.y_true.describe()
                self.logger.debug(f"Target statistics (regression): min={target_stats['min']:.4g}, mean={target_stats['mean']:.4g}, max={target_stats['max']:.4g}")
            elif self.problem_type == 'classification':
                class_counts = self.y_true.value_counts()
                class_pcts = class_counts / len(self.y_true) * 100
                self.logger.debug(f"Target class distribution (classification):")
                for cls, count in class_counts.items():
                    self.logger.debug(f"  - {cls}: {count:,} ({class_pcts[cls]:.2f}%)")
            
            X_pred = prediction_data.drop(columns=[target_column])
            self.logger.debug(f"Remaining feature columns: {X_pred.shape[1]}")
            self.prediction_metadata['preprocessing']['target_present'] = True
            
            # For classification, store original class values for later comparison
            if self.problem_type == 'classification':
                self.logger.debug("Storing original target values for classification metrics")
                self.original_y_true = self.y_true.copy()
                
                # If label encoder is stored in metadata and we're doing classification
                class_encoding = self.metadata.get('preprocessing', {}).get('class_encoding', {})
                if class_encoding:
                    self.logger.info("Found class encoding in metadata")
                    self.class_names = list(class_encoding.keys())
                    self.logger.debug(f"Class names: {self.class_names}")
                    self.prediction_metadata['preprocessing']['class_names'] = self.class_names
                    self.prediction_metadata['preprocessing']['class_encoding'] = class_encoding
                    
                    # We don't encode here - the model pipeline will handle it
                    # Just keep the original for later decoding
                    self.logger.debug("Not applying class encoding now - model pipeline will handle encoding")
        else:
            if target_column:
                self.logger.info(f"Target column '{target_column}' not found in prediction data")
            else:
                self.logger.info("No target column specified")
                
            X_pred = prediction_data
            self.y_true = None
            self.prediction_metadata['preprocessing']['target_present'] = False
        
        self.X_pred = X_pred
        self.logger.info(f"Prepared prediction data with {X_pred.shape[0]:,} rows and {X_pred.shape[1]:,} features")
        
        # Check for suspicious columns that might be identifiers
        self._check_for_id_columns(X_pred)
        
        # Update metadata
        self.prediction_metadata['preprocessing']['prediction_data_shape'] = X_pred.shape
        self.prediction_metadata['preprocessing']['preparation_time_seconds'] = (datetime.now() - start_time).total_seconds()
        
        return X_pred

    def _check_for_id_columns(self, X_pred):
        """Check for columns that might be identifiers rather than features."""
        try:
            # Look for columns that might be unique identifiers
            for col in X_pred.columns:
                if X_pred[col].nunique() == len(X_pred) and len(X_pred) > 10:
                    self.logger.warning(f"Column '{col}' has unique values for each row and might be an identifier, not a feature")
                elif (
                    pd.api.types.is_string_dtype(X_pred[col]) and 
                    X_pred[col].str.contains(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', regex=True).any()
                ):
                    self.logger.warning(f"Column '{col}' appears to contain UUID values and might not be useful for prediction")
                elif col.lower() in ['id', 'uuid', 'guid', 'identifier', 'key', 'row_id', 'index']:
                    self.logger.warning(f"Column '{col}' is named like an identifier and might not be useful for prediction")
        except Exception as e:
            self.logger.debug(f"Error checking for ID columns: {str(e)}")

    def make_predictions(self):
        """
        Make predictions using the loaded model.
        
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        self.logger.info("Making predictions...")
        start_time = datetime.now()
        
        if self.model is None:
            error_msg = "Model not loaded. Call load_model() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not hasattr(self, 'X_pred') or self.X_pred is None:
            error_msg = "Prediction data not prepared. Call prepare_prediction_data() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Log prediction process
            self.logger.debug(f"Starting prediction on {self.X_pred.shape[0]:,} rows")
            predict_start_time = datetime.now()
            
            # Make predictions
            self.predictions = self.model.predict(self.X_pred)
            predict_time = (datetime.now() - predict_start_time).total_seconds()
            self.logger.debug(f"Raw predictions generated in {predict_time:.2f} seconds")
            
            # Log prediction stats
            if self.problem_type == 'regression':
                pred_stats = {
                    'min': float(np.min(self.predictions)),
                    'max': float(np.max(self.predictions)),
                    'mean': float(np.mean(self.predictions)),
                    'std': float(np.std(self.predictions))
                }
                self.logger.debug(f"Prediction statistics: min={pred_stats['min']:.4g}, max={pred_stats['max']:.4g}, mean={pred_stats['mean']:.4g}, std={pred_stats['std']:.4g}")
            elif self.problem_type == 'classification':
                # For classification, log class distribution
                class_counts = pd.Series(self.predictions).value_counts()
                self.logger.debug(f"Predicted class distribution:")
                for cls, count in class_counts.items():
                    pct = count / len(self.predictions) * 100
                    self.logger.debug(f"  - {cls}: {count:,} ({pct:.2f}%)")
            
            # For classification, also get probability estimates if available
            self.prediction_probs = None
            if self.problem_type == 'classification' and hasattr(self.model, 'predict_proba'):
                self.logger.debug("Model supports probability estimation, generating probabilities")
                try:
                    proba_start_time = datetime.now()
                    self.prediction_probs = self.model.predict_proba(self.X_pred)
                    proba_time = (datetime.now() - proba_start_time).total_seconds()
                    
                    # Log some statistics about probabilities
                    max_probs = np.max(self.prediction_probs, axis=1)
                    avg_confidence = np.mean(max_probs) * 100
                    low_conf_count = np.sum(max_probs < 0.7)  # Arbitrary threshold
                    
                    self.logger.info(f"Probability estimates generated in {proba_time:.2f} seconds")
                    self.logger.debug(f"Average prediction confidence: {avg_confidence:.2f}%")
                    self.logger.debug(f"Low confidence predictions (<70%): {low_conf_count:,} ({low_conf_count/len(max_probs)*100:.2f}%)")
                    
                    self.prediction_metadata['results']['probability_estimates_available'] = True
                    self.prediction_metadata['results']['avg_confidence'] = float(avg_confidence)
                    self.prediction_metadata['results']['low_confidence_count'] = int(low_conf_count)
                except Exception as e:
                    self.logger.warning(f"Could not generate probability estimates: {str(e)}")
                    self.prediction_probs = None
                    self.prediction_metadata['results']['probability_estimates_available'] = False
            
            total_prediction_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Predictions generated in {total_prediction_time:.2f} seconds ({self.X_pred.shape[0]/total_prediction_time:.1f} rows/sec)")
            
            # Create result DataFrame
            self.logger.debug("Creating result DataFrame with predictions")
            result_df = pd.DataFrame(self.X_pred.copy())
            
            # For classification, decode predictions if needed
            if self.problem_type == 'classification':
                # Get class encoding from metadata
                class_encoding = self.metadata.get('preprocessing', {}).get('class_encoding', {})
                
                if class_encoding:
                    self.logger.debug("Using class encoding from metadata to decode predictions")
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
                        self.logger.warning(f"Could not decode predictions: {str(e)}", exc_info=True)
                        result_df['prediction'] = self.predictions
                        self.logger.debug("Using raw predictions without decoding")
                else:
                    # No class encoding found in metadata
                    self.logger.warning("No class encoding found in metadata, using raw predictions")
                    result_df['prediction'] = self.predictions
                
                # For classification with probabilities, add top probability
                if hasattr(self, 'prediction_probs') and self.prediction_probs is not None:
                    self.logger.debug("Adding prediction probabilities to result DataFrame")
                    result_df['prediction_probability'] = np.max(self.prediction_probs, axis=1)
            else:
                # For regression
                self.logger.debug("Adding regression predictions to result DataFrame")
                result_df['prediction'] = self.predictions
            
            # Add true values if available
            if self.y_true is not None:
                self.logger.debug("Adding actual values to result DataFrame")
                if self.problem_type == 'classification' and hasattr(self, 'original_y_true'):
                    result_df['actual'] = self.original_y_true
                else:
                    result_df['actual'] = self.y_true
                
                # For regression, add residual
                if self.problem_type == 'regression':
                    self.logger.debug("Calculating residuals for regression predictions")
                    result_df['residual'] = self.y_true - self.predictions
                    
                    # Log residual statistics
                    abs_residuals = np.abs(result_df['residual'])
                    self.logger.debug(f"Residual statistics: mean abs={abs_residuals.mean():.4g}, median abs={abs_residuals.median():.4g}, max abs={abs_residuals.max():.4g}")
            
            # Update metadata
            self.prediction_metadata['results']['prediction_time_seconds'] = total_prediction_time
            self.prediction_metadata['results']['num_predictions'] = len(self.predictions)
            self.prediction_metadata['results']['rows_per_second'] = float(self.X_pred.shape[0]/total_prediction_time)
            
            # Set result_df BEFORE calculating metrics
            self.result_df = result_df
            
            # Calculate appropriate metrics
            if self.y_true is not None:
                self.logger.info("Calculating performance metrics")
                self._calculate_metrics(result_df)
            else:
                self.logger.info("No true values available, skipping metrics calculation")
            
            return result_df
            
        except Exception as e:
            error_msg = f"Error making predictions: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Try to log additional error context
            self.logger.debug(f"Feature columns: {list(self.X_pred.columns)}")
            self.logger.debug(f"Data types: {self.X_pred.dtypes.to_dict()}")
            
            raise RuntimeError(error_msg)

    def _calculate_metrics(self, result_df):
        """
        Calculate performance metrics based on problem type.
        
        Args:
            result_df: DataFrame with predictions and actual values
        """
        start_time = datetime.now()
        
        if self.problem_type == 'regression':
            self._calculate_regression_metrics()
        elif self.problem_type == 'classification':
            self._calculate_classification_metrics()
        else:
            self.logger.warning(f"Metrics calculation not implemented for problem type: {self.problem_type}")
            
        metrics_time = (datetime.now() - start_time).total_seconds()
        self.logger.debug(f"Metrics calculated in {metrics_time:.2f} seconds")
        self.prediction_metadata['results']['metrics_calculation_time'] = metrics_time

    def _calculate_regression_metrics(self):
        """Calculate regression metrics."""
        self.logger.info("Calculating regression metrics...")
        
        # Basic prediction stats
        pred_min = float(np.min(self.predictions))
        pred_max = float(np.max(self.predictions))
        pred_mean = float(np.mean(self.predictions))
        pred_std = float(np.std(self.predictions))
        
        self.prediction_metadata['results']['predictions_stats'] = {
            'min': pred_min,
            'max': pred_max,
            'mean': pred_mean,
            'std': pred_std
        }
        
        self.logger.debug(f"Prediction statistics: min={pred_min:.4g}, max={pred_max:.4g}, mean={pred_mean:.4g}, std={pred_std:.4g}")
        
        # Calculate performance metrics
        self.logger.debug("Calculating regression metrics: MSE, RMSE, MAE, R²")
        mse = mean_squared_error(self.y_true, self.predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_true, self.predictions)
        r2 = r2_score(self.y_true, self.predictions)
        
        # Add MAPE if no zeros in actual values
        try:
            true_min = np.min(np.abs(self.y_true))
            if true_min > 0:
                self.logger.debug("Calculating MAPE (no zero values in target)")
                mape = mean_absolute_percentage_error(self.y_true, self.predictions)
                has_mape = True
            else:
                self.logger.debug(f"Cannot calculate MAPE - minimum absolute target value is {true_min}, which is too close to zero")
                mape = None
                has_mape = False
        except Exception as e:
            self.logger.warning(f"Could not calculate MAPE: {str(e)}")
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
        
        # Add additional metrics
        # Calculate residuals
        residuals = self.y_true - self.predictions
        abs_residuals = np.abs(residuals)
        
        # Add percentile information
        try:
            metrics['residuals_percentiles'] = {
                'p25': float(np.percentile(residuals, 25)),
                'p50': float(np.percentile(residuals, 50)),  # median
                'p75': float(np.percentile(residuals, 75)),
                'p90': float(np.percentile(residuals, 90)),
                'p95': float(np.percentile(residuals, 95)),
                'p99': float(np.percentile(residuals, 99))
            }
            
            metrics['abs_residuals_percentiles'] = {
                'p25': float(np.percentile(abs_residuals, 25)),
                'p50': float(np.percentile(abs_residuals, 50)),  # median
                'p75': float(np.percentile(abs_residuals, 75)),
                'p90': float(np.percentile(abs_residuals, 90)),
                'p95': float(np.percentile(abs_residuals, 95)),
                'p99': float(np.percentile(abs_residuals, 99))
            }
            
            self.logger.debug(f"Residual percentiles: P50 (median)={metrics['residuals_percentiles']['p50']:.4g}, P90={metrics['residuals_percentiles']['p90']:.4g}")
            self.logger.debug(f"Absolute residual percentiles: P50 (median)={metrics['abs_residuals_percentiles']['p50']:.4g}, P90={metrics['abs_residuals_percentiles']['p90']:.4g}")
        except Exception as e:
            self.logger.debug(f"Could not calculate residual percentiles: {str(e)}")
        
        # Check for extreme residuals (potential outliers)
        large_residuals_threshold = np.percentile(abs_residuals, 95)  # Use 95th percentile as threshold
        large_residuals_count = np.sum(abs_residuals > large_residuals_threshold)
        large_residuals_percent = large_residuals_count / len(residuals) * 100
        
        self.logger.debug(f"Large residuals (>{large_residuals_threshold:.4g}): {large_residuals_count} ({large_residuals_percent:.2f}%)")
        
        # Store all metrics in metadata
        self.prediction_metadata['results']['performance_metrics'] = metrics
        
        # Log main metrics
        self.logger.info(f"Regression metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        if has_mape:
            self.logger.info(f"MAPE={mape:.4f}")

    def _calculate_classification_metrics(self):
        """Calculate classification metrics."""
        self.logger.info("Calculating classification metrics...")
        
        # For classification with string labels, we need special handling
        try:
            # Get predictions - consistently use string format for both
            self.logger.debug("Preparing prediction and true values for metrics calculation")
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
            
            # Quick sanity check on data
            self.logger.debug(f"Prediction values sample: {predictions_for_metrics[:5]}")
            self.logger.debug(f"True values sample: {true_values_for_metrics[:5]}")
            
            # Calculate confusion matrix with string labels
            self.logger.debug("Calculating confusion matrix")
            cm = confusion_matrix(true_values_for_metrics, predictions_for_metrics)
            cm_list = cm.tolist()
            
            # Log confusion matrix for small number of classes
            if len(cm) <= 5:  # Only log full matrix for small number of classes
                unique_labels = sorted(set(predictions_for_metrics) | set(true_values_for_metrics))
                self.logger.debug("Confusion matrix:")
                self.logger.debug(f"Labels: {unique_labels}")
                
                for i, row in enumerate(cm):
                    self.logger.debug(f"  {unique_labels[i]}: {row}")
            else:
                self.logger.debug(f"Confusion matrix shape: {cm.shape} (too large to display)")
            
            # Calculate accuracy
            self.logger.debug("Calculating accuracy")
            accuracy = accuracy_score(true_values_for_metrics, predictions_for_metrics)
            self.logger.debug(f"Accuracy: {accuracy:.4f}")
            
            # Store basic metrics
            metrics = {
                'accuracy': float(accuracy),
                'confusion_matrix': cm_list
            }
            
            # Check for class imbalance
            class_counts = pd.Series(true_values_for_metrics).value_counts()
            min_class_pct = class_counts.min() / len(true_values_for_metrics) * 100
            
            if min_class_pct < 10:  # Arbitrary threshold
                self.logger.warning(f"Class imbalance detected: Minority class represents only {min_class_pct:.2f}% of the data")
                metrics['class_imbalance_warning'] = True
                metrics['min_class_percent'] = float(min_class_pct)
            
            # Add classification report
            try:
                self.logger.debug("Generating classification report")
                report = classification_report(true_values_for_metrics, predictions_for_metrics, output_dict=True)
                metrics['classification_report'] = report
                
                # Log weighted averages
                if 'weighted avg' in report:
                    w_avg = report['weighted avg']
                    self.logger.debug(f"Weighted averages - precision: {w_avg['precision']:.4f}, recall: {w_avg['recall']:.4f}, f1: {w_avg['f1-score']:.4f}")
            except Exception as e:
                self.logger.warning(f"Could not generate classification report: {str(e)}")
            
            # For binary classification, calculate more metrics
            unique_classes = np.unique(true_values_for_metrics)
            is_binary = len(unique_classes) == 2
            
            if is_binary:
                self.logger.debug("Calculating binary classification metrics")
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
                    
                    # Check for special cases
                    if precision == 0 or recall == 0:
                        self.logger.warning("Precision or recall is zero - model may be predicting only one class")
                    
                    # Analyze misclassifications
                    fn_mask = (true_values_for_metrics != predictions_for_metrics) & (true_values_for_metrics == unique_classes[1])
                    fp_mask = (true_values_for_metrics != predictions_for_metrics) & (true_values_for_metrics == unique_classes[0])
                    
                    fn_count = np.sum(fn_mask)
                    fp_count = np.sum(fp_mask)
                    
                    self.logger.debug(f"False negatives: {fn_count}, False positives: {fp_count}")
                    
                    metrics['misclassifications'] = {
                        'false_positives': int(fp_count),
                        'false_negatives': int(fn_count),
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Could not calculate some binary classification metrics: {str(e)}")
            else:
                # Multiclass metrics
                self.logger.debug("Calculating multiclass classification metrics")
                try:
                    precision = precision_score(true_values_for_metrics, predictions_for_metrics, average='macro', zero_division=0)
                    recall = recall_score(true_values_for_metrics, predictions_for_metrics, average='macro', zero_division=0)
                    f1 = f1_score(true_values_for_metrics, predictions_for_metrics, average='macro', zero_division=0)
                    
                    # Also calculate micro-averaged metrics
                    precision_micro = precision_score(true_values_for_metrics, predictions_for_metrics, average='micro', zero_division=0)
                    recall_micro = recall_score(true_values_for_metrics, predictions_for_metrics, average='micro', zero_division=0)
                    f1_micro = f1_score(true_values_for_metrics, predictions_for_metrics, average='micro', zero_division=0)
                    
                    metrics.update({
                        'precision_macro': float(precision),
                        'recall_macro': float(recall),
                        'f1_macro': float(f1),
                        'precision_micro': float(precision_micro),
                        'recall_micro': float(recall_micro),
                        'f1_micro': float(f1_micro)
                    })
                    
                    self.logger.info(f"Multiclass metrics (macro-avg): Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                    self.logger.debug(f"Multiclass metrics (micro-avg): Precision={precision_micro:.4f}, Recall={recall_micro:.4f}, F1={f1_micro:.4f}")
                    
                    # Per-class analysis - find worst-performing classes
                    if 'classification_report' in metrics:
                        class_metrics = {k: v for k, v in metrics['classification_report'].items() 
                                        if k not in ['accuracy', 'macro avg', 'weighted avg']}
                        
                        # Find classes with worst F1 scores
                        f1_by_class = {cls: metrics['f1-score'] for cls, metrics in class_metrics.items()}
                        worst_classes = sorted(f1_by_class.items(), key=lambda x: x[1])[:3]
                        
                        if worst_classes:
                            self.logger.debug("Worst performing classes (F1 score):")
                            for cls, f1 in worst_classes:
                                support = class_metrics[cls]['support']
                                self.logger.debug(f"  - Class '{cls}': F1={f1:.4f}, Support={support}")
                    
                except Exception as e:
                    self.logger.warning(f"Could not calculate multiclass metrics: {str(e)}", exc_info=True)
            
            # Store class distribution
            try:
                true_class_counts = pd.Series(true_values_for_metrics).value_counts().to_dict()
                pred_class_counts = pd.Series(predictions_for_metrics).value_counts().to_dict()
                
                metrics['true_class_distribution'] = {str(k): int(v) for k, v in true_class_counts.items()}
                metrics['predicted_class_distribution'] = {str(k): int(v) for k, v in pred_class_counts.items()}
                
                # Check for classes present in true values but not predicted
                missing_pred_classes = set(true_class_counts.keys()) - set(pred_class_counts.keys())
                if missing_pred_classes:
                    self.logger.warning(f"Some classes in true data were never predicted: {missing_pred_classes}")
                    metrics['missing_predicted_classes'] = list(missing_pred_classes)
            except Exception as e:
                self.logger.warning(f"Could not calculate class distribution: {str(e)}")
            
            # Check for confidence correlation with correctness if probabilities are available
            if hasattr(self, 'prediction_probs') and self.prediction_probs is not None and 'prediction_probability' in self.result_df.columns:
                try:
                    correct_mask = self.result_df['prediction'] == self.result_df['actual']
                    correct_probs = self.result_df.loc[correct_mask, 'prediction_probability']
                    incorrect_probs = self.result_df.loc[~correct_mask, 'prediction_probability']
                    
                    if len(correct_probs) > 0 and len(incorrect_probs) > 0:
                        avg_correct_prob = correct_probs.mean()
                        avg_incorrect_prob = incorrect_probs.mean()
                        
                        self.logger.debug(f"Average probability for correct predictions: {avg_correct_prob:.4f}")
                        self.logger.debug(f"Average probability for incorrect predictions: {avg_incorrect_prob:.4f}")
                        
                        metrics['avg_probability_correct'] = float(avg_correct_prob)
                        metrics['avg_probability_incorrect'] = float(avg_incorrect_prob)
                        
                        if avg_correct_prob <= avg_incorrect_prob:
                            self.logger.warning("Unusual pattern: Probability not correlated with correctness")
                except Exception as e:
                    self.logger.debug(f"Could not analyze probability vs correctness: {str(e)}")
            
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
        self.logger.debug(f"Storing up to {max_samples} prediction samples in metadata")
        
        if not hasattr(self, 'result_df') or self.result_df is None:
            self.logger.warning("No prediction results available")
            return
        
        # Create a sample of predictions for metadata
        sample_size = min(max_samples, len(self.result_df))
        
        # For classification, try to include samples from all classes
        if self.problem_type == 'classification' and 'actual' in self.result_df.columns:
            self.logger.debug("Creating balanced classification sample with correct and incorrect predictions")
            # Create a balanced sample with both correct and incorrect predictions
            correct_mask = self.result_df['prediction'] == self.result_df['actual']
            correct_samples = self.result_df[correct_mask]
            incorrect_samples = self.result_df[~correct_mask]
            
            # Include more incorrect samples as they're more interesting
            incorrect_sample_size = min(max_samples // 2, len(incorrect_samples))
            correct_sample_size = min(max_samples - incorrect_sample_size, len(correct_samples))
            
            self.logger.debug(f"Balanced sample: {correct_sample_size} correct + {incorrect_sample_size} incorrect")
            
            incorrect_subset = incorrect_samples.sample(incorrect_sample_size) if len(incorrect_samples) > 0 else pd.DataFrame()
            correct_subset = correct_samples.sample(correct_sample_size) if len(correct_samples) > 0 else pd.DataFrame()
            
            samples_df = pd.concat([incorrect_subset, correct_subset])
            
            # Also store some misclassified examples specifically
            if len(incorrect_samples) > 0:
                self.logger.debug(f"Storing details for {min(10, len(incorrect_samples))} misclassified examples")
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
                
                # Check for patterns in misclassifications
                if len(incorrect_samples) >= 10:
                    try:
                        confusion = incorrect_samples.groupby(['actual', 'prediction']).size().reset_index(name='count')
                        confusion = confusion.sort_values('count', ascending=False)
                        common_errors = confusion.head(5)
                        
                        self.logger.debug("Most common misclassifications:")
                        for _, row in common_errors.iterrows():
                            self.logger.debug(f"  {row['actual']} → {row['prediction']}: {row['count']} instances")
                    except Exception as e:
                        self.logger.debug(f"Could not analyze misclassification patterns: {str(e)}")
        else:
            # For regression or if no actual values, just take a random sample
            self.logger.debug(f"Using random sampling for {sample_size} prediction samples")
            samples_df = self.result_df.sample(sample_size)
            
            # For regression with actual values, try to sample high residual cases
            if self.problem_type == 'regression' and 'residual' in self.result_df.columns:
                try:
                    self.logger.debug("Including high residual cases in regression samples")
                    # Get some samples with high residuals
                    abs_residuals = self.result_df['residual'].abs()
                    high_residual_threshold = np.percentile(abs_residuals, 90)
                    high_residual_samples = self.result_df[abs_residuals > high_residual_threshold].sample(
                        min(max_samples // 4, 
                            sum(abs_residuals > high_residual_threshold))
                    )
                    
                    # Combine with random samples but ensure we don't exceed max_samples
                    regular_sample_size = min(max_samples - len(high_residual_samples), 
                                             len(self.result_df) - len(high_residual_samples))
                    regular_samples = self.result_df[~self.result_df.index.isin(high_residual_samples.index)].sample(regular_sample_size)
                    
                    samples_df = pd.concat([high_residual_samples, regular_samples])
                    self.logger.debug(f"Combined sample: {len(high_residual_samples)} high residual + {len(regular_samples)} regular")
                except Exception as e:
                    self.logger.debug(f"Could not create stratified regression sample: {str(e)}")
        
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
        
        self.logger.debug(f"Storing columns in samples: {cols_to_use}")
        
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
        start_time = datetime.now()
        
        if not hasattr(self, 'result_df') or self.result_df is None:
            error_msg = "No prediction results available. Call make_predictions() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Store prediction samples in metadata
        self._store_prediction_samples()
        
        
        # Make metadata serializable
        self.logger.debug("Processing metadata to ensure JSON serialization")
        
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
        
        # Add timestamp to metadata
        processed_metadata['save_timestamp'] = datetime.now().isoformat()
        
        # Use simpler filename without timestamp
        metadata_path = os.path.join(self.output_dir, "prediction_metadata.json")
        
        try:
            self.logger.debug(f"Writing metadata to {metadata_path}")
            with open(metadata_path, 'w') as f:
                json.dump(processed_metadata, f, indent=2)
            
            file_size_kb = os.path.getsize(metadata_path) / 1024
            self.logger.info(f"Prediction metadata saved to {metadata_path} ({file_size_kb:.2f} KB)")
        except Exception as e:
            self.logger.error(f"Error saving metadata: {str(e)}", exc_info=True)
            
            # Try with a backup approach
            try:
                self.logger.debug("Attempting to save metadata with fallback method")
                backup_path = os.path.join(self.output_dir, f"prediction_metadata_backup_{self.timestamp}.json")
                with open(backup_path, 'w') as f:
                    # Use a more aggressive approach to filter out problematic fields
                    simple_metadata = {
                        'timestamp': processed_metadata.get('timestamp', ''),
                        'model_id': processed_metadata.get('model_id', ''),
                        'version': processed_metadata.get('version', ''),
                        'prediction_data': {
                            'shape': processed_metadata.get('prediction_data', {}).get('shape', [])
                        },
                        'status': 'failed_to_save_full_metadata'
                    }
                    json.dump(simple_metadata, f)
                self.logger.warning(f"Saved simplified metadata to {backup_path}")
                metadata_path = backup_path
            except Exception as nested_e:
                self.logger.error(f"Failed to save simplified metadata: {str(nested_e)}")
        
        save_time = (datetime.now() - start_time).total_seconds()
        self.logger.debug(f"Results and metadata saved in {save_time:.2f} seconds")
        
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
        self.logger.info(f"Starting prediction pipeline - Model ID: {self.model_id}, Version: {self.version}")
        pipeline_start = datetime.now()
        
        # Log system resources at pipeline start
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            disk_usage = psutil.disk_usage(self.output_dir).percent
            
            self.logger.info(f"System resources at pipeline start: CPU {cpu_percent}%, RAM {memory_info.percent}% (Available: {memory_info.available / (1024**3):.2f} GB), Disk {disk_usage}%")
        except ImportError:
            self.logger.debug("psutil not available for resource monitoring")
        
        try:
            # Create a horizontal line in logs for better readability
            self.logger.info("="*50)
            
            # Load model
            self.logger.info("\n## STAGE 1: Loading model and metadata")
            stage_start = datetime.now()
            self.load_model()
            stage_time = (datetime.now() - stage_start).total_seconds()
            self.logger.info(f"Model loading completed in {stage_time:.2f} seconds")
            
            # Load prediction data
            self.logger.info("\n## STAGE 2: Loading prediction data")
            stage_start = datetime.now()
            self.load_data(data_path, df)
            stage_time = (datetime.now() - stage_start).total_seconds()
            self.logger.info(f"Loaded prediction data: {self.prediction_df.shape[0]:,} rows, {self.prediction_df.shape[1]:,} columns in {stage_time:.2f} seconds")
            
            # Validate data
            self.logger.info("\n## STAGE 3: Validating prediction data")
            stage_start = datetime.now()
            try:
                validation_result = self.validate_data()
                if not validation_result:
                    self.logger.warning("Validation found issues but will attempt prediction anyway")
            except Exception as e:
                self.logger.warning(f"Validation error: {str(e)}. Will attempt prediction anyway.")
            stage_time = (datetime.now() - stage_start).total_seconds()
            self.logger.info(f"Data validation completed in {stage_time:.2f} seconds")
            
            # Prepare data
            self.logger.info("\n## STAGE 4: Preparing prediction data")
            stage_start = datetime.now()
            self.prepare_prediction_data()
            stage_time = (datetime.now() - stage_start).total_seconds()
            self.logger.info(f"Data preparation completed in {stage_time:.2f} seconds")
            
            # Make predictions
            self.logger.info("\n## STAGE 5: Making predictions")
            stage_start = datetime.now()
            result_df = self.make_predictions()
            stage_time = (datetime.now() - stage_start).total_seconds()
            self.logger.info(f"Predictions completed in {stage_time:.2f} seconds")
            
            # Save results
            self.logger.info("\n## STAGE 6: Saving results")
            stage_start = datetime.now()
            metadata_path = self.save_results()
            stage_time = (datetime.now() - stage_start).total_seconds()
            self.logger.info(f"Results saved in {stage_time:.2f} seconds")
            
            # Log system resources at pipeline end
            try:
                import psutil
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                self.logger.info(f"System resources at pipeline end: CPU {cpu_percent}%, RAM {memory_info.percent}% (Available: {memory_info.available / (1024**3):.2f} GB)")
            except ImportError:
                pass
            
            # Calculate total runtime
            pipeline_runtime = (datetime.now() - pipeline_start).total_seconds()
            self.logger.info(f"\nPrediction pipeline completed in {pipeline_runtime:.2f} seconds!")
            self.logger.info("="*50)
            
            # Final metadata updates
            self.prediction_metadata['runtime_seconds'] = pipeline_runtime
            self.prediction_metadata['status'] = 'completed'
            
            # Print summary statistics based on problem type
            if self.problem_type == 'regression':
                metrics = self.prediction_metadata['results'].get('performance_metrics', {})
                if metrics:
                    self.logger.info("\nRegression metrics summary:")
                    self.logger.info(f"  RMSE: {metrics.get('rmse', 'N/A')}")
                    self.logger.info(f"  MAE: {metrics.get('mae', 'N/A')}")
                    self.logger.info(f"  R²: {metrics.get('r2', 'N/A')}")
                    if 'mape' in metrics:
                        self.logger.info(f"  MAPE: {metrics.get('mape', 'N/A')}")
            
            elif self.problem_type == 'classification':
                metrics = self.prediction_metadata['results'].get('performance_metrics', {})
                if metrics:
                    self.logger.info("\nClassification metrics summary:")
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
            
            self.logger.info(f"\nResults saved to: {self.output_dir}")
            
            # IMPORTANT: Clean up logger handlers to prevent memory leaks and log collisions
            if hasattr(self, 'logger') and self.logger:
                self.logger.debug(f"Cleaning up logger handlers for {self.model_id}.{self.version}")
                handlers = list(self.logger.handlers)  # Create a copy of the list
                for handler in handlers:
                    self.logger.removeHandler(handler)
                    handler.close()
                self.logger.debug(f"Logger handlers cleaned up for {self.model_id}.{self.version}")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error in prediction pipeline: {str(e)}", exc_info=True)
            
            # Try to log more debugging information
            self.logger.debug("Debug information at failure point:")
            pipeline_state = {
                'has_model': hasattr(self, 'model') and self.model is not None,
                'has_metadata': hasattr(self, 'metadata') and self.metadata is not None,
                'has_prediction_df': hasattr(self, 'prediction_df') and self.prediction_df is not None
            }
            self.logger.debug(f"Pipeline state: {pipeline_state}")
            
            # Update metadata with error information
            self.prediction_metadata['status'] = 'failed'
            self.prediction_metadata['error'] = str(e)
            self.prediction_metadata['error_timestamp'] = datetime.now().isoformat()
            
            # Try to save metadata even if pipeline failed
            try:
                metadata_path = os.path.join(self.output_dir, f"prediction_metadata_failed_{self.timestamp}.json")
                with open(metadata_path, 'w') as f:
                    json.dump(self.prediction_metadata, f, indent=2)
                self.logger.info(f"Error metadata saved to {metadata_path}")
            except Exception as nested_e:
                self.logger.error(f"Failed to save error metadata: {str(nested_e)}")
            
            # Log final runtime
            pipeline_runtime = (datetime.now() - pipeline_start).total_seconds()
            self.logger.error(f"Prediction pipeline failed after {pipeline_runtime:.2f} seconds")
            
            # IMPORTANT: Clean up logger handlers even on error
            if hasattr(self, 'logger') and self.logger:
                handlers = list(self.logger.handlers)  # Create a copy of the list
                for handler in handlers:
                    self.logger.removeHandler(handler)
                    handler.close()
            
            raise

def main():
    """Main function to run the prediction script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ML Model Prediction Pipeline')
    parser.add_argument('--model_dir', type=str, help='Directory containing the trained model and metadata')
    parser.add_argument('--model_id', type=str, help='Unique identifier for the model')
    parser.add_argument('--data_path', type=str, help='Path to prediction data file (auto-detected if model_id is provided)')
    parser.add_argument('--output_dir', type=str, help='Base directory to save prediction results and logs')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure console logging based on verbosity
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, 
                            format='%(levelname)s - %(message)s')
    
    # Get the root logger and add a handler to log to the console
    root_logger = logging.getLogger()
    
    # Ensure either model_dir or model_id is provided
    if not args.model_dir and not args.model_id:
        print("Error: Either --model_dir or --model_id must be provided")
        parser.print_help()
        return 1
    
    print(f"Starting prediction pipeline with model_id={args.model_id}, model_dir={args.model_dir}")
    
    try:
        # Create predictor
        start_time = datetime.now()
        predictor = ModelPredictor(model_dir=args.model_dir, output_dir=args.output_dir, model_id=args.model_id)
        
        # Auto-detect test data if no data_path is provided
        data_path = args.data_path
        if not data_path and predictor.model_id:
            data_path = predictor.auto_detect_test_data()
            
        if not data_path:
            print("Error: Could not determine data path. Provide --data_path or ensure test data exists.")
            return 1
        
        # Run prediction pipeline
        print(f"Running prediction pipeline with data: {data_path}")
        result_df = predictor.run_prediction_pipeline(data_path=data_path)
        
        # Print final outcome
        runtime = (datetime.now() - start_time).total_seconds()
        print(f"\nPrediction pipeline completed successfully in {runtime:.2f} seconds")
        print(f"Results saved to: {predictor.output_dir}")
        
        return 0
    except Exception as e:
        print(f"\nError running prediction pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())