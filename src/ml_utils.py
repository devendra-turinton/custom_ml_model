import os
import json
import yaml
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any, Union, Tuple
from sklearn.base import BaseEstimator, TransformerMixin

# Initialize logger
logger = logging.getLogger(__name__)

#########################################
# Configuration Loading Utilities
#########################################

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise RuntimeError(f"Failed to load configuration from {config_path}: {str(e)}")

def get_model_config(config: Dict[str, Any], problem_type: str) -> Dict[str, Any]:
    """
    Get model-specific configuration based on problem type.
    
    Args:
        config: Full configuration dictionary
        problem_type: Type of ML problem (regression, classification, etc.)
        
    Returns:
        Dictionary with model-specific configuration
    """
    common_config = config.get('common', {})
    model_config = config.get(problem_type, {})
    
    # Deep merge common and model-specific configs
    # Common configs serve as defaults, model-specific configs override them
    merged_config = common_config.copy()
    
    # Helper function to recursively merge dictionaries
    def merge_dicts(base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge_dicts(base[key], value)
            else:
                base[key] = value
    
    merge_dicts(merged_config, model_config)
    
    return merged_config


#########################################
# Problem Detection
#########################################

def detect_problem_type(
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
    """
    Detect the machine learning problem type using statistical analysis.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        config: Configuration dictionary
        
    Returns:
        Problem type as string: 'regression', 'classification', 'time_series', or 'clustering'
    """
    try:
        # Initial validation
        if target_column is None or target_column == "":
            logger.info("No target column provided. Assuming 'clustering'.")
            return 'clustering'

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

        # Get configuration with defaults
        config = config or {}
        detection_config = config.get('problem_detection', {})
        class_detection = detection_config.get('classification', {}).get('class_detection', {})
        max_classes = class_detection.get('max_classes', 20)  # Default to 20 if not found
        
        # Get target series
        target = df[target_column]
        logger.debug(f"Analyzing target column '{target_column}' with dtype: {target.dtype}")

        # Step 1: Basic type detection
        if pd.api.types.is_categorical_dtype(target) or pd.api.types.is_object_dtype(target):
            unique_count = target.nunique()
            min_samples = detection_config.get('min_samples_per_class', 3)
            value_counts = target.value_counts()
            
            if (value_counts >= min_samples).all():
                logger.info(f"Categorical target detected with {unique_count} classes")
                return 'classification'

        # Step 2: Time Series Detection
        date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        index_is_datetime = isinstance(df.index, pd.DatetimeIndex)

        # Time series checks only if datetime columns exist and target is numeric
        if (len(date_columns) > 0 or index_is_datetime) and pd.api.types.is_numeric_dtype(target):
            min_points = detection_config.get('time_series', {}).get('min_points', 10)
            if len(df) >= min_points:
                date_col = date_columns[0] if date_columns else df.index
                
                # Check temporal order
                if isinstance(date_col, pd.Index):
                    is_sorted = date_col.is_monotonic_increasing or date_col.is_monotonic_decreasing
                else:
                    temp_dates = pd.to_datetime(df[date_col])
                    is_sorted = temp_dates.is_monotonic_increasing or temp_dates.is_monotonic_decreasing

                if is_sorted:
                    logger.info("Time series pattern detected")
                    return 'time_series'

        # Step 3: Numeric Analysis for Classification vs Regression
        if pd.api.types.is_numeric_dtype(target):
            # Basic statistics
            unique_values = target.unique()
            n_unique = len(unique_values)
            n_samples = len(target)
            unique_ratio = n_unique / n_samples

            # Binary classification check
            if n_unique == 2:
                logger.info("Binary classification detected")
                return 'classification'

            # Multi-class vs Continuous analysis
            class_indicators = 0
            regression_indicators = 0

            # Calculate gaps between values
            if n_unique > 1:
                sorted_values = np.sort(unique_values)
                gaps = np.diff(sorted_values)
                gaps_std = float(np.std(gaps))
                gaps_mean = float(np.mean(gaps))
                
                # Regular gaps suggest classification
                if gaps_std / gaps_mean < 0.1:
                    class_indicators += 1
            
            # Unique value ratio analysis
            if unique_ratio < 0.01:  # Few unique values
                class_indicators += 1
            if unique_ratio > 0.3:  # Many unique values
                regression_indicators += 1

            # Value range analysis
            value_range = float(target.max() - target.min())
            if value_range > 100 and unique_ratio > 0.1:
                regression_indicators += 1

            # Make decision based on indicators
            logger.debug(f"Classification indicators: {class_indicators}")
            logger.debug(f"Regression indicators: {regression_indicators}")

            if class_indicators > regression_indicators:
                if n_unique <= max_classes:
                    logger.info("Multi-class classification detected")
                    return 'classification'

            # Check against strict classification criteria
            if n_unique > max_classes or unique_ratio > 0.01:
                logger.info("Too many unique values for classification, treating as regression")
                return 'regression'

            # Default to regression for numeric targets
            logger.info("Regression problem detected based on numeric characteristics")
            return 'regression'

        # If we reach here, default based on data type
        if pd.api.types.is_numeric_dtype(target):
            logger.info("Defaulting to regression for numeric target")
            return 'regression'
        else:
            logger.info("Defaulting to classification for non-numeric target")
            return 'classification'

    except Exception as e:
        logger.error(f"Error in problem type detection: {str(e)}")
        raise ValueError(f"Unable to determine problem type: {str(e)}")

def load_custom_function(function_path: str, function_name: str) -> Optional[Callable]:
    """
    Dynamically load a custom function from specified path.
    
    Args:
        function_path: Path to the Python file containing the function
        function_name: Name of the function to load
        
    Returns:
        Callable: The loaded function, or None if loading fails
    """
    try:
        import importlib.util
        import sys
        
        # Dynamically import the module
        spec = importlib.util.spec_from_file_location("custom_module", function_path)
        custom_module = importlib.util.module_from_spec(spec)
        sys.modules["custom_module"] = custom_module
        spec.loader.exec_module(custom_module)
        
        # Check if the module has the specified function
        if hasattr(custom_module, function_name):
            logger.info(f"Custom function '{function_name}' loaded from {function_path}")
            return getattr(custom_module, function_name)
        else:
            logger.error(f"Custom module at {function_path} does not have function '{function_name}'")
            return None
    except Exception as e:
        logger.error(f"Failed to load custom function: {str(e)}")
        return None

#########################################
# Logging Setup
#########################################

def setup_logging(
        log_dir: str = "logs",
        level: str = "INFO",
        log_file: Optional[str] = None,
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ) -> logging.Logger:
    """
    Set up logging for the ML pipeline.
    
    Args:
        log_dir: Directory for log files
        level: Overall logging level
        log_file: Specific log file name (default: timestamped file)
        console_level: Logging level for console output
        file_level: Logging level for file output
        log_format: Format for log messages
        
    Returns:
        Configured logger instance
    """
     # Create logs directory if it doesn't exist and if provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamped log file if not specified
    if log_file is None and log_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"ml_pipeline_{timestamp}.log")
    
    # Convert string levels to logging levels
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    log_level = level_map.get(level.upper(), logging.INFO)
    console_log_level = level_map.get(console_level.upper(), logging.INFO)
    file_log_level = level_map.get(file_level.upper(), logging.DEBUG)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    logger.info(f"Logging configured. Log file: {log_file}")
    return root_logger

def decode_predictions(y_true, y_pred, label_encoder=None):
    """
    Decode encoded predictions back to original labels.
    
    Args:
        y_true: True values (encoded)
        y_pred: Predicted values (encoded)
        label_encoder: Label encoder used for encoding
        
    Returns:
        tuple: (decoded_true, decoded_pred)
    """
    if label_encoder is not None:
        try:
            y_true_decoded = label_encoder.inverse_transform(y_true)
            y_pred_decoded = label_encoder.inverse_transform(y_pred)
            return y_true_decoded, y_pred_decoded
        except Exception as e:
            logger.warning(f"Could not decode predictions: {str(e)}")
    
    return y_true, y_pred


#########################################
# Metadata Management
#########################################

def initialize_metadata() -> Dict[str, Any]:
    """
    Initialize a metadata dictionary for tracking the ML pipeline run.
    
    Returns:
        Dictionary with initial metadata
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return {
        "timestamp": timestamp,
        "start_time": datetime.now().isoformat(),
        "parameters": {},
        "data": {},
        "preprocessing": {},
        "models": {},
        "evaluation": {},
        "best_model": {},
        "status": "initialized"
    }

def save_metadata(metadata: Dict[str, Any], output_dir: str, filename: Optional[str] = None) -> str:
    """
    Save metadata to a JSON file.
    
    Args:
        metadata: Dictionary containing metadata
        output_dir: Directory to save the metadata file
        filename: Optional specific filename
        
    Returns:
        Path to the saved metadata file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create a deep copy to avoid modifying the original
    metadata_to_save = metadata.copy()
    
    # Update end time
    metadata_to_save["end_time"] = datetime.now().isoformat()
    
    # Convert any non-serializable objects to strings
    def make_serializable(obj):
        if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            if len(obj) > 100:  # Limit large arrays
                return obj[:100].tolist() + ["... truncated"]
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            if len(obj) > 10:  # Sample for large DataFrames
                return f"DataFrame: {obj.shape}, sample: {obj.head(5).to_dict()}"
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            if len(obj) > 10:  # Sample for large Series
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
    
    metadata_to_save = process_dict(metadata_to_save)
    
    # Generate filename if not provided
    if not filename:
        timestamp = metadata.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        filename = f"metadata_{timestamp}.json"
    
    # Save to file
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'w') as f:
        json.dump(metadata_to_save, f, indent=2)
    
    logger.info(f"Metadata saved to {file_path}")
    return file_path


#########################################
# Custom Transformers
#########################################

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for outlier detection and handling.
    
    Detects outliers using IQR or Z-score method and handles them 
    via clipping or removal.
    """
    
    def __init__(self, method='iqr', threshold=1.5, strategy='clip'):
        """
        Initialize the outlier handler.
        
        Args:
            method: Detection method, either 'iqr' or 'zscore'
            threshold: Threshold value for outlier detection
            strategy: Handling strategy, either 'clip', 'remove', or 'none'
        """
        self.method = method
        self.threshold = threshold
        self.strategy = strategy
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        self.feature_names_in_ = None
    
    def fit(self, X, y=None):
        """
        Fit the outlier handler by calculating bounds for each feature.
        
        Args:
            X: Features DataFrame or array
            y: Target variable (not used)
            
        Returns:
            Self
        """
        logger.debug("Fitting OutlierHandler")
        
        # Convert numpy array to DataFrame if needed
        if isinstance(X, np.ndarray):
            if hasattr(self, 'feature_names_in_'):
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                # Create generic column names
                self.feature_names_in_ = [f'feature_{i}' for i in range(X.shape[1])]
                X = pd.DataFrame(X, columns=self.feature_names_in_)
        
        # Store column names
        self.feature_names_in_ = X.columns.tolist()
        
        # Calculate bounds for numeric columns
        outlier_stats = {}
        for col in X.columns:
            if np.issubdtype(X[col].dtype, np.number):
                if self.method == 'iqr':
                    q1 = X[col].quantile(0.25)
                    q3 = X[col].quantile(0.75)
                    iqr = q3 - q1
                    self.lower_bounds_[col] = q1 - (self.threshold * iqr)
                    self.upper_bounds_[col] = q3 + (self.threshold * iqr)
                    outlier_stats[col] = {'q1': float(q1), 'q3': float(q3), 'iqr': float(iqr), 
                                         'lower_bound': float(self.lower_bounds_[col]), 
                                         'upper_bound': float(self.upper_bounds_[col])}
                elif self.method == 'zscore':
                    mean = X[col].mean()
                    std = X[col].std()
                    self.lower_bounds_[col] = mean - (self.threshold * std)
                    self.upper_bounds_[col] = mean + (self.threshold * std)
                    outlier_stats[col] = {'mean': float(mean), 'std': float(std), 
                                         'lower_bound': float(self.lower_bounds_[col]), 
                                         'upper_bound': float(self.upper_bounds_[col])}
        
        logger.debug(f"OutlierHandler fitted with {self.method} method, threshold={self.threshold}")
        logger.debug(f"Outlier statistics calculated for {len(outlier_stats)} numeric columns")
        return self
    
    def transform(self, X):
        """
        Transform the data by handling outliers.
        
        Args:
            X: Features DataFrame or array
            
        Returns:
            Transformed DataFrame with outliers handled
        """
        logger.debug("Transforming data with OutlierHandler")
        
        # Convert numpy array to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        
        X_transformed = X.copy()
        
        # Skip if strategy is none
        if self.strategy == 'none':
            return X_transformed
        
        # Process each column
        outlier_counts = {}
        for col in self.lower_bounds_.keys():
            if col in X_transformed.columns:
                if self.strategy == 'clip':
                    # Count outliers before clipping
                    lower_outliers = (X_transformed[col] < self.lower_bounds_[col]).sum()
                    upper_outliers = (X_transformed[col] > self.upper_bounds_[col]).sum()
                    
                    X_transformed[col] = X_transformed[col].clip(
                        lower=self.lower_bounds_[col],
                        upper=self.upper_bounds_[col]
                    )
                    
                    outlier_counts[col] = {'lower_outliers': int(lower_outliers), 
                                         'upper_outliers': int(upper_outliers),
                                         'total_outliers': int(lower_outliers + upper_outliers)}
                    
                elif self.strategy == 'remove':
                    mask = (
                        (X_transformed[col] >= self.lower_bounds_[col]) & 
                        (X_transformed[col] <= self.upper_bounds_[col])
                    )
                    removed_rows = (~mask).sum()
                    X_transformed = X_transformed[mask]
                    
                    outlier_counts[col] = {'removed_rows': int(removed_rows)}
        
        total_outliers = sum(stats.get('total_outliers', 0) for stats in outlier_counts.values())
        logger.debug(f"OutlierHandler processed {total_outliers} outliers with strategy '{self.strategy}'")
        return X_transformed

class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer for datetime feature extraction.
    
    Extracts useful features from datetime columns including date components,
    cyclic encodings, and time distances.
    """
    
    def __init__(self, date_features=True, cyclic_features=True, custom_ref_dates=None):
        """
        Initialize datetime feature extractor.
        
        Args:
            date_features: Whether to extract date components
            cyclic_features: Whether to extract cyclic features
            custom_ref_dates: Custom reference dates for distance calculation
        """
        self.date_features = date_features
        self.cyclic_features = cyclic_features
        self.custom_ref_dates = custom_ref_dates or {}
        self.datetime_columns_ = []
        self.feature_names_in_ = None
    
    def fit(self, X, y=None):
        """
        Fit the transformer by identifying datetime columns.
        
        Args:
            X: Features DataFrame or array
            y: Target variable (not used)
            
        Returns:
            Self
        """
        logger.debug("Fitting DatetimeFeatureExtractor")
        
        # Convert numpy array to DataFrame if needed
        if isinstance(X, np.ndarray):
            if hasattr(self, 'feature_names_in_'):
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                # Create generic column names
                self.feature_names_in_ = [f'feature_{i}' for i in range(X.shape[1])]
                X = pd.DataFrame(X, columns=self.feature_names_in_)
        
        # Store column names
        self.feature_names_in_ = X.columns.tolist()
        self.datetime_columns_ = []
        
        # Identify datetime columns
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
        
        logger.debug(f"DatetimeFeatureExtractor identified {len(self.datetime_columns_)} datetime columns: {self.datetime_columns_}")
        return self

    def transform(self, X):
        """
        Transform the data by extracting datetime features.
        
        Args:
            X: Features DataFrame or array
            
        Returns:
            Transformed DataFrame with datetime features added
        """
        logger.debug("Transforming data with DatetimeFeatureExtractor")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        X_transformed = X.copy()
        
        features_created = 0
        for col in self.datetime_columns_:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(X_transformed[col]):
                try:
                    X_transformed[col] = pd.to_datetime(X_transformed[col], errors='coerce')
                except:
                    # If conversion fails, skip this column
                    logger.warning(f"Failed to convert column '{col}' to datetime. Skipping.")
                    continue
            
            col_features_created = 0
            # Basic date components
            if self.date_features:
                X_transformed[f'{col}_year'] = X_transformed[col].dt.year
                X_transformed[f'{col}_month'] = X_transformed[col].dt.month
                X_transformed[f'{col}_day'] = X_transformed[col].dt.day
                X_transformed[f'{col}_dayofweek'] = X_transformed[col].dt.dayofweek
                X_transformed[f'{col}_dayofyear'] = X_transformed[col].dt.dayofyear
                X_transformed[f'{col}_quarter'] = X_transformed[col].dt.quarter
                col_features_created += 6
                
                # Hour components if time data is present
                if (X_transformed[col].dt.hour > 0).any():
                    X_transformed[f'{col}_hour'] = X_transformed[col].dt.hour
                    X_transformed[f'{col}_minute'] = X_transformed[col].dt.minute
                    col_features_created += 2
                
                # Flag features
                X_transformed[f'{col}_is_weekend'] = X_transformed[col].dt.dayofweek >= 5
                X_transformed[f'{col}_is_month_end'] = X_transformed[col].dt.is_month_end
                col_features_created += 2
            
            # Cyclic features to handle periodicity
            if self.cyclic_features:
                # Month has cycle of 12
                X_transformed[f'{col}_month_sin'] = np.sin(2 * np.pi * X_transformed[col].dt.month / 12)
                X_transformed[f'{col}_month_cos'] = np.cos(2 * np.pi * X_transformed[col].dt.month / 12)
                col_features_created += 2
                
                # Day of week has cycle of 7
                X_transformed[f'{col}_dayofweek_sin'] = np.sin(2 * np.pi * X_transformed[col].dt.dayofweek / 7)
                X_transformed[f'{col}_dayofweek_cos'] = np.cos(2 * np.pi * X_transformed[col].dt.dayofweek / 7)
                col_features_created += 2
                
                # Hour has cycle of 24 (if time data exists)
                if (X_transformed[col].dt.hour > 0).any():
                    X_transformed[f'{col}_hour_sin'] = np.sin(2 * np.pi * X_transformed[col].dt.hour / 24)
                    X_transformed[f'{col}_hour_cos'] = np.cos(2 * np.pi * X_transformed[col].dt.hour / 24)
                    col_features_created += 2
            
            # Distance from reference dates
            for ref_name, ref_date in self.custom_ref_dates.items():
                ref_date = pd.to_datetime(ref_date)
                X_transformed[f'{col}_days_from_{ref_name}'] = (X_transformed[col] - ref_date).dt.days
                col_features_created += 1
            
            # Default time since epoch
            X_transformed[f'{col}_days_from_epoch'] = (X_transformed[col] - pd.Timestamp('1970-01-01')).dt.days
            col_features_created += 1
            
            features_created += col_features_created
            logger.debug(f"Created {col_features_created} datetime features for column '{col}'")
            
        logger.debug(f"DatetimeFeatureExtractor created a total of {features_created} new features")
        return X_transformed

class TimeSeriesFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract time series features from date/time columns."""
    
    def __init__(self, date_features=True, cyclic_features=True, lag_features=True, 
                 rolling_features=True, diff_features=True, lag_orders=None, 
                 rolling_windows=None, rolling_functions=None, diff_orders=None,
                 logger=None):
        """Initialize time series feature extractor."""
        self.date_features = date_features
        self.cyclic_features = cyclic_features
        self.lag_features = lag_features
        self.rolling_features = rolling_features
        self.diff_features = diff_features
        self.lag_orders = lag_orders or [1, 7, 14, 30]
        self.rolling_windows = rolling_windows or [7, 14, 30]
        self.rolling_functions = rolling_functions or ['mean', 'std', 'min', 'max']
        self.diff_orders = diff_orders or [1, 7]
        self.feature_names_in_ = None
        self.logger = logger or logging.getLogger(__name__)
        
    def fit(self, X, y=None):
        """Fit the extractor (just store column names)."""
        # Store column names
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f'feature_{i}' for i in range(X.shape[1])]
        return self
        
    def transform(self, X):
        """Transform data by extracting time series features."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        
        X_transformed = X.copy()
        
        # Make sure the dataframe has a datetime index
        if not isinstance(X_transformed.index, pd.DatetimeIndex):
            self.logger.debug("DataFrame does not have DatetimeIndex")
            return X_transformed
        
        # Extract date features
        if self.date_features:
            X_transformed['day_of_week'] = X_transformed.index.dayofweek
            X_transformed['day_of_month'] = X_transformed.index.day
            X_transformed['day_of_year'] = X_transformed.index.dayofyear
            X_transformed['month'] = X_transformed.index.month
            X_transformed['quarter'] = X_transformed.index.quarter
            X_transformed['year'] = X_transformed.index.year
            X_transformed['is_weekend'] = X_transformed.index.dayofweek >= 5
        
        # Extract cyclic features
        if self.cyclic_features:
            X_transformed['day_of_week_sin'] = np.sin(2 * np.pi * X_transformed.index.dayofweek / 7)
            X_transformed['day_of_week_cos'] = np.cos(2 * np.pi * X_transformed.index.dayofweek / 7)
            X_transformed['month_sin'] = np.sin(2 * np.pi * X_transformed.index.month / 12)
            X_transformed['month_cos'] = np.cos(2 * np.pi * X_transformed.index.month / 12)
        
        # For lag/rolling/diff features, we need numeric columns
        numeric_cols = X_transformed.select_dtypes(include=np.number).columns.tolist()
        
        # Extract lag features
        if self.lag_features and numeric_cols:
            for col in numeric_cols:
                for lag in self.lag_orders:
                    X_transformed[f'{col}_lag_{lag}'] = X_transformed[col].shift(lag)
        
        # Extract rolling window features
        if self.rolling_features and numeric_cols:
            for col in numeric_cols:
                for window in self.rolling_windows:
                    rolling_obj = X_transformed[col].rolling(window)
                    
                    for func in self.rolling_functions:
                        if hasattr(rolling_obj, func):
                            X_transformed[f'{col}_rolling_{window}_{func}'] = getattr(rolling_obj, func)()
        
        # Extract differencing features
        if self.diff_features and numeric_cols:
            for col in numeric_cols:
                for order in self.diff_orders:
                    X_transformed[f'{col}_diff_{order}'] = X_transformed[col].diff(order)
        
        return X_transformed

#########################################
# Directory & Versioning Management
#########################################

def get_next_version_dir(base_output_dir: str, model_id: str, max_versions: int = 5) -> Tuple[str, int]:
    """
    Get the next version directory for model outputs.
    
    Args:
        base_output_dir: Base output directory
        model_id: Model ID
        max_versions: Maximum number of versions to keep
        
    Returns:
        Tuple of (next_version_dir, next_version_number)
    """
    # Create model directory if it doesn't exist (should be base_output_dir)
    model_dir = base_output_dir
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
                logger.info(f"Deleted old version directory: {dir_to_delete}")
            except Exception as e:
                logger.error(f"Error deleting {dir_to_delete}: {str(e)}")
    
    return os.path.join(model_dir, next_version), next_version_num