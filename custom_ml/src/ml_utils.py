import argparse
import os
import json
import pickle
import shutil
import yaml
import logging
import numpy as np
import pandas as pd
import traceback
from datetime import datetime
from typing import Callable, Dict,Optional, Any,Tuple
from sklearn.base import BaseEstimator, TransformerMixin


# Initialize logger
logger = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)

#########################################
# Configuration Loading Utilities
#########################################

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file or return pre-loaded config.
    
    Args:
        config_path: Path to the YAML configuration file (optional)
        
    Returns:
        Dictionary containing the configuration
    """
    from custom_ml_api.config import CONFIG, CONFIG_PATH
    
    # If no path specified or path is the same as main config, return pre-loaded config
    if config_path is None or config_path == CONFIG_PATH:
        return CONFIG
        
    # Otherwise, load from the specified path
    logger.debug(f"Loading configuration from: {config_path}")
    start_time = datetime.now()
    
    if not os.path.exists(config_path):
        error_msg = f"Configuration file not found: {config_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        # Log configuration size and structure
        config_size = os.path.getsize(config_path)
        top_level_keys = list(config.keys())
        load_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Configuration loaded from {config_path} ({config_size/1024:.1f} KB) in {load_time:.3f} seconds")
        logger.debug(f"Configuration has {len(top_level_keys)} top-level sections: {top_level_keys}")
        
        return config
    except yaml.YAMLError as e:
        error_msg = f"Error parsing YAML configuration: {str(e)}"
        logger.error(error_msg)
        logger.debug(f"YAML parsing error details: {traceback.format_exc()}")
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"Error loading configuration: {str(e)}"
        logger.error(error_msg)
        logger.debug(f"Configuration loading error details: {traceback.format_exc()}")
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
    logger.debug(f"Getting model configuration for problem type: {problem_type}")
    start_time = datetime.now()
    
    if not config:
        logger.warning("Empty configuration provided. Returning empty dictionary.")
        return {}
    
    common_config = config.get('common', {})
    model_config = config.get(problem_type, {})
    
    logger.debug(f"Found common config with {len(common_config)} keys and {problem_type} config with {len(model_config)} keys")
    
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
    
    # Log the merge results
    merge_time = (datetime.now() - start_time).total_seconds()
    
    # Log merged configuration structure
    merged_keys = list(merged_config.keys())
    logger.debug(f"Merged configuration has {len(merged_keys)} top-level keys: {merged_keys}")
    
    # Log specific model configuration details
    if 'models' in merged_config:
        enabled_models = merged_config.get('models', {}).get('enabled', [])
        logger.debug(f"Enabled models for {problem_type}: {enabled_models}")
    
    # Log evaluation metrics
    if 'evaluation' in merged_config:
        metrics = merged_config.get('evaluation', {}).get('metrics', [])
        primary_metric = merged_config.get('evaluation', {}).get('primary_metric', None)
        logger.debug(f"Evaluation metrics: {metrics}, primary: {primary_metric}")
    
    # Log hyperparameter optimization settings
    if 'random_search' in merged_config and merged_config['random_search'].get('enabled', False):
        rs_config = merged_config['random_search']
        n_iter = rs_config.get('n_iter', 20)
        cv = rs_config.get('cv', 5)
        rs_models = [k for k, v in rs_config.get('models', {}).items() if v.get('enabled', False)]
        logger.debug(f"Random search enabled: {n_iter} iterations, {cv}-fold CV for models: {rs_models}")
    
    logger.info(f"Model configuration for {problem_type} generated in {merge_time:.3f} seconds")
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
    logger.info(f"Detecting problem type for dataset with shape: {df.shape}")
    start_time = datetime.now()
    
    detection_steps = []  # Track detection steps for debugging
    
    try:
        # Initial validation
        if target_column is None or target_column == "":
            logger.info("No target column provided. Assuming 'clustering'.")
            detection_steps.append("No target column → clustering")
            return 'clustering'

        if target_column not in df.columns:
            error_msg = f"Target column '{target_column}' not found in DataFrame."
            logger.error(error_msg)
            logger.debug(f"Available columns: {df.columns.tolist()}")
            raise ValueError(error_msg)

        # Get configuration with defaults
        config = config or {}
        detection_config = config.get('problem_detection', {})
        class_detection = detection_config.get('classification', {}).get('class_detection', {})
        max_classes = class_detection.get('max_classes', 20)  # Default to 20 if not found
        
        logger.debug(f"Problem detection configuration: max_classes={max_classes}")
        
        # Get target series
        target = df[target_column]
        logger.debug(f"Analyzing target column '{target_column}' with dtype: {target.dtype}")
        
        # Log target column basic statistics
        if pd.api.types.is_numeric_dtype(target):
            target_stats = target.describe()
            logger.debug(f"Target statistics: min={target_stats['min']}, max={target_stats['max']}, mean={target_stats['mean']}, std={target_stats['std']}")
        else:
            unique_values = target.nunique()
            top_values = target.value_counts().head(5).to_dict()
            logger.debug(f"Target has {unique_values} unique values. Top 5: {top_values}")

        # Step 1: Basic type detection
        if pd.api.types.is_categorical_dtype(target) or pd.api.types.is_object_dtype(target):
            unique_count = target.nunique()
            min_samples = detection_config.get('min_samples_per_class', 3)
            value_counts = target.value_counts()
            
            logger.debug(f"Target is categorical/object with {unique_count} unique values")
            logger.debug(f"Value counts: {value_counts.head(10).to_dict() if unique_count > 10 else value_counts.to_dict()}")
            logger.debug(f"Checking if all classes have at least {min_samples} samples")
            
            if (value_counts >= min_samples).all():
                detection_steps.append(f"Categorical target with {unique_count} classes → classification")
                detection_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Categorical target detected with {unique_count} classes. Problem type: classification (detection time: {detection_time:.3f}s)")
                return 'classification'
            else:
                classes_below_threshold = value_counts[value_counts < min_samples].index.tolist()
                logger.debug(f"Some classes have fewer than {min_samples} samples: {classes_below_threshold}")
                detection_steps.append("Some classes have too few samples → continue detection")

        # Step 2: Time Series Detection
        date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        index_is_datetime = isinstance(df.index, pd.DatetimeIndex)
        
        logger.debug(f"Date columns found: {date_columns}")
        logger.debug(f"Index is datetime: {index_is_datetime}")

        # Time series checks only if datetime columns exist and target is numeric
        if (len(date_columns) > 0 or index_is_datetime) and pd.api.types.is_numeric_dtype(target):
            min_points = detection_config.get('time_series', {}).get('min_points', 10)
            logger.debug(f"Checking for time series pattern with minimum {min_points} data points")
            
            if len(df) >= min_points:
                date_col = date_columns[0] if date_columns else df.index
                
                # Check temporal order
                if isinstance(date_col, pd.Index):
                    is_sorted = date_col.is_monotonic_increasing or date_col.is_monotonic_decreasing
                    logger.debug(f"Date index is temporally sorted: {is_sorted}")
                else:
                    temp_dates = pd.to_datetime(df[date_col])
                    is_sorted = temp_dates.is_monotonic_increasing or temp_dates.is_monotonic_decreasing
                    logger.debug(f"Date column '{date_col}' is temporally sorted: {is_sorted}")

                if is_sorted:
                    detection_steps.append("Sorted datetime index/column with numeric target → time_series")
                    detection_time = (datetime.now() - start_time).total_seconds()
                    logger.info(f"Time series pattern detected. Problem type: time_series (detection time: {detection_time:.3f}s)")
                    return 'time_series'
                else:
                    detection_steps.append("Datetime present but not sorted → continue detection")
            else:
                logger.debug(f"Not enough data points for time series: {len(df)} < {min_points}")
                detection_steps.append("Not enough data points for time series → continue detection")

        # Step 3: Numeric Analysis for Classification vs Regression
        if pd.api.types.is_numeric_dtype(target):
            # Basic statistics
            unique_values = target.unique()
            n_unique = len(unique_values)
            n_samples = len(target)
            unique_ratio = n_unique / n_samples
            
            logger.debug(f"Numeric target analysis: {n_unique} unique values out of {n_samples} samples (unique ratio: {unique_ratio:.4f})")

            # Binary classification check
            if n_unique == 2:
                detection_steps.append("Numeric target with exactly 2 unique values → binary classification")
                detection_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Binary classification detected. Problem type: classification (detection time: {detection_time:.3f}s)")
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
                
                gaps_ratio = gaps_std / gaps_mean if gaps_mean != 0 else float('inf')
                logger.debug(f"Gaps analysis: mean={gaps_mean:.4f}, std={gaps_std:.4f}, ratio={gaps_ratio:.4f}")
                
                # Regular gaps suggest classification
                if gaps_ratio < 0.1:
                    class_indicators += 1
                    logger.debug("Regular gaps detected (std/mean < 0.1) → +1 classification indicator")
                else:
                    logger.debug("Irregular gaps detected → no classification indicator")
            
            # Unique value ratio analysis
            if unique_ratio < 0.01:  # Few unique values
                class_indicators += 1
                logger.debug(f"Low unique ratio ({unique_ratio:.4f} < 0.01) → +1 classification indicator")
            if unique_ratio > 0.3:  # Many unique values
                regression_indicators += 1
                logger.debug(f"High unique ratio ({unique_ratio:.4f} > 0.3) → +1 regression indicator")

            # Value range analysis
            value_range = float(target.max() - target.min())
            if value_range > 100 and unique_ratio > 0.1:
                regression_indicators += 1
                logger.debug(f"Wide value range ({value_range:.1f} > 100) with high unique ratio → +1 regression indicator")

            # Make decision based on indicators
            logger.debug(f"Classification indicators: {class_indicators}, Regression indicators: {regression_indicators}")

            if class_indicators > regression_indicators:
                if n_unique <= max_classes:
                    detection_steps.append(f"More classification indicators ({class_indicators} > {regression_indicators}) with {n_unique} classes ≤ {max_classes} → classification")
                    detection_time = (datetime.now() - start_time).total_seconds()
                    logger.info(f"Multi-class classification detected with {n_unique} classes. Problem type: classification (detection time: {detection_time:.3f}s)")
                    return 'classification'
                else:
                    logger.debug(f"Too many unique values for classification: {n_unique} > {max_classes}")
                    detection_steps.append(f"Too many classes ({n_unique} > {max_classes}) → regression")

            # Check against strict classification criteria
            if n_unique > max_classes or unique_ratio > 0.01:
                detection_steps.append(f"Many unique values ({n_unique} > {max_classes} or ratio {unique_ratio:.4f} > 0.01) → regression")
                detection_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Regression problem detected. Problem type: regression (detection time: {detection_time:.3f}s)")
                return 'regression'

            # Default to regression for numeric targets
            detection_steps.append("Numeric target without specific indicators → default to regression")
            detection_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Defaulting to regression for numeric target. Problem type: regression (detection time: {detection_time:.3f}s)")
            return 'regression'

        # If we reach here, default based on data type
        if pd.api.types.is_numeric_dtype(target):
            detection_steps.append("Fallback for numeric target → regression")
            detection_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Defaulting to regression for numeric target. Problem type: regression (detection time: {detection_time:.3f}s)")
            return 'regression'
        else:
            detection_steps.append("Fallback for non-numeric target → classification")
            detection_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Defaulting to classification for non-numeric target. Problem type: classification (detection time: {detection_time:.3f}s)")
            return 'classification'

    except Exception as e:
        detection_time = (datetime.now() - start_time).total_seconds()
        error_msg = f"Error in problem type detection after {detection_time:.3f}s: {str(e)}"
        logger.error(error_msg)
        logger.debug(f"Detection steps attempted: {detection_steps}")
        logger.debug(f"Error details: {traceback.format_exc()}")
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
    logger.info(f"Loading custom function '{function_name}' from {function_path}")
    start_time = datetime.now()
    
    if not os.path.exists(function_path):
        logger.error(f"Custom function file not found: {function_path}")
        return None
    
    try:
        import importlib.util
        import sys
        
        logger.debug(f"Importing module from {function_path}")
        
        # Dynamically import the module
        spec = importlib.util.spec_from_file_location("custom_module", function_path)
        custom_module = importlib.util.module_from_spec(spec)
        sys.modules["custom_module"] = custom_module
        spec.loader.exec_module(custom_module)
        
        # Get module attributes
        module_attributes = dir(custom_module)
        logger.debug(f"Module has {len(module_attributes)} attributes")
        logger.debug(f"Available functions in module: {[attr for attr in module_attributes if callable(getattr(custom_module, attr)) and not attr.startswith('_')]}")
        
        # Check if the module has the specified function
        if hasattr(custom_module, function_name):
            function = getattr(custom_module, function_name)
            
            # Log function signature if possible
            try:
                import inspect
                signature = str(inspect.signature(function))
                logger.debug(f"Function signature: {function_name}{signature}")
                
                # Check if docstring exists
                if function.__doc__:
                    logger.debug(f"Function has docstring: {len(function.__doc__)} characters")
            except Exception as sig_error:
                logger.debug(f"Could not get function signature: {str(sig_error)}")
            
            load_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Successfully loaded function '{function_name}' in {load_time:.3f}s")
            return function
        else:
            available_funcs = [attr for attr in module_attributes if callable(getattr(custom_module, attr)) and not attr.startswith('_')]
            logger.error(f"Custom module at {function_path} does not have function '{function_name}'. Available functions: {available_funcs}")
            return None
    except Exception as e:
        load_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Failed to load custom function after {load_time:.3f}s: {str(e)}")
        logger.debug(f"Error details: {traceback.format_exc()}")
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
    start_time = datetime.now()
    logger.debug(f"Setting up logging with base level: {level}")
    
    # Create logs directory if it doesn't exist and if provided
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            logger.debug(f"Created or confirmed log directory: {log_dir}")
        except Exception as e:
            logger.warning(f"Could not create log directory {log_dir}: {str(e)}")
            log_dir = None
    
    # Create a timestamped log file if not specified
    if log_file is None and log_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"ml_pipeline_{timestamp}.log")
        logger.debug(f"Generated timestamped log file: {log_file}")
    
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
    
    logger.debug(f"Log levels - Overall: {level}, Console: {console_level}, File: {file_level}")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Count existing handlers before removing
    existing_handlers = len(root_logger.handlers)
    if existing_handlers > 0:
        logger.debug(f"Removing {existing_handlers} existing log handlers")
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Set up file handler if we have a log file
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(file_log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logger.debug(f"Added file handler with level {file_level} to {log_file}")
        except Exception as e:
            logger.warning(f"Could not create file handler for {log_file}: {str(e)}")
            logger.debug(f"File handler error details: {traceback.format_exc()}")
    else:
        logger.warning("No log file specified or could not create log directory. File logging disabled.")
    
    # Console handler
    try:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        logger.debug(f"Added console handler with level {console_level}")
    except Exception as e:
        logger.warning(f"Could not create console handler: {str(e)}")
    
    # Check if handlers were successfully added
    if len(root_logger.handlers) == 0:
        logger.warning("No logging handlers could be created. Logging may not work correctly.")
    
    setup_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Logging configured in {setup_time:.3f}s. Log file: {log_file}")
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
    logger.debug("Decoding predictions to original labels")
    start_time = datetime.now()
    
    # Check input shapes
    if hasattr(y_true, 'shape') and hasattr(y_pred, 'shape'):
        logger.debug(f"Input shapes - y_true: {y_true.shape}, y_pred: {y_pred.shape}")
    else:
        logger.debug(f"Input lengths - y_true: {len(y_true)}, y_pred: {len(y_pred)}")
    
    if label_encoder is not None:
        try:
            # Get encoder details if available
            if hasattr(label_encoder, 'classes_'):
                n_classes = len(label_encoder.classes_)
                logger.debug(f"Label encoder has {n_classes} classes: {label_encoder.classes_}")
            
            # Log a sample of encoded values
            sample_size = min(5, len(y_true))
            sample_true = y_true[:sample_size]
            sample_pred = y_pred[:sample_size]
            logger.debug(f"Encoded samples (first {sample_size}): y_true={sample_true}, y_pred={sample_pred}")
            
            # Perform transformation
            decode_start = datetime.now()
            y_true_decoded = label_encoder.inverse_transform(y_true)
            y_pred_decoded = label_encoder.inverse_transform(y_pred)
            decode_time = (datetime.now() - decode_start).total_seconds()
            
            # Log decoded samples
            sample_true_decoded = y_true_decoded[:sample_size]
            sample_pred_decoded = y_pred_decoded[:sample_size]
            logger.debug(f"Decoded samples: y_true={sample_true_decoded}, y_pred={sample_pred_decoded}")
            logger.debug(f"Decoding completed in {decode_time:.3f}s")
            
            total_time = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Prediction decoding completed in {total_time:.3f}s")
            return y_true_decoded, y_pred_decoded
        except Exception as e:
            logger.warning(f"Could not decode predictions: {str(e)}")
            logger.debug(f"Decoding error details: {traceback.format_exc()}")
    else:
        logger.debug("No label encoder provided, returning original values")
    
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
    logger.debug("Initializing ML pipeline metadata")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get system info for metadata
    system_info = {}
    try:
        import platform
        system_info = {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'processor': platform.processor()
        }
        
        # Try to get pandas and numpy versions
        try:
            system_info['pandas_version'] = pd.__version__
            system_info['numpy_version'] = np.__version__
        except:
            pass
            
        # Try to get sklearn version
        try:
            import sklearn
            system_info['sklearn_version'] = sklearn.__version__
        except:
            pass
    except:
        logger.debug("Could not get complete system information")
    
    metadata = {
        "timestamp": timestamp,
        "start_time": datetime.now().isoformat(),
        "parameters": {},
        "data": {},
        "preprocessing": {},
        "models": {},
        "evaluation": {},
        "best_model": {},
        "status": "initialized",
        "system_info": system_info
    }
    
    logger.debug(f"Metadata initialized with timestamp: {timestamp}")
    logger.debug(f"System info captured: {system_info}")
    
    return metadata

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
    logger.info(f"Saving metadata to directory: {output_dir}")
    start_time = datetime.now()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.warning(f"Could not create output directory: {str(e)}")
            # Fallback to current directory
            output_dir = "."
            logger.warning(f"Falling back to current directory: {output_dir}")
    
    # Create a deep copy to avoid modifying the original
    metadata_to_save = metadata.copy()
    
    # Update end time
    metadata_to_save["end_time"] = datetime.now().isoformat()
    
    # Calculate runtime
    try:
        start_time_dt = datetime.fromisoformat(metadata_to_save["start_time"])
        end_time_dt = datetime.fromisoformat(metadata_to_save["end_time"])
        runtime_seconds = (end_time_dt - start_time_dt).total_seconds()
        metadata_to_save["runtime_seconds"] = runtime_seconds
        logger.debug(f"Total runtime: {runtime_seconds:.2f} seconds")
    except Exception as e:
        logger.debug(f"Could not calculate runtime: {str(e)}")
    
    # Log the size of the metadata
    metadata_size = _get_object_size(metadata_to_save)
    logger.debug(f"Metadata size: {metadata_size/1024:.1f} KB")
    
    # Identify potentially large sections
    large_sections = []
    for section, data in metadata_to_save.items():
        section_size = _get_object_size(data)
        if section_size > 100 * 1024:  # If larger than 100KB
            large_sections.append((section, section_size))
    
    if large_sections:
        logger.debug("Large metadata sections detected:")
        for section, size in sorted(large_sections, key=lambda x: x[1], reverse=True):
            logger.debug(f"  - {section}: {size/1024:.1f} KB")
    
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
    
    logger.debug("Converting metadata to JSON-serializable format")
    metadata_to_save = process_dict(metadata_to_save)
    
    # Generate filename if not provided
    if not filename:
        timestamp = metadata.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        filename = f"metadata_{timestamp}.json"
        logger.debug(f"Generated filename: {filename}")
    
    # Save to file
    file_path = os.path.join(output_dir, filename)
    try:
        with open(file_path, 'w') as f:
            json.dump(metadata_to_save, f, indent=2)
        
        file_size = os.path.getsize(file_path)
        save_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Metadata saved to {file_path} ({file_size/1024:.1f} KB) in {save_time:.3f}s")
    except Exception as e:
        logger.error(f"Error saving metadata: {str(e)}")
        logger.debug(f"Error details: {traceback.format_exc()}")
        
        # Try to save to a different location
        try:
            fallback_path = os.path.join(".", f"metadata_fallback_{timestamp}.json")
            with open(fallback_path, 'w') as f:
                json.dump(metadata_to_save, f, indent=2)
            logger.warning(f"Metadata saved to fallback location: {fallback_path}")
            return fallback_path
        except Exception as fallback_error:
            logger.error(f"Could not save metadata to fallback location: {str(fallback_error)}")
            raise
    
    return file_path

def _get_object_size(obj):
    """Helper function to estimate object size in bytes."""
    import sys
    import json
    
    try:
        # For simple objects, use sys.getsizeof
        size = sys.getsizeof(obj)
        
        # For dictionaries, recursively add size of keys and values
        if isinstance(obj, dict):
            size += sum(_get_object_size(k) + _get_object_size(v) for k, v in obj.items())
        # For lists or tuples, recursively add size of items
        elif isinstance(obj, (list, tuple)):
            size += sum(_get_object_size(item) for item in obj)
        
        return size
    except Exception:
        # Fallback: try to estimate by converting to JSON
        try:
            json_str = json.dumps(obj)
            return len(json_str)
        except:
            # If all else fails, return a nominal size
            return 100  # arbitrary small number


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
        
        logger.debug(f"Initialized OutlierHandler with method={method}, threshold={threshold}, strategy={strategy}")
    
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
        start_time = datetime.now()
        
        # Convert numpy array to DataFrame if needed
        if isinstance(X, np.ndarray):
            if hasattr(self, 'feature_names_in_'):
                X = pd.DataFrame(X, columns=self.feature_names_in_)
                logger.debug(f"Converted numpy array to DataFrame using existing feature names")
            else:
                # Create generic column names
                self.feature_names_in_ = [f'feature_{i}' for i in range(X.shape[1])]
                X = pd.DataFrame(X, columns=self.feature_names_in_)
                logger.debug(f"Converted numpy array to DataFrame with generic feature names")
        
        # Store column names
        self.feature_names_in_ = X.columns.tolist()
        logger.debug(f"OutlierHandler will process {len(self.feature_names_in_)} features")
        
        # Calculate bounds for numeric columns
        outlier_stats = {}
        numeric_count = 0
        
        for col in X.columns:
            if np.issubdtype(X[col].dtype, np.number):
                numeric_count += 1
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
        
        # Log statistics and potential outliers for a few columns
        if outlier_stats:
            sample_cols = list(outlier_stats.keys())[:3]  # Take first 3 columns as samples
            logger.debug(f"Sample outlier statistics for first {len(sample_cols)} columns:")
            
            for col in sample_cols:
                lower = self.lower_bounds_[col]
                upper = self.upper_bounds_[col]
                
                # Count potential outliers
                lower_outliers = (X[col] < lower).sum()
                upper_outliers = (X[col] > upper).sum()
                total_outliers = lower_outliers + upper_outliers
                outlier_pct = total_outliers / len(X) * 100
                
                if self.method == 'iqr':
                    logger.debug(f"  - {col}: Q1={outlier_stats[col]['q1']:.4g}, Q3={outlier_stats[col]['q3']:.4g}, IQR={outlier_stats[col]['iqr']:.4g}")
                elif self.method == 'zscore':
                    logger.debug(f"  - {col}: Mean={outlier_stats[col]['mean']:.4g}, Std={outlier_stats[col]['std']:.4g}")
                
                logger.debug(f"    Bounds: [{lower:.4g}, {upper:.4g}], Outliers: {total_outliers} ({outlier_pct:.2f}%)")
        
        fit_time = (datetime.now() - start_time).total_seconds()
        logger.debug(f"OutlierHandler fitted with {numeric_count} numeric columns in {fit_time:.3f}s")
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
        start_time = datetime.now()
        
        # Convert numpy array to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
            logger.debug("Converted numpy array to DataFrame for transformation")
        
        X_transformed = X.copy()
        
        # Log input dimensions
        logger.debug(f"Input shape: {X.shape}")
        
        # Skip if strategy is none
        if self.strategy == 'none':
            logger.debug("Strategy is 'none', returning original data without changes")
            return X_transformed
        
        # Process each column
        outlier_counts = {}
        total_lower_outliers = 0
        total_upper_outliers = 0
        total_removed_rows = 0
        
        for col in self.lower_bounds_.keys():
            if col in X_transformed.columns:
                if self.strategy == 'clip':
                    # Count outliers before clipping
                    lower_outliers = (X_transformed[col] < self.lower_bounds_[col]).sum()
                    upper_outliers = (X_transformed[col] > self.upper_bounds_[col]).sum()
                    
                    total_lower_outliers += lower_outliers
                    total_upper_outliers += upper_outliers
                    
                    if lower_outliers > 0 or upper_outliers > 0:
                        logger.debug(f"Clipping {col}: {lower_outliers} lower outliers, {upper_outliers} upper outliers")
                    
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
                    
                    if removed_rows > 0:
                        logger.debug(f"Removing {removed_rows} rows due to outliers in column {col}")
                        initial_shape = X_transformed.shape
                        X_transformed = X_transformed[mask]
                        logger.debug(f"Shape after removal: {X_transformed.shape} (was {initial_shape})")
                        total_removed_rows += removed_rows
                    
                    outlier_counts[col] = {'removed_rows': int(removed_rows)}
        
        # Log summary statistics
        if self.strategy == 'clip':
            total_outliers = total_lower_outliers + total_upper_outliers
            outlier_pct = total_outliers / (X.shape[0] * len(self.lower_bounds_)) * 100 if len(self.lower_bounds_) > 0 else 0
            logger.debug(f"Clipped {total_outliers} outliers: {total_lower_outliers} lower, {total_upper_outliers} upper ({outlier_pct:.2f}%)")
        elif self.strategy == 'remove':
            removal_pct = total_removed_rows / X.shape[0] * 100 if X.shape[0] > 0 else 0
            logger.debug(f"Removed {total_removed_rows} rows with outliers ({removal_pct:.2f}%)")
        
        transform_time = (datetime.now() - start_time).total_seconds()
        logger.debug(f"OutlierHandler transformation completed in {transform_time:.3f}s")
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
        
        feature_types = []
        if date_features:
            feature_types.append("date components")
        if cyclic_features:
            feature_types.append("cyclical encodings")
        if custom_ref_dates:
            feature_types.append(f"distances from {len(custom_ref_dates)} reference dates")
            
        logger.debug(f"Initialized DatetimeFeatureExtractor to extract: {', '.join(feature_types)}")
    
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
        start_time = datetime.now()
        
        # Convert numpy array to DataFrame if needed
        if isinstance(X, np.ndarray):
            if hasattr(self, 'feature_names_in_'):
                X = pd.DataFrame(X, columns=self.feature_names_in_)
                logger.debug("Converted numpy array to DataFrame using existing feature names")
            else:
                # Create generic column names
                self.feature_names_in_ = [f'feature_{i}' for i in range(X.shape[1])]
                X = pd.DataFrame(X, columns=self.feature_names_in_)
                logger.debug("Converted numpy array to DataFrame with generic feature names")
        
        # Store column names
        self.feature_names_in_ = X.columns.tolist()
        self.datetime_columns_ = []
        
        # Identify datetime columns
        logger.debug(f"Scanning {len(X.columns)} columns for datetime data")
        
        for col in X.columns:
            # Check if it's already a datetime type
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                self.datetime_columns_.append(col)
                logger.debug(f"Identified existing datetime column: {col}")
            # Try to convert string to datetime
            elif X[col].dtype == 'object':
                try:
                    # Try with a small sample first for efficiency
                    sample = X[col].dropna().head(5)
                    if len(sample) > 0:
                        sample_dt = pd.to_datetime(sample, errors='raise')
                        # If successful, check the whole column
                        pd.to_datetime(X[col], errors='raise')
                        self.datetime_columns_.append(col)
                        logger.debug(f"Identified string column that can be converted to datetime: {col}")
                except (ValueError, TypeError) as e:
                    logger.debug(f"Column '{col}' is not a datetime: {str(e)}")
                    continue
        
        fit_time = (datetime.now() - start_time).total_seconds()
        
        if self.datetime_columns_:
            logger.info(f"DatetimeFeatureExtractor identified {len(self.datetime_columns_)} datetime columns in {fit_time:.3f}s: {self.datetime_columns_}")
        else:
            logger.warning("DatetimeFeatureExtractor did not find any datetime columns")
        
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
        start_time = datetime.now()
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
            logger.debug("Converted numpy array to DataFrame for transformation")

        X_transformed = X.copy()
        logger.debug(f"Input shape: {X.shape}")
        
        if not self.datetime_columns_:
            logger.warning("No datetime columns were identified during fit(). No features will be extracted.")
            return X_transformed
        
        features_created = 0
        for col in self.datetime_columns_:
            logger.debug(f"Processing datetime column: {col}")
            col_start_time = datetime.now()
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(X_transformed[col]):
                try:
                    X_transformed[col] = pd.to_datetime(X_transformed[col], errors='coerce')
                    logger.debug(f"Converted '{col}' to datetime")
                except Exception as e:
                    logger.warning(f"Failed to convert column '{col}' to datetime: {str(e)}. Skipping.")
                    continue
            
            col_features_created = 0
            
            # Check for null values
            null_count = X_transformed[col].isna().sum()
            if null_count > 0:
                null_pct = null_count / len(X_transformed) * 100
                logger.warning(f"Column '{col}' has {null_count} null values ({null_pct:.2f}%). Feature extraction may produce NaN values.")
            
            # Basic date components
            if self.date_features:
                logger.debug(f"Extracting date components from '{col}'")
                X_transformed[f'{col}_year'] = X_transformed[col].dt.year
                X_transformed[f'{col}_month'] = X_transformed[col].dt.month
                X_transformed[f'{col}_day'] = X_transformed[col].dt.day
                X_transformed[f'{col}_dayofweek'] = X_transformed[col].dt.dayofweek
                X_transformed[f'{col}_dayofyear'] = X_transformed[col].dt.dayofyear
                X_transformed[f'{col}_quarter'] = X_transformed[col].dt.quarter
                col_features_created += 6
                
                # Check for time components
                has_time = False
                try:
                    # Check if hours are all zeros or if there's actual time data
                    if (X_transformed[col].dt.hour > 0).any():
                        has_time = True
                except:
                    pass
                
                # Hour components if time data is present
                if has_time:
                    logger.debug(f"Time components detected in '{col}', extracting hour and minute")
                    X_transformed[f'{col}_hour'] = X_transformed[col].dt.hour
                    X_transformed[f'{col}_minute'] = X_transformed[col].dt.minute
                    col_features_created += 2
                
                # Flag features
                X_transformed[f'{col}_is_weekend'] = X_transformed[col].dt.dayofweek >= 5
                X_transformed[f'{col}_is_month_end'] = X_transformed[col].dt.is_month_end
                col_features_created += 2
            
            # Cyclic features to handle periodicity
            if self.cyclic_features:
                logger.debug(f"Extracting cyclical encodings from '{col}'")
                
                # Month has cycle of 12
                X_transformed[f'{col}_month_sin'] = np.sin(2 * np.pi * X_transformed[col].dt.month / 12)
                X_transformed[f'{col}_month_cos'] = np.cos(2 * np.pi * X_transformed[col].dt.month / 12)
                col_features_created += 2
                
                # Day of week has cycle of 7
                X_transformed[f'{col}_dayofweek_sin'] = np.sin(2 * np.pi * X_transformed[col].dt.dayofweek / 7)
                X_transformed[f'{col}_dayofweek_cos'] = np.cos(2 * np.pi * X_transformed[col].dt.dayofweek / 7)
                col_features_created += 2
                
                # Hour has cycle of 24 (if time data exists)
                has_time = False
                try:
                    if (X_transformed[col].dt.hour > 0).any():
                        has_time = True
                except:
                    pass
                
                if has_time:
                    X_transformed[f'{col}_hour_sin'] = np.sin(2 * np.pi * X_transformed[col].dt.hour / 24)
                    X_transformed[f'{col}_hour_cos'] = np.cos(2 * np.pi * X_transformed[col].dt.hour / 24)
                    col_features_created += 2
            
            # Distance from reference dates
            if self.custom_ref_dates:
                logger.debug(f"Calculating distances from {len(self.custom_ref_dates)} reference dates")
                
                for ref_name, ref_date in self.custom_ref_dates.items():
                    try:
                        ref_date = pd.to_datetime(ref_date)
                        X_transformed[f'{col}_days_from_{ref_name}'] = (X_transformed[col] - ref_date).dt.days
                        col_features_created += 1
                        logger.debug(f"Created distance feature from reference date '{ref_name}': {ref_date}")
                    except Exception as e:
                        logger.warning(f"Failed to create distance feature for reference date '{ref_name}': {str(e)}")
            
            # Default time since epoch
            X_transformed[f'{col}_days_from_epoch'] = (X_transformed[col] - pd.Timestamp('1970-01-01')).dt.days
            col_features_created += 1
            
            col_time = (datetime.now() - col_start_time).total_seconds()
            features_created += col_features_created
            logger.debug(f"Created {col_features_created} datetime features for column '{col}' in {col_time:.3f}s")
            
        # Log final dimensions
        transform_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"DatetimeFeatureExtractor created {features_created} new features in {transform_time:.3f}s")
        logger.debug(f"Output shape: {X_transformed.shape}")
        
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
        
        feature_types = []
        if date_features:
            feature_types.append("date components")
        if cyclic_features:
            feature_types.append("cyclical encodings")
        if lag_features:
            feature_types.append(f"lag features (orders: {self.lag_orders})")
        if rolling_features:
            feature_types.append(f"rolling window features (windows: {self.rolling_windows}, functions: {self.rolling_functions})")
        if diff_features:
            feature_types.append(f"differencing features (orders: {self.diff_orders})")
            
        self.logger.debug(f"Initialized TimeSeriesFeatureExtractor to extract: {', '.join(feature_types)}")
        
    def fit(self, X, y=None):
        """Fit the extractor (just store column names)."""
        self.logger.debug("Fitting TimeSeriesFeatureExtractor")
        start_time = datetime.now()
        
        # Store column names
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            self.logger.debug(f"Stored {len(self.feature_names_in_)} feature names from DataFrame")
        else:
            self.feature_names_in_ = [f'feature_{i}' for i in range(X.shape[1])]
            self.logger.debug(f"Created {len(self.feature_names_in_)} generic feature names for numpy array")
        
        # Check if index is datetime
        has_datetime_index = False
        if isinstance(X, pd.DataFrame):
            has_datetime_index = isinstance(X.index, pd.DatetimeIndex)
            if has_datetime_index:
                index_range = X.index.max() - X.index.min()
                index_freq = pd.infer_freq(X.index)
                self.logger.debug(f"DataFrame has DatetimeIndex spanning {index_range}, inferred frequency: {index_freq}")
            else:
                self.logger.warning("DataFrame does not have DatetimeIndex. Time series features will be limited.")
        
        fit_time = (datetime.now() - start_time).total_seconds()
        self.logger.debug(f"TimeSeriesFeatureExtractor fit completed in {fit_time:.3f}s")
        return self
        
    def transform(self, X):
        """Transform data by extracting time series features."""
        self.logger.debug("Transforming data with TimeSeriesFeatureExtractor")
        start_time = datetime.now()
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
            self.logger.debug("Converted numpy array to DataFrame for transformation")
        
        X_transformed = X.copy()
        self.logger.debug(f"Input shape: {X.shape}")
        
        # Make sure the dataframe has a datetime index
        if not isinstance(X_transformed.index, pd.DatetimeIndex):
            self.logger.warning("DataFrame does not have DatetimeIndex")
            self.logger.debug(f"Index type: {type(X_transformed.index)}")
            self.logger.debug("Limited time series features will be extracted")
            return X_transformed
        
        features_created = 0
        
        # Extract date features
        if self.date_features:
            date_start = datetime.now()
            self.logger.debug("Extracting date features from index")
            
            X_transformed['day_of_week'] = X_transformed.index.dayofweek
            X_transformed['day_of_month'] = X_transformed.index.day
            X_transformed['day_of_year'] = X_transformed.index.dayofyear
            X_transformed['month'] = X_transformed.index.month
            X_transformed['quarter'] = X_transformed.index.quarter
            X_transformed['year'] = X_transformed.index.year
            X_transformed['is_weekend'] = X_transformed.index.dayofweek >= 5
            
            date_features_count = 7
            features_created += date_features_count
            
            date_time = (datetime.now() - date_start).total_seconds()
            self.logger.debug(f"Created {date_features_count} date features in {date_time:.3f}s")
        
        # Extract cyclic features
        if self.cyclic_features:
            cycle_start = datetime.now()
            self.logger.debug("Extracting cyclical features from index")
            
            X_transformed['day_of_week_sin'] = np.sin(2 * np.pi * X_transformed.index.dayofweek / 7)
            X_transformed['day_of_week_cos'] = np.cos(2 * np.pi * X_transformed.index.dayofweek / 7)
            X_transformed['month_sin'] = np.sin(2 * np.pi * X_transformed.index.month / 12)
            X_transformed['month_cos'] = np.cos(2 * np.pi * X_transformed.index.month / 12)
            
            # Check if time components exist
            has_time = False
            try:
                if (X_transformed.index.hour > 0).any():
                    has_time = True
                    X_transformed['hour_sin'] = np.sin(2 * np.pi * X_transformed.index.hour / 24)
                    X_transformed['hour_cos'] = np.cos(2 * np.pi * X_transformed.index.hour / 24)
                    self.logger.debug("Added hour cyclical features")
            except:
                pass
            
            cycle_features_count = 4 + (2 if has_time else 0)
            features_created += cycle_features_count
            
            cycle_time = (datetime.now() - cycle_start).total_seconds()
            self.logger.debug(f"Created {cycle_features_count} cyclical features in {cycle_time:.3f}s")
        
        # For lag/rolling/
        # For lag/rolling/diff features, we need numeric columns
        numeric_cols = X_transformed.select_dtypes(include=np.number).columns.tolist()
        
        if not numeric_cols:
            self.logger.warning("No numeric columns found for lag/rolling/diff features")
        else:
            self.logger.debug(f"Found {len(numeric_cols)} numeric columns for time series features")
        
        # Extract lag features
        if self.lag_features and numeric_cols:
            lag_start = datetime.now()
            self.logger.debug(f"Creating lag features with orders {self.lag_orders}")
            
            lag_features_count = 0
            for col in numeric_cols:
                col_lag_count = 0
                for lag in self.lag_orders:
                    feature_name = f'{col}_lag_{lag}'
                    X_transformed[feature_name] = X_transformed[col].shift(lag)
                    col_lag_count += 1
                
                self.logger.debug(f"Created {col_lag_count} lag features for column '{col}'")
                lag_features_count += col_lag_count
            
            features_created += lag_features_count
            lag_time = (datetime.now() - lag_start).total_seconds()
            self.logger.debug(f"Created total of {lag_features_count} lag features in {lag_time:.3f}s")
            
            # Log missing value info after lag creation
            null_counts = X_transformed.isna().sum()
            cols_with_nulls = null_counts[null_counts > 0]
            if not cols_with_nulls.empty:
                self.logger.warning(f"Lag features created {cols_with_nulls.sum()} null values across {len(cols_with_nulls)} columns")
                self.logger.debug(f"Columns with nulls after lag creation: {cols_with_nulls.to_dict()}")
        
        # Extract rolling window features
        if self.rolling_features and numeric_cols:
            rolling_start = datetime.now()
            self.logger.debug(f"Creating rolling window features with windows {self.rolling_windows} and functions {self.rolling_functions}")
            
            rolling_features_count = 0
            for col in numeric_cols:
                col_rolling_count = 0
                for window in self.rolling_windows:
                    rolling_obj = X_transformed[col].rolling(window)
                    
                    for func in self.rolling_functions:
                        if hasattr(rolling_obj, func):
                            feature_name = f'{col}_rolling_{window}_{func}'
                            X_transformed[feature_name] = getattr(rolling_obj, func)()
                            col_rolling_count += 1
                        else:
                            self.logger.warning(f"Rolling window function '{func}' not available")
                
                self.logger.debug(f"Created {col_rolling_count} rolling features for column '{col}'")
                rolling_features_count += col_rolling_count
            
            features_created += rolling_features_count
            rolling_time = (datetime.now() - rolling_start).total_seconds()
            self.logger.debug(f"Created total of {rolling_features_count} rolling window features in {rolling_time:.3f}s")
            
            # Log missing value info after rolling window creation
            null_counts = X_transformed.isna().sum()
            cols_with_nulls = null_counts[null_counts > 0]
            if not cols_with_nulls.empty:
                self.logger.warning(f"Rolling features created {cols_with_nulls.sum()} null values across {len(cols_with_nulls)} columns")
        
        # Extract differencing features
        if self.diff_features and numeric_cols:
            diff_start = datetime.now()
            self.logger.debug(f"Creating differencing features with orders {self.diff_orders}")
            
            diff_features_count = 0
            for col in numeric_cols:
                col_diff_count = 0
                for order in self.diff_orders:
                    feature_name = f'{col}_diff_{order}'
                    X_transformed[feature_name] = X_transformed[col].diff(order)
                    col_diff_count += 1
                
                self.logger.debug(f"Created {col_diff_count} differencing features for column '{col}'")
                diff_features_count += col_diff_count
            
            features_created += diff_features_count
            diff_time = (datetime.now() - diff_start).total_seconds()
            self.logger.debug(f"Created total of {diff_features_count} differencing features in {diff_time:.3f}s")
        
        # Handle missing values created by lagging/rolling/differencing
        null_counts = X_transformed.isna().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            self.logger.warning(f"Time series features created {cols_with_nulls.sum()} null values that will need handling")
            
            # Log percentage of nulls in each affected column
            null_pcts = (null_counts / len(X_transformed) * 100)[null_counts > 0]
            high_null_cols = null_pcts[null_pcts > 10]  # Columns with >10% nulls
            
            if not high_null_cols.empty:
                self.logger.warning(f"Columns with high null percentages (>10%): {high_null_cols.to_dict()}")
        
        transform_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"TimeSeriesFeatureExtractor created {features_created} features in {transform_time:.3f}s")
        self.logger.debug(f"Output shape: {X_transformed.shape}")
        
        return X_transformed

def run_default_ml_pipeline(
    df: pd.DataFrame,
    target: str,
    model_id: str,
    output_dir: str,  # This should now be the versioned directory
    config: dict,
    **kwargs
) -> tuple:
    """
    Default ML pipeline implementation with direct output to version directory.
    
    Args:
        df: Input DataFrame
        target: Target column name
        model_id: Model identifier
        output_dir: Versioned output directory path
        config: Configuration dictionary
        **kwargs: Additional arguments
        
    Returns:
        tuple: (best_model, results_df, model_path, metadata_path)
    """
    import os
    import logging
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import pickle
    import json
    
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.info("Running default ML pipeline flow")
    start_time = datetime.now()
    
    # Extract additional parameters from kwargs
    version = kwargs.get('version', 'v1')
    time_col = kwargs.get('time_col', None)
    forecast_horizon = kwargs.get('forecast_horizon', 7)
    data_path = kwargs.get('data_path', None)
    config_path = kwargs.get('config_path', None)
    
    
    logger.info(f"Using versioned output directory: {output_dir}")
    
    # Define final output paths directly in the versioned directory
    model_path = os.path.join(output_dir, f"{model_id}.pkl")
    metadata_path = os.path.join(output_dir, "metadata.json")
    
    # Initialize metadata
    metadata = initialize_metadata()
    metadata["model_id"] = model_id
    metadata["version"] = version
    metadata["data"] = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()}
    }
    
    try:
        # Detect problem type if target exists
        if target is not None:
            problem_type = detect_problem_type(df, target, config)
            logger.info(f"Detected problem type: {problem_type}")
            metadata["parameters"]["problem_type"] = problem_type
        else:
            problem_type = 'clustering'
            logger.info("No target column provided. Using clustering pipeline.")
            metadata["parameters"]["problem_type"] = problem_type
        
        # Initialize the appropriate pipeline based on problem type
        pipeline_args = {
            'df': df,
            'data_path': data_path,
            'target': target,
            'model_id': model_id,
            'output_dir': output_dir,  # Direct to versioned directory
            'config_path': config_path
        }
        
        logger.info(f"Initializing {problem_type} pipeline")
        
        if problem_type == 'regression':
            from custom_ml.models.regression import RegressionPipeline
            pipeline = RegressionPipeline(**pipeline_args)
            
        elif problem_type == 'classification':
            from custom_ml.models.classification import ClassificationPipeline
            pipeline = ClassificationPipeline(**pipeline_args)
            
        elif problem_type == 'clustering':
            from custom_ml.models.cluster import ClusteringPipeline
            pipeline = ClusteringPipeline(**pipeline_args)
            
        elif problem_type == 'time_series':
            from custom_ml.models.time_series import TimeSeriesPipeline
            pipeline = TimeSeriesPipeline(
                **pipeline_args,
                time_col=time_col,
                forecast_horizon=forecast_horizon
            )
        else:
            logger.warning(f"Problem type '{problem_type}' not fully implemented yet")
            logger.info("Defaulting to Regression Pipeline")
            from models.regression import RegressionPipeline
            pipeline = RegressionPipeline(**pipeline_args)
        
        # Run the pipeline
        logger.info(f"Running {problem_type} pipeline...")
        best_model, results = pipeline.run_pipeline()
        
        # Update metadata with pipeline metadata
        if hasattr(pipeline, 'metadata'):
            # Combine with our metadata
            for key, value in pipeline.metadata.items():
                metadata[key] = value
        
        # Save the model directly to the version directory
        if best_model is not None:
            logger.info(f"Saving best model to {model_path}")
            try:
                # Create a complete pipeline with preprocessor if available
                if hasattr(pipeline, 'preprocessor') and pipeline.preprocessor is not None:
                    from sklearn.pipeline import Pipeline as SklearnPipeline
                    complete_model = SklearnPipeline([
                        ('preprocessor', pipeline.preprocessor),
                        ('model', best_model)
                    ])
                    
                    with open(model_path, 'wb') as f:
                        pickle.dump(complete_model, f)
                else:
                    # Just save the model
                    with open(model_path, 'wb') as f:
                        pickle.dump(best_model, f)
                
                logger.info(f"Model saved to {model_path}")
            except Exception as e:
                logger.error(f"Error saving model: {str(e)}")
        
        # Update paths in metadata
        if 'best_model' in metadata and 'filename' in metadata['best_model']:
            metadata['best_model']['filename'] = model_path
        
        # Add runtime information
        metadata["runtime_seconds"] = (datetime.now() - start_time).total_seconds()
        metadata["status"] = "completed"
        metadata["output_dir"] = output_dir
        metadata["end_time"] = datetime.now().isoformat()
        
        # Save metadata directly to the version directory
        with open(metadata_path, 'w') as f:
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    return super(NpEncoder, self).default(obj)
                
            json.dump(metadata, f, indent=2, cls=NpEncoder)
        
        logger.info(f"Metadata saved to {metadata_path}")
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Default ML pipeline completed in {total_time:.2f} seconds")
        
        # At the end, before returning, add validation:
        logger.info("Validating metadata before completing pipeline")
        metadata_path = os.path.join(output_dir, "metadata.json")
        
        return best_model, results, model_path, metadata_path
        
    except Exception as e:
            
        total_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error in default pipeline after {total_time:.2f} seconds: {str(e)}", exc_info=True)
        
        raise

def run_custom_ml_flow(args, config, df, input_dir, version_dir, model_id):
    """
    Run the custom ML pipeline flow, directly using the version directory.

    Args:
        args: Command line arguments
        config: Configuration dictionary
        df: Input DataFrame
        input_dir: Input directory containing data files
        version_dir: Versioned output directory for this run
        model_id: Model identifier

    Returns:
        tuple: (best_model, results, model_path, metadata_path)
    """
    logger.info("Running custom ML flow...")
    start_time = datetime.now()

        
    # Output file paths
    model_path = os.path.join(version_dir, f"{model_id}.pkl")
    metadata_path = os.path.join(version_dir, "metadata.json")
    
    # Log input parameters
    logger.debug(f"Custom ML flow parameters - model_id: {args.model_id}")
    logger.debug(f"DataFrame shape: {df.shape} - {df.shape[0]:,} rows, {df.shape[1]:,} columns")
    logger.debug(f"Input directory: {input_dir}")
    logger.debug(f"Output directory: {version_dir}")
    
    # Log custom configuration
    custom_config = config.get('common', {}).get('custom_ml_model', {})
    function_path = custom_config.get('function_path', '')
    function_name = custom_config.get('function_name', 'run_custom_pipeline')
    
    # Try to get the latest version of the custom code
    if function_path:
        try:
            # Update path to latest version if model_id matches
            path_parts = function_path.split('/')
            if 'input' in path_parts and any(p.startswith('v') for p in path_parts):
                for i, part in enumerate(path_parts):
                    if part == 'input' and i+1 < len(path_parts):
                        possible_model_id = path_parts[i+1]
                        # If model_id matches args.model_id, use it
                        if possible_model_id == args.model_id:
                            logger.debug(f"Extracted model_id from path: {possible_model_id}")
                            # Extract base_path (everything up to input/model_id)
                            base_path_parts = path_parts[:path_parts.index('input')+2]
                            base_path = '/'.join(base_path_parts)
                            # Get latest version
                            latest_path = get_latest_custom_code_path(args.model_id, base_path)
                            if latest_path:
                                function_path = latest_path
                                logger.info(f"Using latest versioned custom code: {function_path}")
                            break
        except Exception as e:
            logger.warning(f"Error parsing function path for versioning: {str(e)}")
    
    # If no path or couldn't extract version, try to get the latest version by model_id
    if not function_path or not os.path.exists(function_path):
        logger.info(f"Searching for latest custom code version for model_id: {args.model_id}")
        latest_path = get_latest_custom_code_path(args.model_id)
        if latest_path:
            function_path = latest_path
            logger.info(f"Found latest versioned custom code: {function_path}")

    logger.debug(f"Custom function configuration:")
    logger.debug(f"  - Function path: {function_path}")
    logger.debug(f"  - Function name: {function_name}")
    
    # Check if the module file exists
    if not os.path.exists(function_path):
        logger.warning(f"Custom function module path does not exist: {function_path}")
    
    if not function_path:
        error_msg = "Custom ML model is enabled but no function_path is specified in config"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Load the custom function
    logger.debug(f"Loading custom function '{function_name}' from {function_path}")
    load_start = datetime.now()
    
    custom_function = load_custom_function(function_path, function_name)
    load_time = (datetime.now() - load_start).total_seconds()
    
    if custom_function is None:
        error_msg = f"Failed to load custom function '{function_name}' from {function_path}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info(f"Successfully loaded custom function in {load_time:.2f} seconds")

    # Prepare arguments for custom function - now with direct paths
    custom_args = {
        'df': df,
        'target': args.target,
        'model_id': args.model_id,
        'output_dir': version_dir,  # Pass the version directory directly
        'config': config,
        'time_col': args.time_col,
        'version': args.version,
        'forecast_horizon': getattr(args, 'forecast_horizon', None)
    }
    
    logger.debug(f"Prepared arguments for custom function: {custom_args.keys()}")
    
    try:
        # Call the custom function
        logger.info(f"Calling custom function '{function_name}'")
        fn_start_time = datetime.now()
        
        # Log system resources before running custom function
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            logger.debug(f"System resources before custom function: CPU {cpu_percent}%, RAM {memory_info.percent}% (Available: {memory_info.available / (1024**3):.2f} GB)")
        except ImportError:
            pass

        # The custom function should return (best_model, results, model_path, metadata_path)
        result = custom_function(**custom_args)
        
        fn_time = (datetime.now() - fn_start_time).total_seconds()
        logger.info(f"Custom function executed in {fn_time:.2f} seconds")
        
        # Log system resources after running
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            logger.debug(f"System resources after custom function: CPU {cpu_percent}%, RAM {memory_info.percent}% (Available: {memory_info.available / (1024**3):.2f} GB)")
        except ImportError:
            pass

        # Check return type and unpack
        if isinstance(result, tuple) and len(result) >= 2:
            logger.debug(f"Custom function returned a tuple with {len(result)} elements")
            best_model, results = result[0], result[1]

            # Get paths if provided, otherwise use defaults
            custom_model_path = result[2] if len(result) > 2 and result[2] else model_path
            custom_metadata_path = result[3] if len(result) > 3 and result[3] else metadata_path
            
            # If custom paths are different from our default paths, update them
            if custom_model_path != model_path and os.path.exists(custom_model_path):
                model_path = custom_model_path
            
            if custom_metadata_path != metadata_path and os.path.exists(custom_metadata_path):
                metadata_path = custom_metadata_path
            
            logger.debug(f"Final model path: {model_path}")
            logger.debug(f"Final metadata path: {metadata_path}")
            
            # Update metadata if it exists
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Add timestamp if not present
                    if 'timestamp' not in metadata:
                        metadata['timestamp'] = datetime.now().isoformat()
                    
                    # Add runtime information
                    if 'runtime_seconds' not in metadata:
                        metadata['runtime_seconds'] = fn_time
                        
                    # Write back updated metadata
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                except Exception as e:
                    logger.warning(f"Error updating metadata: {str(e)}")
        else:
            logger.warning(f"Custom function did not return expected tuple format. Got: {type(result)}")
            
            # Try to extract useful information from the result
            if result is not None:
                best_model = result
                results = None
                
                # Save model if not already saved
                if not os.path.exists(model_path) and best_model is not None:
                    try:
                        with open(model_path, 'wb') as f:
                            pickle.dump(best_model, f)
                        logger.info(f"Saved model to {model_path}")
                    except Exception as e:
                        logger.error(f"Error saving model: {str(e)}")
                
                # Create basic metadata if not present
                if not os.path.exists(metadata_path):
                    metadata = {
                        'model_id': model_id,
                        'timestamp': datetime.now().isoformat(),
                        'runtime_seconds': fn_time,
                        'version': args.version,
                        'status': 'completed',
                        'custom_ml_model': True
                    }
                    
                    try:
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        logger.info(f"Created basic metadata at {metadata_path}")
                    except Exception as e:
                        logger.error(f"Error creating metadata: {str(e)}")
        
        if isinstance(result, tuple) and len(result) >= 4:
            best_model, results, model_path, metadata_path = result
            logger.info(f"Custom function completed successfully")
            logger.info(f"Model saved to: {model_path}")
            logger.info(f"Metadata saved to: {metadata_path}")
        else:
            logger.error(f"Invalid result from custom function. Expected a tuple with at least 4 items.")
            raise ValueError(f"Invalid result from custom function. Expected a tuple with at least 4 items.")
        
        # Validate the metadata from the custom flow
        try:
            logger.info(f"Validating custom model metadata: {metadata_path}")
            
            # Load the metadata
            with open(metadata_path, 'r') as f:
                metadata_content = json.load(f)
            
            # Create validator and validate
            validator = MetadataValidator(strict_mode=False)
            is_valid, messages = validator.validate_metadata(metadata_content)
            
            # Update metadata with validation results
            validation_result = {
                'metadata_validated': True,
                'validation_timestamp': datetime.now().isoformat(),
                'validation_passed': is_valid,
                'validation_issues': messages
            }
            
            metadata_content['validation'] = validation_result
            
            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata_content, f, indent=2)
            
            # Log validation results
            if is_valid:
                logger.info("✅ Custom model metadata validation PASSED")
                print("\n✅ Metadata validation PASSED - Model is ready for testing\n")
            else:
                logger.warning(f"⚠️ Custom model metadata validation found {len(messages)} issues")
                for msg in messages:
                    logger.warning(f"  - {msg}")
                
                # Print warning to console for visibility
                print(f"\n⚠️ WARNING: Custom model metadata has {len(messages)} validation issues")
                print("The model may not work correctly with the testing pipeline.")
                print(f"See validation issues in metadata: {metadata_path}\n")
                
        except Exception as e:
            logger.warning(f"Error during metadata validation (non-fatal): {str(e)}", exc_info=True)
        
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Custom ML flow completed in {total_time:.2f} seconds")



        
        return best_model, results, model_path, metadata_path

    except Exception as e:
        
            
        error_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error in custom ML flow after {error_time:.2f} seconds: {str(e)}", exc_info=True)
        
        # Try to log more context about the error
        try:
            import traceback
            err_trace = traceback.format_exc()
            logger.debug(f"Error traceback:\n{err_trace}")
        except Exception:
            pass
        
        logger.error(f"Custom ML flow failed after {error_time:.2f} seconds")
        raise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Custom ML Pipeline')
    
    # Required arguments
    parser.add_argument('--model_id', type=str, required=True, help='Model ID (e.g. 123456789)')
    
    # Make target optional
    parser.add_argument('--target', type=str, required=False, default=None, 
                        help='Name of the target column (not required for clustering)')
    
    # Add time column for time series
    parser.add_argument('--time_col', type=str, required=False, default=None,
                        help='Name of time/date column for time series forecasting')
    
    # Time series specific arguments
    parser.add_argument('--forecast_horizon', type=int, default=7,
                        help='Number of periods to forecast (time series only)')
    
    # Optional arguments
    parser.add_argument('--version', type=str, default='v1', help='Version (default: v1)')
    parser.add_argument('--config_path', type=str, default='config/config.yaml', 
                        help='Path to configuration file')
    
    # Add verbosity control
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--quiet', action='store_true', help='Suppress info messages')
     
    return parser.parse_args()

#########################################
# Directory & Versioning Management
#########################################

def setup_basic_logging(verbose=False, quiet=False):
    """Configure basic logging based on verbosity."""
    log_level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(level=log_level, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.debug(f"Basic logging configured with level: {log_level}")

def prepare_directories(model_id, version):
    """
    Prepare input and output directories based on model_id and version.
    
    Args:
        model_id: Model identifier
        version: Version string
        
    Returns:
        tuple: (input_dir, output_dir, data_path)
    """
    # Base directories
    base_dir = os.path.join("data")
    input_dir = os.path.join(base_dir, "training", "input", model_id, version)
    output_dir = os.path.join(base_dir, "training", "output", model_id)
    
    # Log directory structure
    logger.debug(f"Directory structure:")
    logger.debug(f"  - Base directory: {base_dir}")
    logger.debug(f"  - Input directory: {input_dir}")
    logger.debug(f"  - Output directory: {output_dir}")
    
    # Input data path
    data_path = os.path.join(input_dir, "input_data.csv")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        logger.warning(f"Input directory does not exist: {input_dir}")
        # Try to create the directory
        try:
            os.makedirs(input_dir, exist_ok=True)
            logger.info(f"Created input directory: {input_dir}")
        except Exception as e:
            logger.error(f"Failed to create input directory: {str(e)}")
    
    return input_dir, output_dir, data_path

def load_input_data(data_path, input_dir):
    """
    Load the input data file, trying alternative formats if necessary.
    
    Args:
        data_path: Path to the primary data file
        input_dir: Input directory to check for alternative files
        
    Returns:
        DataFrame: Loaded data
    """
    # Validate input file exists
    if not os.path.exists(data_path):
        logger.error(f"Input file not found: {data_path}")
        
        # Look for alternative file formats
        alt_formats = ['.xlsx', '.parquet', '.json']
        found_alt = False
        
        for fmt in alt_formats:
            alt_path = data_path.replace('.csv', fmt)
            if os.path.exists(alt_path):
                logger.info(f"Found alternative format input file: {alt_path}")
                data_path = alt_path
                found_alt = True
                break
        
        if not found_alt:
            # Log available files in the input directory
            if os.path.exists(input_dir):
                files = os.listdir(input_dir)
                if files:
                    logger.info(f"Files found in input directory: {files}")
                else:
                    logger.info(f"Input directory exists but is empty: {input_dir}")
            raise FileNotFoundError(f"No valid input data file found in {input_dir}")

    # Log input file information
    file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
    logger.info(f"Using input data: {data_path} ({file_size_mb:.2f} MB)")
    
    # Load the data based on file extension
    file_ext = os.path.splitext(data_path)[1].lower()
    try:
        logger.debug(f"Loading data from {data_path} (format: {file_ext})")
        
        if file_ext == '.csv':
            logger.debug("Reading CSV file")
            # First peek at the CSV to get column count
            with open(data_path, 'r') as f:
                first_line = f.readline().strip()
                approx_columns = len(first_line.split(','))
                logger.debug(f"CSV preview: approximately {approx_columns} columns")
            
            df = pd.read_csv(data_path)
            logger.debug(f"CSV loaded: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
        elif file_ext in ['.xls', '.xlsx']:
            logger.debug("Reading Excel file")
            # Check sheet names
            import openpyxl
            wb = openpyxl.load_workbook(data_path, read_only=True)
            sheet_names = wb.sheetnames
            logger.debug(f"Excel file contains sheets: {sheet_names}")
            wb.close()
            
            df = pd.read_excel(data_path)
            logger.debug(f"Excel loaded: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
        elif file_ext == '.json':
            logger.debug("Reading JSON file")
            df = pd.read_json(data_path)
            logger.debug(f"JSON loaded: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
        elif file_ext == '.parquet':
            logger.debug("Reading Parquet file")
            df = pd.read_parquet(data_path)
            logger.debug(f"Parquet loaded: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
        
        # Log data summary
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]:,} columns, {memory_usage_mb:.2f} MB")
        perform_basic_data_quality_checks(df)
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to load data from {data_path}: {str(e)}")

def perform_basic_data_quality_checks(df):
    """Perform basic data quality checks on the loaded DataFrame."""
    # Basic data quality checks
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        missing_pct = (missing_values / df.size) * 100
        logger.warning(f"Found {missing_values:,} missing values ({missing_pct:.2f}% of all cells)")
        
        # Columns with most missing values
        cols_with_missing = df.columns[df.isnull().any()].tolist()
        missing_by_col = df[cols_with_missing].isnull().sum().sort_values(ascending=False)
        logger.debug(f"Top columns with missing values: {dict(missing_by_col.head(5))}")
    
    # Data type summary
    dtype_counts = df.dtypes.value_counts().to_dict()
    logger.debug(f"Column data types: {dtype_counts}")

def print_pipeline_results(model_path, metadata_path, model_id, output_dir, best_model, results, total_time):
    """Print final results of the pipeline run."""
    logger.info("\n" + "="*50)
    logger.info("Pipeline Results")
    logger.info("="*50)
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Output directory: {output_dir}")
    
    # Check if files exist
    model_exists = os.path.exists(model_path)
    metadata_exists = os.path.exists(metadata_path)
    
    if model_exists:
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"Model saved to: {model_path} ({model_size_mb:.2f} MB)")
    else:
        logger.warning(f"Model file not found at expected path: {model_path}")
    
    if metadata_exists:
        metadata_size_kb = os.path.getsize(metadata_path) / 1024
        logger.info(f"Metadata saved to: {metadata_path} ({metadata_size_kb:.2f} KB)")
        
        # Try to extract key information from metadata
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Get model information
            if 'best_model' in metadata:
                best_model_info = metadata['best_model']
                best_model_name = best_model_info.get('name', 'Unknown')
                
                # Extract metrics
                metrics = {k: v for k, v in best_model_info.items() if 'metric' in k.lower()}
                
                logger.info(f"Best model: {best_model_name}")
                logger.info(f"Model metrics: {metrics}")
                
                # Runtime information
                if 'runtime_seconds' in metadata:
                    runtime = metadata['runtime_seconds']
                    logger.info(f"Pipeline runtime: {runtime:.2f} seconds")
        except Exception as e:
            logger.debug(f"Could not extract information from metadata: {str(e)}")
    else:
        logger.warning(f"Metadata file not found at expected path: {metadata_path}")

    if results is not None and not results.empty:
        logger.info("\nModel Performance Summary:")
        num_models = min(3, len(results))
        
        # Display top models based on results
        try:
            logger.info(f"Top {num_models} models:")
            for i, (idx, row) in enumerate(results.head(num_models).iterrows()):
                model_name = row.get('model_name', idx)
                metrics = {col: row[col] for col in results.columns if col != 'model_name'}
                logger.info(f"  {i+1}. {model_name}: {metrics}")
        except Exception as e:
            logger.debug(f"Could not display model performance: {str(e)}")
        
    logger.info(f"\nPipeline completed successfully in {total_time:.2f} seconds!")

def get_latest_custom_code_path(model_id, base_path=None):
    """
    Get the path to the latest version of custom_code.py for a given model_id.
    
    Args:
        model_id: Model identifier
        base_path: Base path for custom code (defaults to standard location)
        
    Returns:
        str: Path to the latest version of custom_code.py
    """
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Define default base path if not provided
    if base_path is None:
        base_path = os.path.join("custom_ml_data", "custom_code", "input", model_id)
    
    logger.debug(f"Searching for custom code versions in: {base_path}")
    
    # Check if the directory exists
    if not os.path.exists(base_path):
        logger.warning(f"Custom code base path does not exist: {base_path}")
        # Try alternate paths
        alt_paths = [
            os.path.join("/home/devendra_yadav/custom_ml_model/custom_ml_data/custom_code/input", model_id),
            os.path.join("custom_code", "input", model_id),
            os.path.join("input", model_id)
        ]
        
        for path in alt_paths:
            if os.path.exists(path):
                base_path = path
                logger.info(f"Found alternate custom code path: {base_path}")
                break
        else:
            logger.error(f"No custom code directory found for model_id: {model_id}")
            return None
    
    # Get all version directories (they should be named v1, v2, etc.)
    version_dirs = []
    try:
        for item in os.listdir(base_path):
            full_path = os.path.join(base_path, item)
            
            if os.path.isdir(full_path) and item.startswith('v'):
                try:
                    # Extract version number from directory name (v1, v2, etc.)
                    version_num = int(item[1:])
                    version_dirs.append((item, version_num, full_path))
                    logger.debug(f"Found version directory: {item} (version {version_num})")
                except ValueError:
                    # Skip directories that don't follow the vN pattern
                    logger.debug(f"Skipping directory that doesn't match version pattern: {item}")
                    continue
    except Exception as e:
        logger.error(f"Error scanning directory {base_path}: {str(e)}")
        return None
    
    # If no version directories found
    if not version_dirs:
        logger.warning(f"No version directories found for model_id: {model_id}")
        
        # Check for custom_code.py directly in the base path as fallback
        fallback_path = os.path.join(base_path, "custom_code.py")
        if os.path.exists(fallback_path):
            logger.info(f"Found custom code at fallback location: {fallback_path}")
            return fallback_path
        
        return None
    
    # Sort by version number to find the highest
    version_dirs.sort(key=lambda x: x[1], reverse=True)
    latest_version = version_dirs[0]
    logger.info(f"Latest custom code version: {latest_version[0]} (version {latest_version[1]})")
    
    # Check for custom_code.py in the latest version directory
    custom_code_path = os.path.join(latest_version[2], "custom_code.py")
    if os.path.exists(custom_code_path):
        logger.info(f"Found custom code at: {custom_code_path}")
        return custom_code_path
    else:
        logger.warning(f"custom_code.py not found in latest version directory: {latest_version[2]}")
        
        # Try to find it in other version directories in descending order
        for _, _, dir_path in version_dirs[1:]:
            alt_path = os.path.join(dir_path, "custom_code.py")
            if os.path.exists(alt_path):
                logger.info(f"Found custom code in alternate version: {alt_path}")
                return alt_path
        
        logger.error(f"No custom_code.py found in any version directory for model_id: {model_id}")
        return None

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
    logger.debug(f"Getting next version directory for model_id={model_id} in {base_output_dir}")
    start_time = datetime.now()
    
    # Create model directory if it doesn't exist (should be base_output_dir)
    model_dir = base_output_dir
    if not os.path.exists(model_dir):
        try:
            os.makedirs(model_dir, exist_ok=True)
            logger.debug(f"Created new model directory: {model_dir}")
            dir_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Created new model directory in {dir_time:.3f}s, starting with version v1")
            return os.path.join(model_dir, "v1"), 1
        except Exception as e:
            logger.error(f"Failed to create model directory {model_dir}: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            # Create a fallback directory in the current working directory
            fallback_dir = os.path.join(os.getcwd(), "fallback_output", model_id)
            os.makedirs(fallback_dir, exist_ok=True)
            logger.warning(f"Using fallback directory: {fallback_dir}")
            return os.path.join(fallback_dir, "v1"), 1
    
    # Get existing version directories
    version_dirs = []
    try:
        logger.debug(f"Scanning {model_dir} for existing version directories")
        
        for item in os.listdir(model_dir):
            full_path = os.path.join(model_dir, item)
            
            if os.path.isdir(full_path) and item.startswith('v'):
                try:
                    version_num = int(item[1:])  # Extract number part from 'v1', 'v2', etc.
                    version_dirs.append((item, version_num))
                    logger.debug(f"Found existing version directory: {item} (version {version_num})")
                except ValueError:
                    # Skip directories that don't follow the vN pattern
                    logger.debug(f"Skipping directory that doesn't match version pattern: {item}")
                    continue
        
        # Check for empty version directories
        for version_name, version_num in version_dirs:
            version_path = os.path.join(model_dir, version_name)
            if not os.listdir(version_path):
                logger.warning(f"Empty version directory found: {version_path}")
    
    except Exception as e:
        logger.error(f"Error scanning directory {model_dir}: {str(e)}")
        logger.debug(f"Error details: {traceback.format_exc()}")
        # Return first version in case of error
        return os.path.join(model_dir, "v1"), 1
    
    # Sort by version number
    version_dirs.sort(key=lambda x: x[1])
    logger.debug(f"Found {len(version_dirs)} version directories: {[v[0] for v in version_dirs]}")
    
    # If no valid versions exist, start with v1
    if not version_dirs:
        logger.info("No existing version directories found, starting with version v1")
        return os.path.join(model_dir, "v1"), 1
    
    # Get the next version number
    next_version_num = version_dirs[-1][1] + 1
    next_version = f"v{next_version_num}"
    logger.debug(f"Last version was v{version_dirs[-1][1]}, next version will be {next_version}")
    
    # Check if we need to delete old versions
    if len(version_dirs) >= max_versions:
        # Keep the newest (max_versions-1) to make room for the new one
        versions_to_delete = version_dirs[:-(max_versions-1)]
        logger.warning(f"Maximum versions ({max_versions}) reached, will delete {len(versions_to_delete)} oldest version(s)")
        
        for version, version_num in versions_to_delete:
            dir_to_delete = os.path.join(model_dir, version)
            
            try:
                # Check size before deletion
                dir_size = 0
                for dirpath, dirnames, filenames in os.walk(dir_to_delete):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        dir_size += os.path.getsize(fp)
                
                dir_size_mb = dir_size / (1024 * 1024)
                
                # Log content before deletion
                file_count = sum([len(files) for _, _, files in os.walk(dir_to_delete)])
                logger.debug(f"Directory to delete contains {file_count} files, total size: {dir_size_mb:.2f} MB")
                
                # Delete directory
                import shutil
                delete_start = datetime.now()
                shutil.rmtree(dir_to_delete)
                delete_time = (datetime.now() - delete_start).total_seconds()
                
                logger.info(f"Deleted old version directory: {dir_to_delete} ({dir_size_mb:.2f} MB) in {delete_time:.3f}s")
            except Exception as e:
                logger.error(f"Error deleting {dir_to_delete}: {str(e)}")
                logger.debug(f"Error details: {traceback.format_exc()}")
    
    # Create the new version directory
    next_version_dir = os.path.join(model_dir, next_version)
    try:
        os.makedirs(next_version_dir, exist_ok=True)
        logger.debug(f"Created new version directory: {next_version_dir}")
    except Exception as e:
        logger.error(f"Failed to create version directory {next_version_dir}: {str(e)}")
        # Use model_dir as fallback
        next_version_dir = os.path.join(model_dir, "fallback_" + next_version)
        os.makedirs(next_version_dir, exist_ok=True)
        logger.warning(f"Using fallback version directory: {next_version_dir}")
    
    total_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Generated next version v{next_version_num} at {next_version_dir} in {total_time:.3f}s")
    
    return next_version_dir, next_version_num