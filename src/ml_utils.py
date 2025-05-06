import os
import json
import yaml
import logging
import numpy as np
import pandas as pd
import traceback
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any, Union, Tuple
from sklearn.base import BaseEstimator, TransformerMixin

# Initialize logger
logger = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)

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
        
        # Log important configuration sections if they exist
        if 'common' in config:
            common_keys = list(config['common'].keys())
            logger.debug(f"Common configuration contains: {common_keys}")
        
        # Log problem-specific configurations
        problem_types = ['regression', 'classification', 'time_series', 'clustering']
        for problem in problem_types:
            if problem in config:
                problem_keys = list(config[problem].keys())
                logger.debug(f"{problem.capitalize()} configuration contains: {problem_keys}")
        
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