import os
import sys
import argparse
import logging
import json
import shutil
from datetime import datetime
import pandas as pd
import src.ml_utils as ml_utils
from models.regression import RegressionPipeline
from models.classification import ClassificationPipeline
from models.cluster import ClusteringPipeline
from models.time_series import TimeSeriesPipeline
logger = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)


def run_default_ml_pipeline(
    df: pd.DataFrame,
    target: str,
    model_id: str,
    output_dir: str,
    config: dict,
    **kwargs
) -> tuple:
    """
    Default ML pipeline implementation - wraps the existing pipeline classes.
    
    Args:
        df: Input DataFrame
        target: Target column name
        model_id: Model identifier
        output_dir: Output directory path
        config: Configuration dictionary
        **kwargs: Additional arguments
        
    Returns:
        tuple: (best_model, results_df, model_path, metadata_path)
    """
    import os
    import logging
    import pandas as pd
    from datetime import datetime
    
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.info("Running default ML pipeline flow")
    start_time = datetime.now()
    
    # Log input parameters
    logger.debug(f"Input parameters - model_id: {model_id}, target: {target}")
    logger.debug(f"DataFrame shape: {df.shape} - {df.shape[0]:,} rows, {df.shape[1]:,} columns")
    logger.debug(f"Output directory: {output_dir}")
    
    # Extract additional parameters from kwargs
    version = kwargs.get('version', 'v1')
    time_col = kwargs.get('time_col', None)
    forecast_horizon = kwargs.get('forecast_horizon', 7)
    
    logger.debug(f"Additional parameters - version: {version}, time_col: {time_col}, forecast_horizon: {forecast_horizon}")
    
    # Log column data types
    dtypes_summary = df.dtypes.value_counts().to_dict()
    logger.debug(f"DataFrame column types: {dtypes_summary}")
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        missing_pct = (missing_values / df.size) * 100
        logger.debug(f"Found {missing_values:,} missing values ({missing_pct:.2f}% of all cells)")
        
        # Log columns with highest missing values
        cols_with_missing = df.columns[df.isnull().any()].tolist()
        missing_by_col = df[cols_with_missing].isnull().sum().sort_values(ascending=False)
        logger.debug(f"Top columns with missing values: {dict(missing_by_col.head(5))}")
    else:
        logger.debug("No missing values found in the dataset")
    
    try:
        # Create a temporary file to save the DataFrame if needed
        # This is only needed if your pipelines require a file path rather than a DataFrame
        temp_file = False
        data_path = kwargs.get('data_path', None)
        
        if data_path is None or not os.path.exists(data_path):
            # Create temporary directory if needed
            temp_dir = os.path.join(output_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            data_path = os.path.join(temp_dir, f"temp_data_{model_id}.csv")
            
            logger.debug(f"Creating temporary data file at: {data_path}")
            temp_file_start = datetime.now()
            df.to_csv(data_path, index=False)
            temp_file_time = (datetime.now() - temp_file_start).total_seconds()
            
            file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
            temp_file = True
            logger.info(f"Created temporary data file at {data_path} ({file_size_mb:.2f} MB) in {temp_file_time:.2f} seconds")
        else:
            # Check existing file size
            file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
            logger.debug(f"Using existing data file: {data_path} ({file_size_mb:.2f} MB)")
        
        # Detect problem type if target exists
        if target is not None:
            logger.debug(f"Attempting to detect problem type for target column: '{target}'")
            
            try:
                # Log target column statistics
                if target in df.columns:
                    is_numeric = pd.api.types.is_numeric_dtype(df[target])
                    unique_values = df[target].nunique()
                    unique_pct = (unique_values / len(df)) * 100
                    
                    logger.debug(f"Target column '{target}' statistics:")
                    logger.debug(f"  - Data type: {df[target].dtype}")
                    logger.debug(f"  - Is numeric: {is_numeric}")
                    logger.debug(f"  - Unique values: {unique_values:,} ({unique_pct:.2f}% of rows)")
                    
                    if is_numeric:
                        # Log numeric target stats
                        num_stats = df[target].describe()
                        logger.debug(f"  - Min: {num_stats['min']}, Max: {num_stats['max']}")
                        logger.debug(f"  - Mean: {num_stats['mean']}, Median: {num_stats['50%']}")
                    else:
                        # Log categorical target stats
                        value_counts = df[target].value_counts()
                        top_classes = dict(value_counts.head(3))
                        bottom_classes = dict(value_counts.tail(3))
                        logger.debug(f"  - Most common classes: {top_classes}")
                        logger.debug(f"  - Least common classes: {bottom_classes}")
                else:
                    logger.warning(f"Target column '{target}' not found in DataFrame columns: {df.columns.tolist()}")
            except Exception as e:
                logger.warning(f"Error analyzing target column: {str(e)}")
            
            # Detect problem type
            problem_type_start = datetime.now()
            problem_type = ml_utils.detect_problem_type(df, target, config)
            problem_type_time = (datetime.now() - problem_type_start).total_seconds()
            logger.info(f"Detected problem type: {problem_type} in {problem_type_time:.2f} seconds")
        else:
            problem_type = 'clustering'
            logger.info("No target column provided. Using clustering pipeline.")
        
        # Initialize the appropriate pipeline based on problem type
        pipeline_init_start = datetime.now()
        
        if problem_type == 'regression':
            logger.info("Initializing Regression Pipeline")
            
            # Log common regression metrics for reference
            logger.debug("Common metrics for this problem type: RMSE, MAE, R², MSE")
            
            pipeline = RegressionPipeline(
                df=df,  # Pass DataFrame directly
                data_path=data_path,
                target=target,
                model_id=model_id,
                output_dir=output_dir,
                config_path=kwargs.get('config_path', None)
            )
        elif problem_type == 'classification':
            logger.info("Initializing Classification Pipeline")
            
            # Check class distribution for classification
            if target in df.columns:
                class_counts = df[target].value_counts(normalize=True) * 100
                min_class_pct = class_counts.min()
                max_class_pct = class_counts.max()
                
                if min_class_pct < 10:  # If imbalanced classes
                    logger.warning(f"Class imbalance detected: minority class represents only {min_class_pct:.2f}% of data")
                    logger.debug(f"Class distribution: {dict(class_counts)}")
                
                logger.debug(f"Classification problem with {len(class_counts)} classes")
                logger.debug(f"Class balance ratio (min/max): {min_class_pct:.2f}% / {max_class_pct:.2f}%")
                logger.debug("Common metrics for this problem type: Accuracy, Precision, Recall, F1-score")
            
            pipeline = ClassificationPipeline(
                df=df,
                data_path=data_path,
                target=target,
                model_id=model_id,
                output_dir=output_dir,
                config_path=kwargs.get('config_path', None)
            )
        elif problem_type == 'clustering':
            logger.info("Initializing Clustering Pipeline")
            
            # Log clustering-specific information
            numeric_cols = df.select_dtypes(include=['number']).columns
            logger.debug(f"Clustering with {len(numeric_cols)} numeric features out of {df.shape[1]} total columns")
            logger.debug("Common metrics for this problem type: Silhouette score, Inertia, Davies-Bouldin index")
            
            pipeline = ClusteringPipeline(
                df=df,
                data_path=data_path,
                model_id=model_id,
                output_dir=output_dir,
                config_path=kwargs.get('config_path', None)
            )
        elif problem_type == 'time_series':
            logger.info("Initializing Time Series Pipeline")
            
            # Check time column for time series
            if time_col in df.columns:
                logger.debug(f"Time column '{time_col}' details:")
                logger.debug(f"  - Data type: {df[time_col].dtype}")
                
                if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                    time_range = df[time_col].max() - df[time_col].min()
                    n_periods = len(df[time_col].unique())
                    logger.debug(f"  - Date range: {df[time_col].min()} to {df[time_col].max()} ({time_range})")
                    logger.debug(f"  - Unique time periods: {n_periods}")
                    logger.debug(f"  - Forecast horizon: {forecast_horizon} periods")
                else:
                    logger.warning(f"Time column '{time_col}' is not a datetime type: {df[time_col].dtype}")
            else:
                logger.warning(f"Time column '{time_col}' not found in DataFrame columns: {df.columns.tolist()}")
            
            logger.debug("Common metrics for this problem type: RMSE, MAE, MAPE, AIC, BIC")
            
            pipeline = TimeSeriesPipeline(
                df=df,
                data_path=data_path,
                target=target,
                time_col=time_col,
                model_id=model_id,
                output_dir=output_dir,
                config_path=kwargs.get('config_path', None),
                forecast_horizon=forecast_horizon
            )
        else:
            logger.warning(f"Problem type '{problem_type}' not fully implemented yet")
            logger.info("Defaulting to Regression Pipeline")
            pipeline = RegressionPipeline(
                df=df,
                data_path=data_path,
                target=target,
                model_id=model_id,
                output_dir=output_dir,
                config_path=kwargs.get('config_path', None)
            )
        
        pipeline_init_time = (datetime.now() - pipeline_init_start).total_seconds()
        logger.debug(f"Pipeline initialized in {pipeline_init_time:.2f} seconds")
        
        # Run the pipeline
        logger.info(f"Running {problem_type} pipeline...")
        run_start = datetime.now()
        
        try:
            # Log system resources before running
            try:
                import psutil
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=0.1)
                logger.debug(f"System resources before pipeline: CPU {cpu_percent}%, RAM {memory_info.percent}% (Available: {memory_info.available / (1024**3):.2f} GB)")
            except ImportError:
                logger.debug("psutil not available for resource monitoring")
        
            best_model, results = pipeline.run_pipeline()
            run_time = (datetime.now() - run_start).total_seconds()
            logger.info(f"Pipeline execution completed in {run_time:.2f} seconds")
            
            # Log system resources after running
            try:
                import psutil
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=0.1)
                logger.debug(f"System resources after pipeline: CPU {cpu_percent}%, RAM {memory_info.percent}% (Available: {memory_info.available / (1024**3):.2f} GB)")
            except ImportError:
                pass
        except Exception as e:
            run_time = (datetime.now() - run_start).total_seconds()
            logger.error(f"Pipeline execution failed after {run_time:.2f} seconds: {str(e)}")
            raise
        
        # Extract paths from pipeline
        model_path = os.path.join(pipeline.output_dir, f"{model_id}.pkl")
        metadata_path = os.path.join(pipeline.output_dir, "metadata.json")
        
        # Log model file information
        if os.path.exists(model_path):
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            logger.info(f"Model saved to {model_path} ({model_size_mb:.2f} MB)")
        else:
            logger.warning(f"Model file not found at {model_path}")
            
        # Log metadata file information    
        if os.path.exists(metadata_path):
            metadata_size_kb = os.path.getsize(metadata_path) / 1024
            logger.info(f"Metadata saved to {metadata_path} ({metadata_size_kb:.2f} KB)")
            
            # Try to extract key metrics from metadata
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                if 'best_model' in metadata:
                    best_model_name = metadata['best_model'].get('name', 'Unknown')
                    metrics = {k: v for k, v in metadata['best_model'].items() if 'metric' in k.lower()}
                    logger.info(f"Best model: {best_model_name} with metrics: {metrics}")
            except Exception as e:
                logger.debug(f"Could not extract metrics from metadata: {str(e)}")
        else:
            logger.warning(f"Metadata file not found at {metadata_path}")
        
        # Clean up temporary file if created
        if temp_file and os.path.exists(data_path):
            try:
                logger.debug(f"Removing temporary data file: {data_path}")
                os.remove(data_path)
                logger.info(f"Removed temporary data file {data_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {str(e)}")
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Default ML pipeline completed in {total_time:.2f} seconds")
        
        return best_model, results, model_path, metadata_path
        
    except Exception as e:
        total_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error in default pipeline after {total_time:.2f} seconds: {str(e)}", exc_info=True)
        
        # Try to log more context about the error
        try:
            import traceback
            err_trace = traceback.format_exc()
            logger.debug(f"Error traceback:\n{err_trace}")
            
            # Log system state
            logger.debug("Error context:")
            logger.debug(f"  - Problem type: {problem_type if 'problem_type' in locals() else 'unknown'}")
            logger.debug(f"  - Data path: {data_path if 'data_path' in locals() else 'unknown'}")
            
            if 'df' in locals():
                logger.debug(f"  - DataFrame shape: {df.shape}")
                logger.debug(f"  - DataFrame columns: {df.columns.tolist()}")
            
        except Exception as context_error:
            logger.debug(f"Could not log error context: {str(context_error)}")
        
        raise

def run_custom_ml_flow(args, config, df, input_dir, output_dir):
    """
    Run the custom ML pipeline flow, handling versioning and directory structure.

    Args:
        args: Command line arguments
        config: Configuration dictionary
        df: Input DataFrame
        input_dir: Input directory for data
        output_dir: Base output directory

    Returns:
        tuple: (best_model, results, model_path, metadata_path)
    """
    logger.info("Running custom ML flow...")
    start_time = datetime.now()
    
    # Log input parameters
    logger.debug(f"Custom ML flow parameters - model_id: {args.model_id}")
    logger.debug(f"DataFrame shape: {df.shape} - {df.shape[0]:,} rows, {df.shape[1]:,} columns")
    logger.debug(f"Input directory: {input_dir}")
    logger.debug(f"Output directory: {output_dir}")
    
    # Log custom configuration
    custom_config = config.get('common', {}).get('custom_ml_model', {})
    function_path = custom_config.get('function_path', '')
    function_name = custom_config.get('function_name', 'run_custom_pipeline')
    
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
    logger.debug(f"Attempting to load custom function '{function_name}' from {function_path}")
    load_start = datetime.now()
    
    custom_function = ml_utils.load_custom_function(function_path, function_name)
    load_time = (datetime.now() - load_start).total_seconds()
    
    if custom_function is None:
        error_msg = f"Failed to load custom function '{function_name}' from {function_path}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info(f"Successfully loaded custom function in {load_time:.2f} seconds")

    # Create versioned output directory using existing utility
    dir_start = datetime.now()
    versioned_output_dir, version_num = ml_utils.get_next_version_dir(
        output_dir, args.model_id,
        max_versions=config.get('common', {}).get('output', {}).get('max_versions', 5)
    )
    dir_time = (datetime.now() - dir_start).total_seconds()

    # Ensure versioned directory exists
    os.makedirs(versioned_output_dir, exist_ok=True)
    logger.info(f"Created versioned output directory: {versioned_output_dir} (v{version_num}) in {dir_time:.2f} seconds")

    # Create a temporary working directory for the custom function
    temp_dir = os.path.join(output_dir, "temp_custom_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(temp_dir, exist_ok=True)
    logger.debug(f"Created temporary working directory: {temp_dir}")

    # Prepare arguments for custom function
    custom_args = {
        'df': df,
        'target': args.target,
        'model_id': args.model_id,
        'output_dir': temp_dir,
        'config': config,
        'time_col': args.time_col,
        'version': version_num,
        'forecast_horizon': getattr(args, 'forecast_horizon', None)
    }
    
    logger.debug(f"Prepared arguments for custom function: {custom_args.keys()}")
    
    # Try to log function signature for debugging
    try:
        import inspect
        signature = inspect.signature(custom_function)
        logger.debug(f"Custom function signature: {signature}")
    except Exception as e:
        logger.debug(f"Could not inspect custom function signature: {str(e)}")

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

        # Check return type and unpack appropriately
        if isinstance(result, tuple) and len(result) >= 2:
            logger.debug(f"Custom function returned a tuple with {len(result)} elements")
            best_model, results = result[0], result[1]

            # Extract paths if provided
            temp_model_path = result[2] if len(result) > 2 else None
            temp_metadata_path = result[3] if len(result) > 3 else None
            
            logger.debug(f"Extracted from result - model path: {temp_model_path}, metadata path: {temp_metadata_path}")

            # Copy model file to versioned directory
            if temp_model_path and os.path.exists(temp_model_path):
                model_filename = os.path.basename(temp_model_path)
                final_model_path = os.path.join(versioned_output_dir, model_filename)
                os.makedirs(os.path.dirname(final_model_path), exist_ok=True)

                logger.debug(f"Copying model file from {temp_model_path} to {final_model_path}")
                copy_start = datetime.now()
                
                model_size_mb = os.path.getsize(temp_model_path) / (1024 * 1024)
                shutil.copy2(temp_model_path, final_model_path)
                
                copy_time = (datetime.now() - copy_start).total_seconds()
                logger.info(f"Copied model from {temp_model_path} to {final_model_path} ({model_size_mb:.2f} MB) in {copy_time:.2f} seconds")
            else:
                if temp_model_path:
                    logger.warning(f"Model file not found at {temp_model_path}")
                else:
                    logger.warning("No model path was returned by the custom function")
                    
                final_model_path = os.path.join(versioned_output_dir, f"{args.model_id}.pkl")
                logger.warning(f"No model file found, using default path: {final_model_path}")

            # Copy metadata file to versioned directory
            if temp_metadata_path and os.path.exists(temp_metadata_path):
                try:
                    logger.debug(f"Processing metadata from {temp_metadata_path}")
                    process_start = datetime.now()
                    
                    with open(temp_metadata_path, 'r') as f:
                        metadata = json.load(f)

                    # Update version information
                    metadata['version'] = f"v{version_num}"
                    metadata['output_dir'] = versioned_output_dir
                    
                    # Add timestamp if not present
                    if 'timestamp' not in metadata:
                        metadata['timestamp'] = datetime.now().isoformat()
                    
                    # Add runtime information
                    if 'runtime_seconds' not in metadata:
                        metadata['runtime_seconds'] = fn_time

                    final_metadata_path = os.path.join(versioned_output_dir, "metadata.json")
                    os.makedirs(os.path.dirname(final_metadata_path), exist_ok=True)

                    with open(final_metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    process_time = (datetime.now() - process_start).total_seconds()
                    metadata_size_kb = os.path.getsize(final_metadata_path) / 1024
                    logger.info(f"Updated metadata and saved to {final_metadata_path} ({metadata_size_kb:.2f} KB) in {process_time:.2f} seconds")
                    
                    # Log metadata contents summary
                    try:
                        keys = list(metadata.keys())
                        if 'best_model' in metadata:
                            best_model_info = metadata['best_model']
                            best_model_name = best_model_info.get('name', 'Unknown')
                            metrics = {k: v for k, v in best_model_info.items() if 'metric' in k.lower()}
                            logger.debug(f"Metadata contains best model: {best_model_name} with metrics: {metrics}")
                    except Exception as e:
                        logger.debug(f"Could not summarize metadata contents: {str(e)}")
                        
                except Exception as e:
                    logger.error(f"Error processing metadata: {str(e)}", exc_info=True)
                    final_metadata_path = os.path.join(versioned_output_dir, "metadata.json")
            else:
                if temp_metadata_path:
                    logger.warning(f"Metadata file not found at {temp_metadata_path}")
                else:
                    logger.warning("No metadata path was returned by the custom function")
                    
                final_metadata_path = os.path.join(versioned_output_dir, "metadata.json")
                logger.warning(f"No metadata file found, using default path: {final_metadata_path}")

            # Copy any log files if present
            logger.debug(f"Checking for log files in {temp_dir}")
            log_start = datetime.now()
            log_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.log'):
                        src = os.path.join(root, file)
                        rel_path = os.path.relpath(src, temp_dir)
                        dst = os.path.join(versioned_output_dir, rel_path)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        
                        file_size_kb = os.path.getsize(src) / 1024
                        logger.debug(f"Copying log file: {src} ({file_size_kb:.2f} KB) to {dst}")
                        
                        shutil.copy2(src, dst)
                        log_files.append(dst)

            log_copy_time = (datetime.now() - log_start).total_seconds()
            if log_files:
                logger.info(f"Copied {len(log_files)} log files to versioned directory in {log_copy_time:.2f} seconds")
            else:
                logger.warning("No log files found in the temporary directory")

            # Clean up temporary directory
            cleanup_start = datetime.now()
            try:
                logger.debug(f"Removing temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir)
                cleanup_time = (datetime.now() - cleanup_start).total_seconds()
                logger.info(f"Removed temporary directory: {temp_dir} in {cleanup_time:.2f} seconds")
            except Exception as e:
                logger.warning(f"Failed to remove temporary directory: {str(e)}")
            
            total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Custom ML flow completed in {total_time:.2f} seconds")
            
            return best_model, results, final_model_path, final_metadata_path
        else:
            logger.warning(f"Custom function did not return expected tuple format. Got: {type(result)}")
            
            # Try to extract useful information from the result
            if result is not None:
                logger.debug(f"Result type: {type(result)}")
                
                if hasattr(result, '__dict__'):
                    logger.debug(f"Result attributes: {dir(result)}")
            
            final_model_path = os.path.join(versioned_output_dir, f"{args.model_id}.pkl")
            final_metadata_path = os.path.join(versioned_output_dir, "metadata.json")
            
            total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Custom ML flow completed with unexpected return format in {total_time:.2f} seconds")
            
            return result, None, final_model_path, final_metadata_path

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

        error_metadata = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model_id": args.model_id,
            "version": f"v{version_num}",
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "runtime_seconds": error_time
        }

        # Add context to error metadata
        error_metadata["context"] = {
            "function_path": function_path,
            "function_name": function_name,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "temp_dir": temp_dir,
            "df_shape": list(df.shape) if 'df' in locals() else None,
        }

        error_metadata_path = os.path.join(versioned_output_dir, "metadata.json")
        os.makedirs(os.path.dirname(error_metadata_path), exist_ok=True)

        logger.debug(f"Saving error metadata to {error_metadata_path}")
        try:
            with open(error_metadata_path, 'w') as f:
                json.dump(error_metadata, f, indent=2)
            logger.info(f"Error metadata saved to {error_metadata_path}")
        except Exception as metadata_error:
            logger.error(f"Could not save error metadata: {str(metadata_error)}")

        # Try to clean up temporary directory
        try:
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                logger.debug(f"Cleaning up temporary directory after error: {temp_dir}")
                shutil.rmtree(temp_dir)
                logger.debug(f"Removed temporary directory: {temp_dir}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up temporary directory: {str(cleanup_error)}")

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

def main():
   """Main function that orchestrates the ML pipeline."""
   # Parse command line arguments
   args = parse_arguments()
   
   # Configure logging based on verbosity
   log_level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
   logging.basicConfig(level=log_level, 
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
   
   # Log pipeline start
   start_time = datetime.now()
   logger.info(f"Starting ML pipeline with model_id: {args.model_id}")
   logger.debug(f"Arguments: {vars(args)}")
   
   # Construct paths based on model_id and version
   model_id = args.model_id
   version = args.version
   
   # Base directories
   base_dir = os.path.join("custom_ml_data")
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
           return 1

   # Log input file information
   file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
   logger.info(f"Using input data: {data_path} ({file_size_mb:.2f} MB)")
   
   try:
       # Load configuration
       config_start = datetime.now()
       try:
           config = ml_utils.load_config(args.config_path)
           config_time = (datetime.now() - config_start).total_seconds()
           logger.info(f"Configuration loaded from {args.config_path} in {config_time:.2f} seconds")
           
           # Log key configuration sections
           if config:
               top_level_keys = list(config.keys())
               logger.debug(f"Configuration sections: {top_level_keys}")
               
               # Log common settings
               if 'common' in config:
                   common_sections = list(config['common'].keys())
                   logger.debug(f"Common configuration sections: {common_sections}")
                   
                   # Check for important settings
                   train_test_split = config.get('common', {}).get('train_test_split', {})
                   if train_test_split:
                       test_size = train_test_split.get('test_size', 0.2)
                       stratify = train_test_split.get('stratify', True)
                       logger.debug(f"Train-test split: test_size={test_size}, stratify={stratify}")
           else:
               logger.warning("Config loaded but is empty")
       except Exception as e:
           logger.warning(f"Error loading config: {str(e)}. Using default configuration.")
           logger.debug(f"Config error details: {str(e)}", exc_info=True)
           config = {}
       
       # Log system information
       try:
           import platform
           import psutil
           
           logger.debug(f"System information:")
           logger.debug(f"  - OS: {platform.system()} {platform.release()}")
           logger.debug(f"  - Python: {sys.version}")
           
           mem = psutil.virtual_memory()
           logger.debug(f"  - Memory: {mem.total / (1024**3):.2f} GB total, {mem.available / (1024**3):.2f} GB available")
           logger.debug(f"  - CPU count: {psutil.cpu_count()} logical cores")
           
           disk = psutil.disk_usage(os.path.dirname(os.path.abspath(data_path)))
           logger.debug(f"  - Disk space: {disk.total / (1024**3):.2f} GB total, {disk.free / (1024**3):.2f} GB free")
       except ImportError:
           logger.debug("System information logging skipped (psutil not available)")
       except Exception as e:
           logger.debug(f"Error collecting system information: {str(e)}")
       
       # Load the data
       data_load_start = datetime.now()
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
           
           data_load_time = (datetime.now() - data_load_start).total_seconds()
           memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
           
           logger.info(f"Data loaded in {data_load_time:.2f} seconds: {df.shape[0]:,} rows, {df.shape[1]:,} columns, {memory_usage_mb:.2f} MB")
           
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
           
           # Check for target column
           if args.target and args.target in df.columns:
               logger.debug(f"Target column '{args.target}' found in data")
           elif args.target:
               logger.warning(f"Target column '{args.target}' NOT found in data columns: {df.columns.tolist()}")
           
           # Check for time column
           if args.time_col and args.time_col in df.columns:
               logger.debug(f"Time column '{args.time_col}' found in data")
               
               # Check if time column is actually a datetime
               if not pd.api.types.is_datetime64_any_dtype(df[args.time_col]):
                   logger.warning(f"Time column '{args.time_col}' is not a datetime type: {df[args.time_col].dtype}")
                   
                   # Try to convert
                   try:
                       logger.debug(f"Attempting to convert '{args.time_col}' to datetime")
                       df[args.time_col] = pd.to_datetime(df[args.time_col])
                       logger.info(f"Successfully converted '{args.time_col}' to datetime")
                   except Exception as e:
                       logger.warning(f"Could not convert time column to datetime: {str(e)}")
           elif args.time_col:
               logger.warning(f"Time column '{args.time_col}' NOT found in data columns: {df.columns.tolist()}")
               
       except Exception as e:
           logger.error(f"Error loading data: {str(e)}", exc_info=True)
           raise
       
       # Check if custom ML flow is enabled
       custom_ml_enabled = config.get('common', {}).get('custom_ml_model', {}).get('enabled', False)
       
       logger.info(f"Custom ML flow enabled: {custom_ml_enabled}")
       
       pipeline_start = datetime.now()
       if custom_ml_enabled:
           logger.info("="*50)
           logger.info("Running custom ML model flow")
           logger.info("="*50)
           
           # Run custom ML flow
           best_model, results, model_path, metadata_path = run_custom_ml_flow(args, config, df, input_dir, output_dir)
           
           pipeline_time = (datetime.now() - pipeline_start).total_seconds()
           logger.info(f"Custom pipeline completed in {pipeline_time:.2f} seconds")
           logger.info(f"Model saved to: {model_path}")
       else:
           logger.info("="*50)
           logger.info("Running default ML pipeline")
           logger.info("="*50)
           
           # Prepare common parameters for default pipeline
           pipeline_args = {
               'df': df,
               'target': args.target,
               'model_id': model_id,
               'output_dir': output_dir,
               'config': config,
               'data_path': data_path,
               'config_path': args.config_path,
               'version': version,
               'time_col': args.time_col,
               'forecast_horizon': getattr(args, 'forecast_horizon', 7)
           }
           
           # Log pipeline arguments
           logger.debug(f"Pipeline arguments: {pipeline_args.keys()}")
           
           # Run default pipeline
           best_model, results, model_path, metadata_path = run_default_ml_pipeline(**pipeline_args)
           
           pipeline_time = (datetime.now() - pipeline_start).total_seconds()
           logger.info(f"Default pipeline completed in {pipeline_time:.2f} seconds")
           logger.info(f"Model saved to: {model_path}")
       
       # Print final results
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
           
       total_time = (datetime.now() - start_time).total_seconds()
       logger.info(f"\nPipeline completed successfully in {total_time:.2f} seconds!")
       return 0
       
   except Exception as e:
       total_time = (datetime.now() - start_time).total_seconds()
       logger.error(f"Error in pipeline execution after {total_time:.2f} seconds: {str(e)}", exc_info=True)
       
       # Try to create error report
       try:
           error_metadata = {
               "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
               "model_id": model_id,
               "status": "failed",
               "error": str(e),
               "error_type": type(e).__name__,
               "start_time": start_time.isoformat(),
               "end_time": datetime.now().isoformat(),
               "runtime_seconds": total_time
           }
           
           # Create error output directory
           error_dir = os.path.join(output_dir, "error_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
           os.makedirs(error_dir, exist_ok=True)
           
           error_path = os.path.join(error_dir, "error_metadata.json")
           with open(error_path, 'w') as f:
               json.dump(error_metadata, f, indent=2)
               
           logger.info(f"Error report saved to: {error_path}")
       except Exception as err_ex:
           logger.debug(f"Could not save error report: {str(err_ex)}")
           
       return 1

if __name__ == "__main__":
    sys.exit(main())