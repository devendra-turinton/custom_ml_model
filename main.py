
import os
import sys
import argparse
import logging
from datetime import datetime
import pandas as pd
import src.ml_utils as ml_utils
from models.regression import RegressionPipeline
from models.classification import ClassificationPipeline
from models.cluster import ClusteringPipeline
from models.time_series import TimeSeriesPipeline
logger = logging.getLogger(__name__)

def run_custom_ml_flow(args, config, input_dir, output_dir):
    """
    Run the custom ML pipeline flow.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
        input_dir: Input directory for data
        output_dir: Output directory for results
        
    Returns:
        tuple: (best_model, results) similar to standard pipeline
    """
    logger.info("Running custom ML flow...")
    
    # Get custom function configuration
    custom_config = config.get('common', {}).get('custom_ml_model', {})
    function_path = custom_config.get('function_path', '')
    function_name = custom_config.get('function_name', 'run_custom_pipeline')
    
    if not function_path:
        error_msg = "Custom ML model is enabled but no function_path is specified in config"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Load the custom function
    custom_function = ml_utils.load_custom_function(function_path, function_name)
    if custom_function is None:
        error_msg = f"Failed to load custom function '{function_name}' from {function_path}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Load data (similar to standard flow)
    data_path = os.path.join(input_dir, "input_data.csv")
    if not os.path.exists(data_path):
        logger.error(f"Input file not found: {data_path}")
        raise FileNotFoundError(f"Input file not found: {data_path}")
    
    # Determine file type and load
    file_ext = os.path.splitext(data_path)[1].lower()
    try:
        if file_ext == '.csv':
            df = pd.read_csv(data_path)
        elif file_ext in ['.xls', '.xlsx']:
            df = pd.read_excel(data_path)
        elif file_ext == '.json':
            df = pd.read_json(data_path)
        elif file_ext == '.parquet':
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
    
    # Prepare arguments for custom function
    custom_args = {
        'df': df,
        'target': args.target,
        'model_id': args.model_id,
        'output_dir': output_dir,
        'config': config,
        # Add any other arguments that might be needed
        'time_col': args.time_col,
        'version': args.version,
        'forecast_horizon': getattr(args, 'forecast_horizon', None)
    }
    
    try:
        # Call the custom function
        logger.info(f"Calling custom function '{function_name}'")
        start_time = datetime.now()
        
        # The custom function should return (best_model, results, model_path, metadata_path)
        result = custom_function(**custom_args)
        
        # Check return type and unpack appropriately
        if isinstance(result, tuple) and len(result) >= 2:
            best_model, results = result[0], result[1]
            
            # Log additional return values if available
            if len(result) > 2:
                model_path = result[2] if len(result) > 2 else None
                metadata_path = result[3] if len(result) > 3 else None
                
                if model_path:
                    logger.info(f"Custom model saved to: {model_path}")
                if metadata_path:
                    logger.info(f"Custom metadata saved to: {metadata_path}")
        else:
            logger.warning("Custom function did not return expected tuple format")
            best_model, results = result, None
        
        # Log execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Custom ML flow completed in {execution_time:.2f} seconds")
        
        return best_model, results
        
    except Exception as e:
        logger.error(f"Error in custom ML flow: {str(e)}", exc_info=True)
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
    
    # Custom code paths
    parser.add_argument('--custom_models_path', type=str, 
                        help='Path to custom models implementation')
    parser.add_argument('--custom_features_path', type=str, 
                        help='Path to custom feature engineering implementation')
    
    return parser.parse_args()

def main():
    """Main function that orchestrates the ML pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Construct paths based on model_id and version
    model_id = args.model_id
    version = args.version
    
    # Base directories
    base_dir = os.path.join("custom_ml_data")
    input_dir = os.path.join(base_dir, "training", "input", model_id, version)
    output_dir = os.path.join(base_dir, "training", "output", model_id)
    
    # Input data path
    data_path = os.path.join(input_dir, "input_data.csv")
    
    # Validate input file exists
    if not os.path.exists(data_path):
        logger.error(f"Input file not found: {data_path}")
        return 1

    logger.info(f"Using input data: {data_path}")
    
    
    try:
        # Load configuration regardless of pipeline type
        try:
            config = ml_utils.load_config(args.config_path)
            logger.info(f"Configuration loaded from {args.config_path}")
        except Exception as e:
            logger.warning(f"Error loading config: {str(e)}. Using default configuration.")
            config = {}
        
        # Check if custom ML flow is enabled
        custom_ml_enabled = config.get('common', {}).get('custom_ml_model', {}).get('enabled', False)
        
        if custom_ml_enabled:
            logger.info("Custom ML model flow is enabled")
            # Execute custom ML flow
            best_model, results = run_custom_ml_flow(args, config, input_dir, output_dir)
        else:
            # Check if target is provided - clustering if not
            if args.target is None:
                logger.info("No target column provided. Using clustering pipeline.")
                problem_type = 'clustering'
            else:
                logger.info(f"Target column: {args.target}")
                
                # Load the data to detect problem type
                import pandas as pd
                file_ext = os.path.splitext(data_path)[1].lower()
                if file_ext == '.csv':
                    df = pd.read_csv(data_path)
                elif file_ext in ['.xls', '.xlsx']:
                    df = pd.read_excel(data_path)
                elif file_ext == '.json':
                    df = pd.read_json(data_path)
                elif file_ext == '.parquet':
                    df = pd.read_parquet(data_path)
                else:
                    raise ValueError(f"Unsupported file extension: {file_ext}")
                    
                # Detect problem type
                problem_type = ml_utils.detect_problem_type(df, args.target, config)
                logger.info(f"Detected problem type: {problem_type}")

            # Initialize the appropriate pipeline based on problem type    
            if problem_type == 'regression':
                logger.info("Initializing Regression Pipeline")
                pipeline = RegressionPipeline(
                    data_path=data_path,
                    target=args.target,
                    model_id=model_id,
                    output_dir=output_dir,
                    config_path=args.config_path
                    
                )
            elif problem_type == 'classification':
                logger.info("Initializing Classification Pipeline")
                pipeline = ClassificationPipeline(
                    data_path=data_path,
                    target=args.target,
                    model_id=model_id,
                    output_dir=output_dir,
                    config_path=args.config_path
                )
            elif problem_type == 'clustering':
                logger.info("Initializing Clustering Pipeline")
                pipeline = ClusteringPipeline(
                    data_path=data_path,
                    model_id=model_id,
                    output_dir=output_dir,
                    config_path=args.config_path
                )
            elif problem_type == 'time_series':
                logger.info("Initializing Time Series Pipeline")
                pipeline = TimeSeriesPipeline(
                    data_path=data_path,
                    target=args.target,
                    time_col=args.time_col,
                    model_id=model_id,
                    output_dir=output_dir,
                    config_path=args.config_path,
                    forecast_horizon=args.forecast_horizon
                )
            else:
                logger.warning(f"Problem type '{problem_type}' not fully implemented yet")
                logger.info("Defaulting to Regression Pipeline")
                pipeline = RegressionPipeline(
                    data_path=data_path,
                    target=args.target,
                    model_id=model_id,
                    output_dir=output_dir,
                    config_path=args.config_path
                )

            # Run the pipeline
            logger.info("Running ML pipeline...")
            best_model, results = pipeline.run_pipeline()
            
            # Print final results
            logger.info("\n===== Pipeline Results =====")
            logger.info(f"Problem type: {problem_type}")
            logger.info(f"Model ID: {model_id}")
            logger.info(f"Output directory: {pipeline.output_dir}")

            if results is not None and not results.empty:
                logger.info("\nModel Performance:")
                num_models = min(3, len(results))
                
                if problem_type == 'regression':
                    metric_col = 'test_r2'
                    sort_ascending = False
                elif problem_type == 'classification':
                    metric_col = 'test_accuracy'
                    sort_ascending = False
                elif problem_type == 'clustering':
                    if 'silhouette' in results.columns:
                        metric_col = 'silhouette'
                        sort_ascending = False
                    else:
                        logger.info(results.to_string())
                        return 0
                else:
                    metric_col = next(iter(results.columns))  # Just use the first column
                    sort_ascending = False
                
                # Sort and display top models
                try:
                    top_models = results.sort_values(metric_col, ascending=sort_ascending).head(num_models)
                    
                    for idx, row in top_models.iterrows():
                        model_name = row['model']
                        if problem_type == 'regression':
                            metrics = f"R²: {row['test_r2']:.4f}, RMSE: {row['test_rmse']:.4f}, MAE: {row['test_mae']:.4f}"
                        elif problem_type == 'classification':
                            metrics = f"Accuracy: {row.get('test_accuracy', 0):.4f}"
                        elif problem_type == 'clustering':
                            silhouette = row.get('silhouette', None)
                            silhouette_str = f"Silhouette: {silhouette:.4f}" if silhouette is not None else "Silhouette: N/A"
                            metrics = f"Clusters: {int(row['n_clusters'])}, {silhouette_str}"
                        else:
                            metrics = f"{metric_col}: {row.get(metric_col, 'N/A')}"
                        
                        logger.info(f"  {model_name}: {metrics}")
                except Exception as e:
                    logger.warning(f"Could not sort results: {str(e)}")
                    logger.info(results.to_string())

            # Continue with the rest of your code
            logger.info(f"\nBest model saved to: {os.path.join(pipeline.output_dir, model_id + '.pkl')}")
            logger.info(f"Metadata saved to: {os.path.join(pipeline.output_dir, 'metadata.json')}")
            logger.info("\nPipeline completed successfully!")

    except Exception as e:
        logger.error(f"Error in pipeline execution: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())