import os
import sys
import logging
from datetime import datetime
import src.ml_utils as ml_utils
from custom_ml_api.config import ML_CONFIG
logger = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)

def main():
    """Main function that orchestrates the ML pipeline."""
    # Parse command line arguments and set up basic logging
    args = ml_utils.parse_arguments()
    ml_utils.setup_basic_logging(args.verbose, args.quiet)
    
    start_time = datetime.now()
    logger.info(f"Starting ML pipeline with model_id: {args.model_id}")
    
    try:
        # Load configuration

        config = ML_CONFIG
        custom_ml_enabled = config.get('common', {}).get('custom_ml_model', {}).get('enabled', False)
        
        # Define direct input and output paths
        input_data_path = f"data/training/input/{args.model_id}/v1/input_data.csv"
        output_base_dir = f"data/training/output/{args.model_id}"
        
        # Get the next version directory
        version_dir, version_num = ml_utils.get_next_version_dir(output_base_dir, args.model_id)
        logger.info(f"Using output directory: {version_dir} (v{version_num})")
        
        # Load data directly from input path
        logger.info(f"Loading data from {input_data_path}")
        try:
            df = ml_utils.load_input_data(input_data_path, os.path.dirname(input_data_path))
            logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        except FileNotFoundError:
            logger.error(f"Input data file not found: {input_data_path}")
            return 1
        except Exception as e:
            logger.error(f"Error loading input data: {str(e)}")
            return 1
        
        # Set up version in args
        args.version = f"v{version_num}"
        
        # Run custom ML flow or default pipeline based on configuration
        if custom_ml_enabled:
            logger.info("=" * 50)
            logger.info("Running custom ML model flow")
            logger.info("=" * 50)
            
            # Update the config if needed to point to latest version
            custom_config = config.get('common', {}).get('custom_ml_model', {})
            function_path = custom_config.get('function_path', '')
            
            # Try to get the latest version of the custom code
            latest_code_path = ml_utils.get_latest_custom_code_path(args.model_id)
            if latest_code_path:
                # Update the config to use the latest version
                logger.info(f"Using latest custom code version: {latest_code_path}")
                if 'common' not in config:
                    config['common'] = {}
                if 'custom_ml_model' not in config['common']:
                    config['common']['custom_ml_model'] = {}
                config['common']['custom_ml_model']['function_path'] = latest_code_path

            # Run custom ML flow with direct paths
            best_model, results, model_path, metadata_path = ml_utils.run_custom_ml_flow(
                args, config, df, os.path.dirname(input_data_path), version_dir, args.model_id
            )
        else:
            logger.info("=" * 50)
            logger.info("Running default ML pipeline")
            logger.info("=" * 50)
            
            # Run default pipeline with direct paths
            best_model, results, model_path, metadata_path = ml_utils.run_default_ml_pipeline(
                df=df,
                target=args.target,
                model_id=args.model_id,
                output_dir=version_dir,
                config=config,
                data_path=input_data_path,
                config_path=args.config_path,
                version=args.version,
                time_col=args.time_col,
                forecast_horizon=getattr(args, 'forecast_horizon', 7)
            )
        
        # Print final results
        total_time = (datetime.now() - start_time).total_seconds()
        ml_utils.print_pipeline_results(model_path, metadata_path, args.model_id, version_dir, 
                                      best_model, results, total_time)
        return 0
        
    except Exception as e:
        total_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error in pipeline execution after {total_time:.2f} seconds: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())