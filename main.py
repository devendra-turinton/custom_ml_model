import os
import sys
import logging
import json
from datetime import datetime
import src.ml_utils as ml_utils

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
        # Run the appropriate ML flow based on configuration
        config = ml_utils.load_config(args.config_path)
        custom_ml_enabled = config.get('common', {}).get('custom_ml_model', {}).get('enabled', False)
        
        # Prepare input and output directories
        input_dir, output_dir, data_path = ml_utils.prepare_directories(args.model_id, args.version)
        
        # Load data
        df = ml_utils.load_input_data(data_path, input_dir)
        
        if custom_ml_enabled:
            logger.info("=" * 50)
            logger.info("Running custom ML model flow")
            logger.info("=" * 50)
            
            # Run custom ML flow
            best_model, results, model_path, metadata_path = ml_utils.run_custom_ml_flow(
                args, config, df, input_dir, output_dir
            )
        else:
            logger.info("=" * 50)
            logger.info("Running default ML pipeline")
            logger.info("=" * 50)
            
            # Run default pipeline
            best_model, results, model_path, metadata_path = ml_utils.run_default_ml_pipeline(
                df=df,
                target=args.target,
                model_id=args.model_id,
                output_dir=output_dir,
                config=config,
                data_path=data_path,
                config_path=args.config_path,
                version=args.version,
                time_col=args.time_col,
                forecast_horizon=getattr(args, 'forecast_horizon', 7)
            )
        
        # Print final results
        total_time = (datetime.now() - start_time).total_seconds()
        ml_utils.print_pipeline_results(model_path, metadata_path, args.model_id, output_dir, 
                                       best_model, results, total_time)
        return 0
        
    except Exception as e:
        total_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error in pipeline execution after {total_time:.2f} seconds: {str(e)}", exc_info=True)
        
        # Create error report
        ml_utils.create_error_report(args.model_id, output_dir, e, start_time, total_time)
        return 1

if __name__ == "__main__":
    sys.exit(main())