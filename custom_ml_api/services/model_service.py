"""
Service layer for model training operations.
"""
import os
import re
import json
import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import traceback
from datetime import datetime
from custom_ml_api.config import ML_CONFIG
# Import from the existing ML pipeline code
from custom_ml.src import ml_utils

# Import local modules
from custom_ml_api.utils.error_handler import (
    ValidationError,
    ResourceNotFoundError,
    TrainingError
)
from custom_ml_api.config import (
    MODEL_DIR,
    CONFIG_PATH,
    REQUIRED_OUTPUT_FILES,
    MODEL_ID_PATTERN
)

logger = logging.getLogger(__name__)


class ModelTrainingService:
    """Service for model training operations."""
    
    @staticmethod
    def validate_model_id(model_id: str) -> bool:
        """
        Validate if the model_id is in the correct format.
        
        Args:
            model_id: The model ID to validate
            
        Returns:
            bool: True if valid, raises ValidationError otherwise
        """
        if not model_id:
            raise ValidationError("model_id cannot be empty")
        
        if not re.match(MODEL_ID_PATTERN, model_id):
            raise ValidationError(
                f"Invalid model_id format. Must match pattern: {MODEL_ID_PATTERN}",
                {"model_id": model_id, "pattern": MODEL_ID_PATTERN}
            )
        
        return True

    def train_model(self, model_id: str, target: str = 'target') -> Dict[str, Any]:
        """
        Train a model with the specified model_id and target column.
        
        Args:
            model_id: The model ID
            target: The target column name (defaults to 'target')
            
        Returns:
            Dict with training results
        """
        start_time = datetime.now()
        logger.info(f"Starting model training for model_id: {model_id}, target: {target}")
        
        # Validate model_id
        self.validate_model_id(model_id)
        
        try:
            # Set up direct paths without creating temporary directories
            input_data_path = f"data/training/input/{model_id}/v1/input_data.csv"
            output_base_dir = f"data/training/output/{model_id}"
            version_dir, version_num = ml_utils.get_next_version_dir(output_base_dir, model_id)
            
            # Create version directory if needed
            os.makedirs(version_dir, exist_ok=True)
            
            # Set up log file in version directory
            
            logger.info(f"Using input data: {input_data_path}")
            logger.info(f"Using output directory: {version_dir} (v{version_num})")
            logger.info(f"Using target column: {target}")
            
            # Load data directly
            try:
                logger.info(f"Loading data from {input_data_path}")
                df = pd.read_csv(input_data_path)
                logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            except FileNotFoundError:
                raise ValidationError(f"Input data file not found: {input_data_path}")
            except Exception as e:
                raise ValidationError(f"Error loading input data: {str(e)}")
            
            # Load configuration
            config = ML_CONFIG
            
            # Set up args for the ML pipeline
            class Args:
                pass
            
            args = Args()
            args.model_id = model_id
            args.target = target  # Assuming 'target' is the target column
            args.config_path = CONFIG_PATH
            args.version = f'v{version_num}'
            args.time_col = None
            args.verbose = True
            args.quiet = False
            
            # Run the custom ML flow if enabled, otherwise use default
            custom_ml_enabled = config.get('common', {}).get('custom_ml_model', {}).get('enabled', False)
            
            try:
                # Execute either custom or default ML flow
                if custom_ml_enabled:
                    logger.info("Running custom ML model flow")
                    best_model, results, model_path, metadata_path = ml_utils.run_custom_ml_flow(
                        args, config, df, os.path.dirname(input_data_path), 
                        version_dir, model_id
                    )
                    validation_report_path = os.path.join(version_dir, "validation_report.md")
                    custom_validation_info = {}
                    
                    if os.path.exists(validation_report_path):
                        custom_validation_info["validation_report"] = validation_report_path
                        custom_validation_info["passed_validation"] = "✅ PASSED" in open(validation_report_path).read()
                        logger.info(f"Custom pipeline validation: {'passed' if custom_validation_info['passed_validation'] else 'failed'}")
                    else:
                        logger.warning("No validation report found for custom pipeline")

                else:
                    logger.info("Running default ML pipeline")
                    best_model, results, model_path, metadata_path = ml_utils.run_default_ml_pipeline(
                        df=df,
                        target=args.target,
                        model_id=model_id,
                        output_dir=version_dir,
                        config=config,
                        data_path=input_data_path,
                        config_path=args.config_path,
                        version=args.version,
                        time_col=args.time_col,
                        forecast_horizon=7
                    )
            except Exception as e:
                raise 

            
            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Model training completed in {training_time:.2f} seconds")
            
            # Return training results
            result = {
                "model_id": model_id,
                "version": f"v{version_num}",
                "status": "completed",
                "training_time_seconds": training_time,
                "output_directory": version_dir
            }

            if custom_ml_enabled and 'custom_validation_info' in locals():
                result["custom_validation"] = custom_validation_info

            return result
            
        except Exception as e:
            logger.error(f"Error in model training for model_id {model_id}: {str(e)}", exc_info=True)
            
            if isinstance(e, (ValidationError, ResourceNotFoundError, TrainingError)):
                raise
            
            raise TrainingError(
                f"Model training failed for model_id: {model_id}", 
                {"error": str(e), "traceback": traceback.format_exc()}
            )

    def get_model_files(self, model_id: str) -> Dict[str, str]:
        """
        Get the paths to model files for the specified model_id.
        
        Args:
            model_id: The model ID
            
        Returns:
            Dict with paths to model files
        """
        # Validate model_id
        self.validate_model_id(model_id)
        
        # Get base model directory
        base_dir = f"data/training/output/{model_id}"
        
        if not os.path.exists(base_dir):
            raise ResourceNotFoundError(f"No model found with id: {model_id}")
        
        # Find the latest version directory
        version_dirs = []
        for item in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, item)) and item.startswith('v'):
                try:
                    version_num = int(item[1:])
                    version_dirs.append((item, version_num))
                except ValueError:
                    continue
        
        if not version_dirs:
            raise ResourceNotFoundError(f"No versions found for model {model_id}")
            
        # Get the latest version
        version_dirs.sort(key=lambda x: x[1], reverse=True)
        latest_version = version_dirs[0][0]
        model_dir = os.path.join(base_dir, latest_version)
        
        # Check for required files
        model_path = os.path.join(model_dir, f"{model_id}.pkl")
        metadata_path = os.path.join(model_dir, "metadata.json")
        log_path = os.path.join(model_dir, "training.log")
        
        # If model.pkl doesn't exist, try other names
        if not os.path.exists(model_path):
            for file in os.listdir(model_dir):
                if file.endswith('.pkl'):
                    model_path = os.path.join(model_dir, file)
                    break
        
        missing_files = []
        if not os.path.exists(model_path):
            missing_files.append('model file (*.pkl)')
        if not os.path.exists(metadata_path):
            missing_files.append('metadata.json')
        if not os.path.exists(log_path):
            # Try to find any log file
            log_files = [f for f in os.listdir(model_dir) if f.endswith('.log')]
            if log_files:
                log_path = os.path.join(model_dir, log_files[0])
            else:
                missing_files.append('training.log')
        
        if missing_files:
            raise ResourceNotFoundError(
                f"Missing files for model {model_id}",
                {"missing_files": missing_files}
            )
        
        return {
            "model": model_path,
            "metadata": metadata_path,
            "log": log_path
        }