"""
Service layer for model testing operations.
"""
import os
import re
import json
import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import traceback
from datetime import datetime

# Import ModelPredictor class
from custom_ml.testing_pipeline import ModelPredictor

# Import from existing ML pipeline code
from custom_ml.src import ml_utils

# Import local modules
from custom_ml_api.utils.error_handler import (
    ValidationError,
    ResourceNotFoundError,
    TestingError
)
from custom_ml_api.config import (
    CONFIG_PATH,
    MODEL_ID_PATTERN
)

logger = logging.getLogger(__name__)


class ModelTestingService:
    """Service for model testing operations."""    
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

    def test_model(self, model_id: str) -> Dict[str, Any]:
        """
        Test a model with the specified model_id.
        
        Args:
            model_id: The model ID
            
        Returns:
            Dict with testing results
        """
        start_time = datetime.now()
        logger.info(f"Starting model testing for model_id: {model_id}")
        
        # Validate model_id
        self.validate_model_id(model_id)
        
        try:
            # Define direct paths for input and output
            input_data_path = f"data/testing/input/{model_id}/v1/input_data.csv"
            output_base_dir = f"data/testing/output/{model_id}"
            
            # Create output version directory
            version_dir, version_num = ml_utils.get_next_version_dir(output_base_dir, model_id)
            
            # Ensure version directory exists
            os.makedirs(version_dir, exist_ok=True)
            
            logger.info(f"Using input test data: {input_data_path}")
            logger.info(f"Using output directory: {version_dir} (v{version_num})")
            
            # Look for the trained model in the training output directory
            training_output_dir = f"data/training/output/{model_id}"
            model_dir = self._find_latest_model_version(training_output_dir)
            
            if not model_dir:
                raise ResourceNotFoundError(f"No trained model found for model_id: {model_id}")
            
            logger.info(f"Found trained model at: {model_dir}")
            
            # Extract the version from the directory name
            version_name = os.path.basename(version_dir)
            
            # Initialize the ModelPredictor with explicit instructions to use our directory
            predictor = ModelPredictor(
                model_dir=model_dir,
                output_dir=version_dir,  # Direct path to use
                model_id=model_id,           # Set to None to prevent auto-versioning
                base_data_dir="data"     # Base data directory
            )
            
            # Update the predictor's model_id after initialization to maintain correct metadata
            predictor.model_id = model_id
            predictor.version = f"v{version_num}"
            predictor.prediction_metadata['model_id'] = model_id
            predictor.prediction_metadata['version'] = f"v{version_num}"
            
            # Check if test data exists
            if not os.path.exists(input_data_path):
                # Try to auto-detect test data
                logger.warning(f"Test data not found at {input_data_path}, attempting to auto-detect")
                auto_detected_path = predictor.auto_detect_test_data()
                
                if not auto_detected_path:
                    raise ValidationError(f"No test data found for model_id: {model_id}")
                
                input_data_path = auto_detected_path
                logger.info(f"Auto-detected test data at: {input_data_path}")
            else:
                logger.info(f"Using provided test data at: {input_data_path}")
            
            # Run the prediction pipeline
            logger.info("Running prediction pipeline...")
            result_df = predictor.run_prediction_pipeline(data_path=input_data_path)
            
            # Extract metrics from metadata
            metrics = self._extract_metrics_from_predictor(predictor)
            
            # Get file information
            test_files = self._get_test_file_info(version_dir)
            
            # Calculate testing time
            testing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Model testing completed in {testing_time:.2f} seconds")
            
            # Return testing results
            return {
                "model_id": model_id,
                "version": f"v{version_num}",
                "status": "completed",
                "testing_time_seconds": testing_time,
                "output_directory": version_dir,
                "files": test_files,
                "metrics": metrics,
                "rows_tested": len(result_df) if result_df is not None else 0,
                "target_column_present": predictor.prediction_metadata.get('preprocessing', {}).get('target_present', False)
            }
            
        except Exception as e:
            logger.error(f"Error in model testing for model_id {model_id}: {str(e)}", exc_info=True)
            
            if isinstance(e, (ValidationError, ResourceNotFoundError, TestingError)):
                raise
            
            raise TestingError(
                f"Model testing failed for model_id: {model_id}", 
                {"error": str(e), "traceback": traceback.format_exc()}
            )

    def _find_latest_model_version(self, training_output_dir: str) -> Optional[str]:
        """
        Find the latest version directory containing a trained model.
        
        Args:
            training_output_dir: Base directory for trained models
            
        Returns:
            str: Path to the latest model version directory, or None if not found
        """
        if not os.path.exists(training_output_dir):
            logger.warning(f"Training output directory not found: {training_output_dir}")
            return None
        
        # Find all version directories
        version_dirs = []
        for item in os.listdir(training_output_dir):
            if os.path.isdir(os.path.join(training_output_dir, item)) and item.startswith('v'):
                try:
                    version_num = int(item[1:])  # Extract number from 'v1', 'v2', etc.
                    version_dirs.append((item, version_num, os.path.join(training_output_dir, item)))
                except ValueError:
                    continue
        
        if not version_dirs:
            logger.warning(f"No version directories found in {training_output_dir}")
            return None
        
        # Sort by version number (descending)
        version_dirs.sort(key=lambda x: x[1], reverse=True)
        
        # Find the first directory with a model file
        for _, _, dir_path in version_dirs:
            # Check for .pkl files
            model_files = [f for f in os.listdir(dir_path) if f.endswith('.pkl')]
            if model_files:
                logger.info(f"Found model in {dir_path}: {model_files[0]}")
                return dir_path
        
        logger.warning(f"No model files found in any version directory for {training_output_dir}")
        return None

    def _extract_metrics_from_predictor(self, predictor: ModelPredictor) -> Dict[str, Any]:
        """
        Extract metrics from the predictor's metadata.
        
        Args:
            predictor: ModelPredictor instance with results
            
        Returns:
            Dict with metrics information
        """
        metrics = {}
        
        # Get performance metrics if available
        if hasattr(predictor, 'prediction_metadata'):
            performance_metrics = predictor.prediction_metadata.get('results', {}).get('performance_metrics', {})
            if performance_metrics:
                if predictor.problem_type == 'regression':
                    # Extract regression metrics
                    metrics['regression'] = {
                        'r2': performance_metrics.get('r2'),
                        'rmse': performance_metrics.get('rmse'),
                        'mae': performance_metrics.get('mae')
                    }
                    
                    if 'mape' in performance_metrics:
                        metrics['regression']['mape'] = performance_metrics.get('mape')
                    
                elif predictor.problem_type == 'classification':
                    # Extract classification metrics
                    metrics['classification'] = {
                        'accuracy': performance_metrics.get('accuracy')
                    }
                    
                    # For binary classification
                    if 'precision' in performance_metrics:
                        metrics['classification'].update({
                            'precision': performance_metrics.get('precision'),
                            'recall': performance_metrics.get('recall'),
                            'f1': performance_metrics.get('f1')
                        })
                    
                    # For multiclass classification
                    elif 'precision_macro' in performance_metrics:
                        metrics['classification'].update({
                            'precision_macro': performance_metrics.get('precision_macro'),
                            'recall_macro': performance_metrics.get('recall_macro'),
                            'f1_macro': performance_metrics.get('f1_macro')
                        })
        
        # Add problem type information
        if hasattr(predictor, 'problem_type'):
            metrics['problem_type'] = predictor.problem_type
        
        return metrics

    @staticmethod
    def _get_test_file_info(output_dir: str) -> List[Dict[str, Any]]:
        """
        Get information about output files.
        
        Args:
            output_dir: Directory containing output files
            
        Returns:
            List of dictionaries with file info
        """
        file_info = []
        
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    file_info.append({
                        "name": filename,
                        "path": file_path,
                        "size_bytes": file_size,
                        "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                    })
        
        return file_info

    def get_test_files(self, model_id: str, version: Optional[str] = None) -> Dict[str, str]:
        """
        Get the paths to test result files for the specified model_id.
        
        Args:
            model_id: The model ID
            version: Optional specific version (default: latest)
            
        Returns:
            Dict with paths to test result files
        """
        # Validate model_id
        self.validate_model_id(model_id)
        
        # Get base test directory
        base_dir = f"data/testing/output/{model_id}"
        
        if not os.path.exists(base_dir):
            raise ResourceNotFoundError(f"No test results found for model_id: {model_id}")
        
        # Find the specified or latest version directory
        if version:
            # Use the specified version
            version_dir = os.path.join(base_dir, version)
            if not os.path.exists(version_dir):
                raise ResourceNotFoundError(f"Version {version} not found for model_id: {model_id}")
        else:
            # Find the latest version
            version_dirs = []
            for item in os.listdir(base_dir):
                if os.path.isdir(os.path.join(base_dir, item)) and item.startswith('v'):
                    try:
                        version_num = int(item[1:])
                        version_dirs.append((item, version_num))
                    except ValueError:
                        continue
            
            if not version_dirs:
                raise ResourceNotFoundError(f"No test versions found for model_id: {model_id}")
                
            # Get the latest version
            version_dirs.sort(key=lambda x: x[1], reverse=True)
            latest_version = version_dirs[0][0]
            version_dir = os.path.join(base_dir, latest_version)
        
        # Check for required files
        results_file = None
        metadata_file = None
        log_file = None
        
        for filename in os.listdir(version_dir):
            if filename.endswith('.csv') and 'result' in filename.lower():
                results_file = os.path.join(version_dir, filename)
            elif filename.endswith('.json') and 'metadata' in filename.lower():
                metadata_file = os.path.join(version_dir, filename)
            elif filename.endswith('.log'):
                log_file = os.path.join(version_dir, filename)
        
        missing_files = []
        if not results_file:
            missing_files.append('results file (.csv)')
        if not metadata_file:
            missing_files.append('metadata file (.json)')
        if not log_file:
            missing_files.append('log file (.log)')
        
        if missing_files:
            raise ResourceNotFoundError(
                f"Missing files for test results {model_id}",
                {"missing_files": missing_files}
            )
        
        return {
            "results": results_file,
            "metadata": metadata_file,
            "log": log_file,
            "version_directory": version_dir
        }