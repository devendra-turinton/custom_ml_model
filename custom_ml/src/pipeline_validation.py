"""
Module for validating custom ML pipelines and their outputs.
"""
import os
import json
import pickle
import inspect
import logging
import pandas as pd
from datetime import datetime
from typing import Any, Dict, Tuple, Callable, List, Optional

logger = logging.getLogger(__name__)

def validate_custom_pipeline_output(
    custom_function: Callable,
    result: Any, 
    expected_model_path: str,
    expected_metadata_path: str,
    output_dir: str,
    raise_errors: bool = True  # New parameter to control error behavior
) -> Tuple[bool, Dict[str, Dict[str, bool]], Optional[Dict]]:
    """
    Validate the results of a custom ML pipeline execution.
    
    Args:
        custom_function: The custom pipeline function that was called
        result: The return value from the custom function
        expected_model_path: Expected path to the model file
        expected_metadata_path: Expected path to the metadata file
        output_dir: Directory where outputs should be stored
        raise_errors: Whether to raise exceptions for critical validation failures
        
    Returns:
        Tuple of (is_valid, validation_results, metadata_obj)
        
    Raises:
        ValueError: If critical testing fields are missing and raise_errors=True
    """
    validation_start = datetime.now()
    logger.info("Starting validation of custom pipeline outputs")
    
    validation_results = {
        "method_validation": {
            "exists": True,  # We already know it exists if we're here
            "correct_signature": False,
            "docstring_present": False
        },
        "return_value_validation": {
            "returns_tuple": False,
            "has_four_elements": False,
            "has_model": False,
            "has_results_df": False,
            "has_model_path": False,
            "has_metadata_path": False
        },
        "file_validation": {
            "model_file_exists": False,
            "model_file_not_empty": False,
            "model_file_loadable": False,
            "metadata_file_exists": False,
            "metadata_file_not_empty": False,
            "metadata_parsable": False,
            "log_file_exists": False,
            "log_file_not_empty": False
        },
        "metadata_validation": {
            "required_fields_present": False,
            "best_model_info_present": False,
            "status_completed": False,
            "test_metrics_present": False,
            "preprocessing_info_present": False,
            # New testing-specific validations
            "problem_type_present": False,
            "target_column_specified": False,
            "feature_types_defined": False
        }
    }
    
    # 1. Validate method signature
    try:
        logger.debug("Validating custom function signature")
        sig = inspect.signature(custom_function)
        
        # Expected parameters: df, target, model_id, output_dir, config
        required_params = ['df', 'target', 'model_id', 'output_dir', 'config']
        has_required_params = all(param in sig.parameters for param in required_params)
        
        # Check if there's a **kwargs parameter or similar
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        
        validation_results["method_validation"]["correct_signature"] = has_required_params
        
        # Check for docstring
        validation_results["method_validation"]["docstring_present"] = custom_function.__doc__ is not None and len(custom_function.__doc__.strip()) > 0
        
        if not has_required_params:
            missing = [p for p in required_params if p not in sig.parameters]
            logger.warning(f"Custom function is missing required parameters: {missing}")
        
        if not has_kwargs:
            logger.warning("Custom function does not accept **kwargs, which may cause issues with additional parameters")
    
    except Exception as e:
        logger.error(f"Error validating function signature: {str(e)}")
    
    # 2. Validate return value
    metadata_path_to_check = None
    try:
        logger.debug("Validating return value structure")
        
        # Check if it's a tuple
        if isinstance(result, tuple):
            validation_results["return_value_validation"]["returns_tuple"] = True
            
            # Check tuple length
            if len(result) >= 4:
                validation_results["return_value_validation"]["has_four_elements"] = True
                
                # Unpack and check each element
                model, results_df, model_path, metadata_path = result[:4]
                
                # Check model
                validation_results["return_value_validation"]["has_model"] = model is not None
                
                # Check results_df
                validation_results["return_value_validation"]["has_results_df"] = isinstance(results_df, pd.DataFrame)
                
                # Check paths
                validation_results["return_value_validation"]["has_model_path"] = isinstance(model_path, str) and len(model_path) > 0
                validation_results["return_value_validation"]["has_metadata_path"] = isinstance(metadata_path, str) and len(metadata_path) > 0
                
                if not validation_results["return_value_validation"]["has_model"]:
                    logger.warning("Custom function returned None for the model")
                
                if not validation_results["return_value_validation"]["has_results_df"]:
                    if results_df is None:
                        logger.warning("Custom function returned None for results_df")
                    else:
                        logger.warning(f"Custom function returned {type(results_df)} instead of DataFrame for results_df")
                
                if not validation_results["return_value_validation"]["has_model_path"]:
                    logger.warning("Custom function returned invalid model_path")
                
                if not validation_results["return_value_validation"]["has_metadata_path"]:
                    logger.warning("Custom function returned invalid metadata_path")
                else:
                    metadata_path_to_check = metadata_path
            else:
                logger.warning(f"Custom function returned tuple with {len(result)} elements, expected at least 4")
        else:
            logger.warning(f"Custom function did not return a tuple, got {type(result)} instead")
    
    except Exception as e:
        logger.error(f"Error validating return value: {str(e)}")
    
    # 3. Validate files
    try:
        logger.debug("Validating output files")
        
        # Determine model path to check
        model_path_to_check = None
        if validation_results["return_value_validation"]["has_model_path"] and os.path.exists(result[2]):
            model_path_to_check = result[2]
        elif os.path.exists(expected_model_path):
            model_path_to_check = expected_model_path
        
        # Determine metadata path to check
        if not metadata_path_to_check:
            if validation_results["return_value_validation"]["has_metadata_path"] and os.path.exists(result[3]):
                metadata_path_to_check = result[3]
            elif os.path.exists(expected_metadata_path):
                metadata_path_to_check = expected_metadata_path
        
        # Check model file
        if model_path_to_check:
            validation_results["file_validation"]["model_file_exists"] = True
            
            # Check if not empty
            file_size = os.path.getsize(model_path_to_check)
            validation_results["file_validation"]["model_file_not_empty"] = file_size > 0
            
            if validation_results["file_validation"]["model_file_not_empty"]:
                # Try to load the model
                try:
                    with open(model_path_to_check, 'rb') as f:
                        loaded_model = pickle.load(f)
                    
                    validation_results["file_validation"]["model_file_loadable"] = True
                    
                    # Additional check: does it look like a model/pipeline?
                    has_predict = hasattr(loaded_model, 'predict')
                    has_pipeline = hasattr(loaded_model, 'steps') or hasattr(loaded_model, 'named_steps')
                    
                    if not (has_predict or has_pipeline):
                        logger.warning(f"Model file at {model_path_to_check} doesn't appear to be a valid model or pipeline")
                
                except Exception as e:
                    logger.warning(f"Cannot load model file at {model_path_to_check}: {str(e)}")
            else:
                logger.warning(f"Model file at {model_path_to_check} is empty (0 bytes)")
        else:
            logger.warning("No model file found at expected locations")
        
        # Check metadata file
        metadata = None
        if metadata_path_to_check:
            validation_results["file_validation"]["metadata_file_exists"] = True
            
            # Check if not empty
            file_size = os.path.getsize(metadata_path_to_check)
            validation_results["file_validation"]["metadata_file_not_empty"] = file_size > 0
            
            if validation_results["file_validation"]["metadata_file_not_empty"]:
                # Try to load the metadata
                try:
                    with open(metadata_path_to_check, 'r') as f:
                        metadata = json.load(f)
                    
                    validation_results["file_validation"]["metadata_parsable"] = True
                    
                    # Further metadata validation will happen in step 4
                
                except Exception as e:
                    logger.warning(f"Cannot parse metadata file at {metadata_path_to_check}: {str(e)}")
            else:
                logger.warning(f"Metadata file at {metadata_path_to_check} is empty (0 bytes)")
        else:
            logger.warning("No metadata file found at expected locations")
        
        # Check for log file
        log_files = [f for f in os.listdir(output_dir) if f.endswith('.log')]
        
        if log_files:
            validation_results["file_validation"]["log_file_exists"] = True
            
            # Check if not empty
            latest_log = sorted([os.path.join(output_dir, f) for f in log_files], 
                            key=os.path.getmtime, reverse=True)[0]
            file_size = os.path.getsize(latest_log)
            validation_results["file_validation"]["log_file_not_empty"] = file_size > 0
            
            if not validation_results["file_validation"]["log_file_not_empty"]:
                logger.warning(f"Log file at {latest_log} is empty (0 bytes)")
        else:
            validation_results["file_validation"]["log_file_exists"] = False
            logger.error(f"No log files found in output directory: {output_dir}")
            
            # Create a minimal fallback log file
            try:
                fallback_log = os.path.join(output_dir, "validation_fallback.log")
                with open(fallback_log, 'w') as f:
                    f.write(f"Validation fallback log created at {datetime.now().isoformat()}\n")
                    f.write(f"No log files were found during validation.\n")
                    f.write("This may indicate that the custom pipeline didn't set up logging properly.\n")
                logger.info(f"Created fallback log file: {fallback_log}")
            except Exception as e:
                logger.error(f"Error creating fallback log: {str(e)}")

    except Exception as e:
        logger.error(f"Error validating output files: {str(e)}")
    
    # 4. Validate metadata content
    try:
        if validation_results["file_validation"]["metadata_parsable"] and metadata:
            logger.debug("Validating metadata content")
            
            # Check required fields
            required_fields = ['timestamp', 'model_id', 'target', 'status']
            has_required_fields = all(field in metadata for field in required_fields)
            validation_results["metadata_validation"]["required_fields_present"] = has_required_fields
            
            if not has_required_fields:
                missing = [f for f in required_fields if f not in metadata]
                logger.warning(f"Metadata is missing required fields: {missing}")
            
            # Check if status is "completed"
            if 'status' in metadata:
                validation_results["metadata_validation"]["status_completed"] = metadata['status'] == 'completed'
                
                if not validation_results["metadata_validation"]["status_completed"]:
                    logger.warning(f"Metadata status is '{metadata.get('status')}', expected 'completed'")
            
            # Check best model information
            if 'best_model' in metadata:
                best_model = metadata['best_model']
                
                has_best_model_info = (
                    isinstance(best_model, dict) and
                    'name' in best_model
                )
                
                validation_results["metadata_validation"]["best_model_info_present"] = has_best_model_info
                
                if not has_best_model_info:
                    logger.warning("Metadata is missing best_model information")
                    
                # Check test metrics
                if 'metrics' in best_model and isinstance(best_model['metrics'], dict):
                    metrics = best_model['metrics']
                    
                    has_test_metrics = any(
                        key.startswith('test_') for key in metrics.keys()
                    )
                    
                    validation_results["metadata_validation"]["test_metrics_present"] = has_test_metrics
                    
                    if not has_test_metrics:
                        logger.warning("Metadata is missing test metrics in best_model.metrics")
                else:
                    logger.warning("Metadata best_model is missing metrics dictionary")
            else:
                logger.warning("Metadata is missing best_model section")
            
            # Check preprocessing information
            if 'preprocessing' in metadata:
                preproc = metadata['preprocessing']
                
                has_preprocessing_info = (
                    isinstance(preproc, dict) and
                    ('numeric_features' in preproc or 'categorical_features' in preproc)
                )
                
                validation_results["metadata_validation"]["preprocessing_info_present"] = has_preprocessing_info
                validation_results["metadata_validation"]["feature_types_defined"] = has_preprocessing_info
                
                if not has_preprocessing_info:
                    logger.warning("Metadata is missing essential preprocessing information")
            else:
                # Check alternative locations for feature information
                alt_locations = [
                    ('data', 'columns'),
                    ('feature_importance', None)
                ]
                
                found_alt = False
                for section, field in alt_locations:
                    if section in metadata:
                        if field is None or field in metadata[section]:
                            found_alt = True
                            logger.debug(f"Found alternative feature information in metadata.{section}")
                            break
                
                if found_alt:
                    validation_results["metadata_validation"]["preprocessing_info_present"] = True
                else:
                    logger.warning("Metadata is missing preprocessing information")
            
            # CRITICAL: Check problem_type - needed for testing pipeline
            problem_type = None
            
            # Check in parameters
            if 'parameters' in metadata and 'problem_type' in metadata['parameters']:
                problem_type = metadata['parameters']['problem_type']
                logger.debug(f"Found problem_type '{problem_type}' in parameters section")
            
            # Check in best_model
            elif 'best_model' in metadata and 'problem_type' in metadata['best_model']:
                problem_type = metadata['best_model']['problem_type']
                logger.debug(f"Found problem_type '{problem_type}' in best_model section")
                
            validation_results["metadata_validation"]["problem_type_present"] = problem_type is not None
            
            if not validation_results["metadata_validation"]["problem_type_present"]:
                error_msg = "Metadata is missing critical field 'problem_type' - required for testing pipeline"
                logger.error(error_msg)
                if raise_errors:
                    raise ValueError(error_msg)
            
            # CRITICAL: Check target column - needed for testing pipeline
            target_column = None
            
            # Check in parameters
            if 'parameters' in metadata and 'target_column' in metadata['parameters']:
                target_column = metadata['parameters']['target_column']
            # Check in target field directly
            elif 'target' in metadata:
                target_column = metadata['target']
            # Check in preprocessing
            elif 'preprocessing' in metadata and 'target_column' in metadata['preprocessing']:
                target_column = metadata['preprocessing']['target_column']
                
            validation_results["metadata_validation"]["target_column_specified"] = target_column is not None
            
            if not validation_results["metadata_validation"]["target_column_specified"]:
                error_msg = "Metadata is missing target column specification - required for testing"
                logger.warning(error_msg)
                if raise_errors:
                    raise ValueError(error_msg)
    
    except Exception as e:
        logger.error(f"Error validating metadata content: {str(e)}")
        if raise_errors and "Metadata is missing critical field" in str(e):
            raise
    
    # Calculate overall validity
    section_validities = {}
    for section, checks in validation_results.items():
        section_validities[section] = all(checks.values())
    
    is_valid = all(section_validities.values())
    
    # Log summary results
    validation_time = (datetime.now() - validation_start).total_seconds()
    
    if is_valid:
        logger.info(f"Custom pipeline validation passed in {validation_time:.2f} seconds")
    else:
        logger.warning(f"Custom pipeline validation found issues (completed in {validation_time:.2f} seconds):")
        for section, valid in section_validities.items():
            if not valid:
                failed_checks = [check for check, passed in validation_results[section].items() if not passed]
                logger.warning(f"  - {section}: Failed checks: {failed_checks}")
                
        # Special error for critical fields missing
        if (not validation_results["metadata_validation"]["problem_type_present"] or 
            not validation_results["metadata_validation"]["target_column_specified"]) and raise_errors:
            raise ValueError("Metadata is missing critical fields required for testing pipeline")
    
    return is_valid, validation_results, metadata
# 