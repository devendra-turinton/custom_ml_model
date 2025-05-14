"""
Integration layer between model training service and container management.
"""
import os
import logging
import pandas as pd
import json
import tempfile
from typing import Dict, Any, Tuple

# Import the container manager
from custom_ml_api.services.container_manager import ContainerPoolManager, ModelIsolation

logger = logging.getLogger(__name__)

# Singleton container pool manager
_container_pool_manager = None

def get_container_pool_manager() -> ContainerPoolManager:
    """Get or create the container pool manager singleton."""
    global _container_pool_manager
    if _container_pool_manager is None:
        _container_pool_manager = ContainerPoolManager()
    return _container_pool_manager

def train_model_in_container(
    model_id: str, 
    target: str, 
    input_dir: str,
    output_dir: str,
    custom_flow: bool = False,
    config: Dict[str, Any] = None,
    version: str = None,  # Add this parameter
    **kwargs
) -> Tuple[Dict[str, Any], str]:
    """
    Train a model in a container.
    
    Args:
        model_id: Model ID
        target: Target column name
        input_dir: Input directory containing data
        output_dir: Output directory for model artifacts
        custom_flow: Whether to use custom flow
        version: Version string (e.g., "v1")
        config: Configuration dictionary
        
    Returns:
        Tuple of (result, log_output)
    """
    logger.info(f"Setting up container training for model {model_id}")
    
    # Get container pool manager
    pool_manager = get_container_pool_manager()
    
    # For now, we'll use simple requirements detection
    model_requirements = {
        'python_version': '3.12',
        'custom_flow': custom_flow
    }
    
    # Get a suitable container
    container_id = pool_manager.get_container(model_requirements)
    logger.info(f"Using container {container_id} for model {model_id}")
    
    # Assign model to container
    pool_manager.assign_model(container_id, model_id)
    
    # Create model directory structure in container
    model_dir = pool_manager.get_model_directory(container_id, model_id)
    container_input_dir = f"{model_dir}/input"
    container_output_dir = f"{model_dir}/output"
    
    # Create isolation for this model
    isolation = pool_manager.create_model_isolation(container_id, model_id)
    
    # Create directories in container
    isolation.execute_command(f"/bin/bash -c 'mkdir -p {container_input_dir} {container_output_dir}'")
    
    # Copy input data to container
    input_data_path = os.path.join(input_dir, "input_data.csv")
    container_data_path = f"{container_input_dir}/input_data.csv"
    pool_manager.copy_to_container(container_id, input_data_path, container_data_path)
    
    # Install common dependencies first
    isolation.execute_command("pip install numpy pandas scikit-learn")
    
    # If custom flow, copy and use custom code and requirements
    if custom_flow:
        # Copy custom code
        custom_code_path = os.path.join(input_dir, "custom_code.py")
        container_code_path = f"{model_dir}/custom_code.py"
        
        if os.path.exists(custom_code_path):
            pool_manager.copy_to_container(container_id, custom_code_path, container_code_path)
        else:
            logger.warning(f"Custom code not found at {custom_code_path}")
            return {"error": "Custom code not found"}, "Error: Custom code not found"
        
        # Install custom requirements if present
        custom_req_path = os.path.join(input_dir, "requirements.txt")
        if os.path.exists(custom_req_path):
            isolation.install_requirements(custom_req_path)
        
        # Create Python script to execute custom code
        exec_script = f"""
import os
import sys
import pandas as pd
import json
import traceback
from datetime import datetime

# Add current directory to path
sys.path.append('.')

try:
    # Load custom module
    import custom_code
    
    # Load data
    df = pd.read_csv('{container_data_path}')
    
    # Define parameters for custom function
    params = {{
        'df': df,
        'target': '{target}',
        'model_id': '{model_id}',
        'output_dir': '{container_output_dir}',
        'version': 'v1',
        'config': {{}},
    }}
    
    # Find the appropriate function
    if hasattr(custom_code, 'run_custom_pipeline'):
        func = custom_code.run_custom_pipeline
    else:
        # Try to find any callable function
        for name in dir(custom_code):
            if callable(getattr(custom_code, name)) and not name.startswith('_'):
                func = getattr(custom_code, name)
                break
    
    # Execute the custom function
    print(f"Executing custom function for model {model_id}")
    start_time = datetime.now()
    result = func(**params)
    end_time = datetime.now()
    runtime = (end_time - start_time).total_seconds()
    
    # Create the nested directory structure if needed
    os.makedirs('{container_output_dir}', exist_ok=True)
    
    # Write results 
    with open('{container_output_dir}/result.json', 'w') as f:
        # Convert result to dict if it's a tuple
        if isinstance(result, tuple):
            result_dict = {{
                'success': True,
                'runtime': runtime,
                'model_id': '{model_id}',
                'target': '{target}'
            }}
        else:
            result_dict = {{
                'success': True,
                'runtime': runtime,
                'model_id': '{model_id}',
                'target': '{target}',
                'result': str(result)
            }}
        json.dump(result_dict, f)
    
    # IMPORTANT: Also write metadata.json for API compatibility
    if isinstance(result, tuple) and len(result) >= 4:
        # If 4-tuple returned, metadata_path is the 4th element
        metadata_path = result[3]
        if os.path.exists(metadata_path) and os.path.abspath(metadata_path) != os.path.abspath('{container_output_dir}/metadata.json'):
            import shutil
            shutil.copy(metadata_path, '{container_output_dir}/metadata.json')
        elif os.path.exists(metadata_path):
            print(f"Metadata file already in correct location: {{metadata_path}}")
    else:
        # Create a basic metadata.json
        metadata = {{
            'model_id': '{model_id}',
            'target_column': '{target}',
            'problem_type': 'classification',  # Default
            'preprocessing': {{
                'target_column': '{target}',
                'numeric_features': list(df.drop(columns=['{target}']).columns),
                'categorical_features': []
            }}
        }}
        # Update with any values from result_dict
        if isinstance(result_dict, dict):
            if 'accuracy' in result_dict:
                metadata['accuracy'] = result_dict['accuracy']
            if 'problem_type' in result_dict:
                metadata['problem_type'] = result_dict['problem_type']
        
        with open('{container_output_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f)
            
    # Also save the model file if it's returned as part of the tuple
    if isinstance(result, tuple) and len(result) >= 1:
        model = result[0]
        if model is not None:
            import pickle
            with open('{container_output_dir}/{model_id}.pkl', 'wb') as f:
                pickle.dump(model, f)
    
    print(f"Custom function execution completed in {{runtime:.2f}} seconds")
    
except Exception as e:
    error = {{
        'success': False,
        'error': str(e),
        'traceback': traceback.format_exc()
    }}
    with open('{container_output_dir}/error.json', 'w') as f:
        json.dump(error, f)
    print(f"Error executing custom function: {{str(e)}}")
    print(traceback.format_exc())
"""
        
        # Write execution script to container using a temporary file approach
        exec_script_path = f"{model_dir}/exec_script.py"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
            temp_script.write(exec_script)
            temp_script_path = temp_script.name

        try:
            # Copy to container
            pool_manager.copy_to_container(container_id, temp_script_path, exec_script_path)
        finally:
            # Clean up temp file
            os.unlink(temp_script_path)
        
        # Log environment info for debugging
        env_info = isolation.execute_command("python --version && which python && echo $PATH")
        logger.info(f"Container Python environment: {env_info}")

        # Execute the script
        log_output = isolation.execute_command(f"python {exec_script_path}")
        logger.info(f"Script execution log: {log_output}")

        # Check what files were created
        ls_output = isolation.execute_command(f"/bin/bash -c 'ls -la {container_output_dir}'")
        logger.info(f"Container output directory contents: {ls_output}")

        # Try to capture any potential errors in the script execution
        error_check = isolation.execute_command(f"/bin/bash -c 'if [ -f {container_output_dir}/error.json ]; then cat {container_output_dir}/error.json; else echo \"No error file found\"; fi'")
        logger.info(f"Error file contents (if any): {error_check}")

        # Copy results back from container
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy all relevant files from container to host - use the direct paths, not nested
        files_to_copy = [
            (f"{container_output_dir}/result.json", os.path.join(output_dir, "result.json")),
            (f"{container_output_dir}/metadata.json", os.path.join(output_dir, "metadata.json")),
            (f"{container_output_dir}/error.json", os.path.join(output_dir, "error.json")),
            (f"{container_output_dir}/{model_id}.pkl", os.path.join(output_dir, f"{model_id}.pkl"))
        ]

        for src, dest in files_to_copy:
            try:
                pool_manager.copy_from_container(container_id, src, dest)
                logger.info(f"Copied {src} to {dest}")
            except Exception as e:
                logger.warning(f"Could not copy {src} to {dest}: {str(e)}")

        # Parse results
        metadata_path = os.path.join(output_dir, "metadata.json")
        result_path = os.path.join(output_dir, "result.json")
        error_path = os.path.join(output_dir, "error.json")
        
        # First check for metadata
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    result = json.load(f)
                    result['success'] = True
                logger.info(f"Found and loaded metadata.json")
            except Exception as e:
                logger.error(f"Error loading metadata.json: {str(e)}")
                result = {"success": False, "error": f"Error loading metadata: {str(e)}"}
        # Then check for result
        elif os.path.exists(result_path):
            try:
                with open(result_path, 'r') as f:
                    result = json.load(f)
                logger.info(f"Found and loaded result.json")
            except Exception as e:
                logger.error(f"Error loading result.json: {str(e)}")
                result = {"success": False, "error": f"Error loading result: {str(e)}"}
        # Finally check for error
        elif os.path.exists(error_path):
            try:
                with open(error_path, 'r') as f:
                    result = json.load(f)
                logger.info(f"Found and loaded error.json")
            except Exception as e:
                logger.error(f"Error loading error.json: {str(e)}")
                result = {"success": False, "error": f"Error loading error file: {str(e)}"}
        else:
            logger.error("No result, metadata, or error files found")
            result = {"success": False, "error": "No result, metadata, or error files found"}
        
        return result, log_output
    
    else:
        # Default flow - implement your default pipeline here
        # Create the default execution script
        exec_script = f"""
import os
import sys
import pandas as pd
import json
import traceback
from datetime import datetime
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    # Load data
    df = pd.read_csv('{container_data_path}')
    
    # Simple problem type detection
    target_series = df['{target}']
    unique_values = target_series.nunique()
    
    # Decide if classification or regression
    problem_type = 'classification' if unique_values < 10 else 'regression'
    
    # Prepare X and y
    X = df.drop(columns=['{target}'])
    y = df['{target}']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    print(f"Training a {{problem_type}} model")
    start_time = datetime.now()
    
    if problem_type == 'classification':
        model = RandomForestClassifier()
    else:
        model = RandomForestRegressor()
    
    model.fit(X_train, y_train)
    
    # Evaluate
    score = model.score(X_test, y_test)
    end_time = datetime.now()
    runtime = (end_time - start_time).total_seconds()
    
    # Save model directly in output directory (not in v1 subdirectory)
    import pickle
    os.makedirs('{container_output_dir}', exist_ok=True)
    with open('{container_output_dir}/{model_id}.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata directly in output directory
    metadata = {{
        'model_id': '{model_id}',
        'problem_type': problem_type,
        'accuracy': float(score),
        'runtime_seconds': runtime,
        'target_column': '{target}',
        'feature_count': X.shape[1],
        'sample_count': X.shape[0],
        'model_type': model.__class__.__name__,
        'preprocessing': {{
            'target_column': '{target}',
            'numeric_features': X.columns.tolist(),
            'categorical_features': []
        }}
    }}
    
    with open('{container_output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    print(f"Model trained with score: {{score:.4f}} in {{runtime:.2f}} seconds")
    
except Exception as e:
    error = {{
        'success': False,
        'error': str(e),
        'traceback': traceback.format_exc()
    }}
    os.makedirs('{container_output_dir}', exist_ok=True)
    with open('{container_output_dir}/error.json', 'w') as f:
        json.dump(error, f)
    print(f"Error training model: {{str(e)}}")
    print(traceback.format_exc())
"""
        
        # Write the execution script using the temporary file approach
        exec_script_path = f"{model_dir}/default_exec.py"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
            temp_script.write(exec_script)
            temp_script_path = temp_script.name

        try:
            # Copy to container
            pool_manager.copy_to_container(container_id, temp_script_path, exec_script_path)
        finally:
            # Clean up temp file
            os.unlink(temp_script_path)
        
        # Execute the script
        log_output = isolation.execute_command(f"python {exec_script_path}")
        logger.info(f"Default script execution log: {log_output}")
        
        # Check what files were created
        ls_output = isolation.execute_command(f"/bin/bash -c 'ls -la {container_output_dir}'")
        logger.info(f"Container output directory contents: {ls_output}")
        
        # Copy results back from container - directly to the output_dir, not to a nested v1 directory
        os.makedirs(output_dir, exist_ok=True)
        
        files_to_copy = [
            (f"{container_output_dir}/{model_id}.pkl", os.path.join(output_dir, f"{model_id}.pkl")),
            (f"{container_output_dir}/metadata.json", os.path.join(output_dir, "metadata.json")),
            (f"{container_output_dir}/error.json", os.path.join(output_dir, "error.json"))
        ]

        for src, dest in files_to_copy:
            try:
                pool_manager.copy_from_container(container_id, src, dest)
                logger.info(f"Copied {src} to {dest}")
            except Exception as e:
                logger.warning(f"Could not copy {src} to {dest}: {str(e)}")
        
        # Parse results - look directly in output_dir, not in nested v1 directory
        metadata_path = os.path.join(output_dir, "metadata.json")
        error_path = os.path.join(output_dir, "error.json")
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    result = json.load(f)
                    result['success'] = True
                logger.info(f"Found and loaded metadata.json from {metadata_path}")
            except Exception as e:
                logger.error(f"Error loading metadata.json: {str(e)}")
                result = {"success": False, "error": f"Error loading metadata: {str(e)}"}
        elif os.path.exists(error_path):
            try:
                with open(error_path, 'r') as f:
                    result = json.load(f)
                logger.info(f"Found and loaded error.json")
            except Exception as e:
                logger.error(f"Error loading error.json: {str(e)}")
                result = {"success": False, "error": f"Error loading error file: {str(e)}"}
        else:
            logger.error("No metadata or error files found")
            result = {"success": False, "error": "No result or error file found"}
        
        return result, log_output