"""
Container manager for handling container pools and model isolation.
"""
import os
import io
import tarfile
import tempfile
import time
import uuid
import logging
import subprocess
import docker
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ContainerPoolManager:
    """Manages pools of containers for model training."""
    
    def __init__(self, base_image="python:3.12-slim", max_containers=5):
        """
        Initialize the container pool manager.
        
        Args:
            base_image: Docker base image to use
            max_containers: Maximum number of containers in the pool
        """
        self.base_image = base_image
        self.max_containers = max_containers
        self.client = docker.from_env()
        self.containers = {}  # container_id -> container info
        self.model_assignments = {}  # model_id -> container_id
        
        # Ensure the base image is available
        try:
            self.client.images.get(base_image)
            logger.info(f"Base image {base_image} is available")
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling base image {base_image}...")
            self.client.images.pull(base_image)
    
    def get_container(self, model_requirements: Dict[str, Any]) -> str:
        """
        Get a suitable container for the given model requirements.
        
        Args:
            model_requirements: Dictionary of model requirements
            
        Returns:
            container_id: ID of suitable container
        """
        # For now, simply use the least loaded container or create a new one
        available_containers = self._get_available_containers()
        
        if available_containers:
            container_id = available_containers[0]
            logger.info(f"Using existing container {container_id}")
            return container_id
        
        # Create a new container if needed
        if len(self.containers) < self.max_containers:
            return self._create_container()
            
        # If we're at capacity, return the container with the fewest models
        container_loads = {}
        for model_id, container_id in self.model_assignments.items():
            container_loads[container_id] = container_loads.get(container_id, 0) + 1
        
        least_loaded = min(container_loads.items(), key=lambda x: x[1])[0]
        logger.info(f"Using least loaded container {least_loaded}")
        return least_loaded
    
    def _create_container(self) -> str:
        """Create a new container and return its ID."""
        container_name = f"model-container-{uuid.uuid4().hex[:8]}"
        
        # Mount the models directory
        volumes = {
            os.path.abspath('data'): {'bind': '/app/data', 'mode': 'rw'}
        }
        
        # Create the container
        container = self.client.containers.run(
            self.base_image,
            name=container_name,
            command="tail -f /dev/null",  # Keep container running
            detach=True,
            volumes=volumes,
            working_dir="/app",
            network_mode="bridge",
            environment={"PYTHONUNBUFFERED": "1"}
        )
        
        container_id = container.id
        logger.info(f"Created new container {container_id} ({container_name})")
        
        # Install common dependencies
        self._install_common_requirements(container_id)
        
        # Store container info
        self.containers[container_id] = {
            'name': container_name,
            'container': container,
            'models': []
        }
        
        return container_id
    
    def _install_common_requirements(self, container_id: str) -> None:
        """Install common requirements in the container."""
        # Install basic dependencies
        self.execute_in_container(
            container_id,
            "pip install --no-cache-dir numpy pandas scikit-learn Flask"
        )
    
    def _get_available_containers(self) -> List[str]:
        """Get IDs of available containers."""
        # For simplicity, consider all running containers as available
        return list(self.containers.keys())
    
    def assign_model(self, container_id: str, model_id: str) -> None:
        """Assign a model to a container."""
        self.model_assignments[model_id] = container_id
        if model_id not in self.containers[container_id]['models']:
            self.containers[container_id]['models'].append(model_id)
        logger.info(f"Assigned model {model_id} to container {container_id}")
    
    def create_model_isolation(self, container_id: str, model_id: str) -> 'ModelIsolation':
        """Create isolation for a model within a container."""
        return ModelIsolation(self, container_id, model_id)
    
    def execute_in_container(self, container_id: str, command: str) -> str:
        """Execute a command in the container and return output."""
        container = self.client.containers.get(container_id)
        
        # If command already includes bash -c, don't wrap it again
        if not command.startswith("/bin/bash -c") and ("source" in command or "&&" in command):
            # Wrap in bash for more complex commands
            command = f"/bin/bash -c '{command}'"
        
        logger.debug(f"Executing in container {container_id}: {command}")
        
        result = container.exec_run(command)
        
        if result.exit_code != 0:
            logger.error(f"Command failed in container {container_id}: {result.output.decode()}")
            raise RuntimeError(f"Command failed: {result.output.decode()}")
            
        return result.output.decode()
    
    def copy_to_container(self, container_id: str, src_path: str, dest_path: str) -> None:
        """Copy a file or directory to the container."""
        container = self.client.containers.get(container_id)
        
        try:
            # Ensure the destination directory exists
            dir_path = os.path.dirname(dest_path)
            container.exec_run(f"mkdir -p {dir_path}")
            
            # Read the source file
            with open(src_path, 'rb') as src_file:
                data = src_file.read()
            
            # Use container.put_archive instead of subprocess
            tar_data = self._create_tar_archive(os.path.basename(dest_path), data)
            container.put_archive(os.path.dirname(dest_path), tar_data)
            
            logger.debug(f"Copied {src_path} to {dest_path} in container {container_id}")
        except Exception as e:
            logger.error(f"Error copying to container: {str(e)}")
            raise RuntimeError(f"Failed to copy to container: {str(e)}")

    def copy_from_container(self, container_id: str, src_path: str, dest_path: str) -> None:
        """Copy a file or directory from the container."""
        container = self.client.containers.get(container_id)
        
        try:
            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Use container.get_archive instead of subprocess
            bits, stat = container.get_archive(src_path)
            
            # Write the tar archive to a temporary file
            with tempfile.NamedTemporaryFile() as tmp:
                for chunk in bits:
                    tmp.write(chunk)
                tmp.flush()
                
                # Extract the tar archive
                with tarfile.open(tmp.name) as tar:
                    tar.extractall(path=os.path.dirname(dest_path))
            
            logger.debug(f"Copied {src_path} from container {container_id} to {dest_path}")
        except Exception as e:
            logger.error(f"Error copying from container: {str(e)}")
            raise RuntimeError(f"Failed to copy from container: {str(e)}")

    # Add this helper method
    def _create_tar_archive(self, file_name, file_data):
        """Create a tar archive from file data."""
        import io
        import tarfile
        
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode='w') as tar:
            tarinfo = tarfile.TarInfo(name=file_name)
            tarinfo.size = len(file_data)
            tarinfo.mtime = time.time()
            tar.addfile(tarinfo, io.BytesIO(file_data))
        
        tar_stream.seek(0)
        return tar_stream.read()

    def get_model_directory(self, container_id: str, model_id: str) -> str:
        """Get the path to a model's directory in the container."""
        return f"/app/data/models/{model_id}"
    
    def cleanup(self) -> None:
        """Clean up all containers."""
        for container_id, info in self.containers.items():
            try:
                container = info['container']
                container.stop()
                container.remove()
                logger.info(f"Removed container {container_id}")
            except Exception as e:
                logger.error(f"Error cleaning up container {container_id}: {str(e)}")


class ModelIsolation:
    """Provides isolation for a model within a container."""
    
    def __init__(self, pool_manager: ContainerPoolManager, container_id: str, model_id: str):
        """
        Initialize model isolation.
        
        Args:
            pool_manager: Container pool manager
            container_id: Container ID
            model_id: Model ID
        """
        self.pool_manager = pool_manager
        self.container_id = container_id
        self.model_id = model_id
        self.venv_path = f"/app/data/models/{model_id}/venv"
        self.model_dir = pool_manager.get_model_directory(container_id, model_id)
        self._create_environment()
    
    def _create_environment(self) -> None:
        """Create an isolated environment for the model."""
        # Create model directory first
        self.pool_manager.execute_in_container(
            self.container_id,
            f"mkdir -p {self.model_dir}"
        )
        
        # Then create virtual environment in a separate command
        try:
            self.pool_manager.execute_in_container(
                self.container_id,
                f"python -m venv {self.venv_path}"
            )
            logger.info(f"Created virtual environment for model {self.model_id} in container {self.container_id}")
        except Exception as e:
            # Fallback if python venv module fails
            logger.warning(f"Failed to create venv, trying with virtualenv: {str(e)}")
            try:
                # Install virtualenv if needed
                self.pool_manager.execute_in_container(
                    self.container_id,
                    "pip install virtualenv"
                )
                # Create environment with virtualenv
                self.pool_manager.execute_in_container(
                    self.container_id,
                    f"virtualenv {self.venv_path}"
                )
                logger.info(f"Created virtualenv for model {self.model_id}")
            except Exception as fallback_error:
                logger.error(f"Could not create isolated environment: {str(fallback_error)}")
                # Create a minimal isolation using just a directory
                self.pool_manager.execute_in_container(
                    self.container_id,
                    f"mkdir -p {self.venv_path}/bin"
                )
                # Create a dummy activate script
                self.pool_manager.execute_in_container(
                    self.container_id,
                    f"echo '# Dummy activation' > {self.venv_path}/bin/activate"
                )
                logger.warning(f"Created minimal isolation directory structure")
    
    def install_requirements(self, requirements_path: str) -> None:
        """Install requirements in the isolated environment."""
        # Copy requirements file to container if it's a local path
        if os.path.exists(requirements_path):
            container_req_path = f"{self.model_dir}/requirements.txt"
            self.pool_manager.copy_to_container(
                self.container_id, 
                requirements_path,
                container_req_path
            )
        else:
            container_req_path = requirements_path
        
        # First install build dependencies
        self.pool_manager.execute_in_container(
            self.container_id,
            f"/bin/bash -c 'source {self.venv_path}/bin/activate && " 
            f"pip install --upgrade pip wheel setuptools && "
            f"apt-get update && apt-get install -y --no-install-recommends "
            f"build-essential gcc g++ gfortran libatlas-base-dev'"
        )
        
        # Then install requirements with extra pip options for compatibility
        self.pool_manager.execute_in_container(
            self.container_id,
            f"/bin/bash -c 'source {self.venv_path}/bin/activate && "
            f"pip install --no-cache-dir -r {container_req_path} "
            f"--use-deprecated=legacy-resolver'"
        )
        logger.info(f"Installed requirements for model {self.model_id} in container {self.container_id}")

    def execute_command(self, command: str) -> str:
        """Execute a command in the isolated environment."""
        # If the command uses shell features, wrap it in bash
        shell_features = ['>', '<', '|', ';', '&&', '||', '*', '?', '{', '}', '[', ']', '$', '`', '(', ')', '#']
        needs_shell = any(feature in command for feature in shell_features)
        
        if needs_shell:
            # Escape single quotes in the command
            escaped_command = command.replace("'", "'\\''")
            # Wrap in bash
            full_command = f"/bin/bash -c '{escaped_command}'"
        else:
            # Source the virtual env activation first
            if self.venv_path and not command.startswith("/bin/bash"):
                full_command = f"/bin/bash -c 'source {self.venv_path}/bin/activate && {command}'"
            else:
                full_command = command
                
        return self.pool_manager.execute_in_container(self.container_id, full_command)

    def cleanup(self) -> None:
        """Clean up the isolated environment."""
        try:
            # Just remove the virtual environment
            self.pool_manager.execute_in_container(
                self.container_id,
                f"rm -rf {self.venv_path}"
            )
            logger.info(f"Cleaned up environment for model {self.model_id} in container {self.container_id}")
        except Exception as e:
            logger.error(f"Error cleaning up environment: {str(e)}")