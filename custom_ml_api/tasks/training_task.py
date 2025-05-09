"""
Task handling for model training.

This simplified version runs training tasks synchronously without Celery.
"""
import logging
import traceback
from typing import Dict, Any
from datetime import datetime

from custom_ml_api.services.model_service import ModelTrainingService
from custom_ml_api.utils.error_handler import TrainingError

logger = logging.getLogger(__name__)

# Dictionary to track active jobs
active_jobs = {}

def train_model_task(model_id: str, target: str = 'target') -> Dict[str, Any]:
    """
    Run a model training task synchronously.
    
    Args:
        model_id: The model ID to train
        
    Returns:
        Dict with training results
    """
    job_id = f"job_{datetime.now().strftime('%Y%m%d%H%M%S')}_{model_id}"
    
    try:
        logger.info(f"Starting model training for model_id: {model_id}, target: {target}, job_id: {job_id}")

        
        # Track this job
        active_jobs[job_id] = {
            "model_id": model_id,
            "target": target,
            "status": "running",
            "started_at": datetime.now().isoformat()
        }
        
        # Train the model
        model_service = ModelTrainingService()
        result = model_service.train_model(model_id)
        
        # Update job status
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        active_jobs[job_id]["result"] = result
        
        logger.info(f"Completed model training for model_id: {model_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error in train_model_task for model_id {model_id}: {str(e)}", exc_info=True)
        
        if job_id in active_jobs:
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["error"] = str(e)
            active_jobs[job_id]["failed_at"] = datetime.now().isoformat()
        
        # Re-raise as TrainingError
        raise TrainingError(
            f"Model training failed for model_id: {model_id}",
            {"error": str(e), "traceback": traceback.format_exc()}
        )

def get_active_jobs() -> Dict[str, Dict[str, Any]]:
    """Get information about all tracked training jobs."""
    return active_jobs


def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get the status of a specific training job.
    
    Args:
        job_id: The job ID
        
    Returns:
        Dict with job status information
    """
    if job_id in active_jobs:
        return active_jobs[job_id]
    
    return {
        "job_id": job_id,
        "status": "not_found",
        "error": "Job not found"
    }