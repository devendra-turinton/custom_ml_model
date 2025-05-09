"""
View layer for model training API endpoints.
"""
import os
import json
import logging
from flask import request, jsonify, make_response, send_file
from flask.views import MethodView
from typing import Dict, Any, Tuple

from custom_ml_api.services.model_service import ModelTrainingService
from custom_ml_api.utils.error_handler import handle_exception

logger = logging.getLogger(__name__)


class ModelTrainingView(MethodView):
    """
    View for model training operations.
    Implements the custom_train_model endpoint.
    """
    def post(self) -> Tuple[Dict[str, Any], int]:
        """
        Handle POST request to train a model.
        
        Expected input: JSON payload with model_id and optional target
        
        Returns:
            Tuple of (response_dict, status_code)
        """
        try:
            logger.info("Received POST request to /api/custom_train_model")
            
            # Validate request content type
            if not request.is_json:
                return jsonify({
                    "error": True,
                    "message": "Request must be JSON"
                }), 400
            
            # Parse request data
            data = request.get_json()
            
            # Validate required fields
            if not data or 'model_id' not in data:
                return jsonify({
                    "error": True,
                    "message": "Missing required field: model_id"
                }), 400
            
            model_id = data['model_id']
            # Get target from payload or use default
            target = data.get('target', 'target')
            
            logger.info(f"Processing model training request for model_id: {model_id}, target: {target}")
            
            # Validate model_id format
            model_service = ModelTrainingService()
            model_service.validate_model_id(model_id)
            
            # Start training synchronously with the provided target
            logger.info(f"Starting training for model_id: {model_id}, target: {target}")
            result = model_service.train_model(model_id, target)
            logger.info(f"Training completed for model_id: {model_id}")
            
            # Return results
            response = {
                "success": True,
                "message": f"Model training completed for model_id: {model_id}",
                "model_id": model_id,
                "target": target,
                "result": result,
                "status": "completed"
            }

            if "custom_validation" in result:
                response["custom_validation"] = result["custom_validation"]
            
            return jsonify(response), 200
            
        except Exception as e:
            error_response, status_code = handle_exception(e)
            return jsonify(error_response), status_code


class ModelFilesView(MethodView):
    """View for retrieving model files."""
    
    def get(self, model_id: str, file_type: str) -> Any:
        """
        Handle GET request to download model files.
        
        Args:
            model_id: The model ID
            file_type: The type of file to download (model, metadata, log)
            
        Returns:
            File response or error
        """
        try:
            logger.info(f"Downloading {file_type} file for model_id: {model_id}")
            
            if not model_id or not file_type:
                return jsonify({
                    "error": True,
                    "message": "Missing model_id or file_type parameter"
                }), 400
            
            # Validate file_type
            valid_file_types = ['model', 'metadata', 'log']
            if file_type not in valid_file_types:
                return jsonify({
                    "error": True,
                    "message": f"Invalid file_type. Must be one of: {', '.join(valid_file_types)}"
                }), 400
            
            # Get file paths
            model_service = ModelTrainingService()
            file_paths = model_service.get_model_files(model_id)
            
            # Get the requested file path
            file_path = file_paths.get(file_type)
            
            if not file_path or not os.path.exists(file_path):
                return jsonify({
                    "error": True,
                    "message": f"File not found for model_id: {model_id}, file_type: {file_type}"
                }), 404
            
            # Set content type based on file_type
            content_type = {
                'model': 'application/octet-stream',
                'metadata': 'application/json',
                'log': 'text/plain'
            }[file_type]
            
            # Get filename
            filename = os.path.basename(file_path)
            
            # Return the file
            return send_file(
                file_path,
                mimetype=content_type,
                as_attachment=True,
                download_name=filename
            )
            
        except Exception as e:
            error_response, status_code = handle_exception(e)
            return jsonify(error_response), status_code