"""
View layer for model testing API endpoints.
"""
import os
import json
import logging
from flask import request, jsonify, send_file
from flask.views import MethodView
from typing import Dict, Any, Tuple

from ml_api.services.model_testing_service import ModelTestingService
from ml_api.utils.error_handler import handle_exception, TestingError

logger = logging.getLogger(__name__)


class ModelTestingView(MethodView):
    """
    View for model testing operations.
    Implements the custom_test_model endpoint.
    """
    
    def post(self) -> Tuple[Dict[str, Any], int]:
        """
        Handle POST request to test a model.
        
        Expected input: JSON payload with a single key: model_id
        
        Returns:
            Tuple of (response_dict, status_code)
        """
        try:
            logger.info("Received POST request to /api/custom_test_model")
            
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
            logger.info(f"Processing model testing request for model_id: {model_id}")
            
            # Validate model_id format
            model_service = ModelTestingService()
            model_service.validate_model_id(model_id)
            
            # Test the model
            logger.info(f"Starting testing for model_id: {model_id}")
            result = model_service.test_model(model_id)
            logger.info(f"Testing completed for model_id: {model_id}")
            
            # Return results
            response = {
                "success": True,
                "message": f"Model testing completed for model_id: {model_id}",
                "model_id": model_id,
                "result": result,
                "status": "completed"
            }
            
            return jsonify(response), 200
            
        except Exception as e:
            error_response, status_code = handle_exception(e)
            return jsonify(error_response), status_code


class TestResultFilesView(MethodView):
    """View for retrieving test result files."""
    
    def get(self, model_id: str, file_type: str) -> Any:
        """
        Handle GET request to download test result files.
        
        Args:
            model_id: The model ID
            file_type: The type of file to download (results, metadata, log)
            
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
            valid_file_types = ['results', 'metadata', 'log']
            if file_type not in valid_file_types:
                return jsonify({
                    "error": True,
                    "message": f"Invalid file_type. Must be one of: {', '.join(valid_file_types)}"
                }), 400
            
            # Get version from query parameter (optional)
            version = request.args.get('version', None)
            
            # Get file paths
            model_service = ModelTestingService()
            try:
                file_paths = model_service.get_test_files(model_id, version)
            except Exception as e:
                return jsonify({
                    "error": True,
                    "message": str(e)
                }), 404
            
            # Get the requested file path
            file_path = file_paths.get(file_type)
            
            if not file_path or not os.path.exists(file_path):
                return jsonify({
                    "error": True,
                    "message": f"File not found for model_id: {model_id}, file_type: {file_type}"
                }), 404
            
            # Set content type based on file_type
            content_type = {
                'results': 'text/csv',
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