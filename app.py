"""
Main Flask application for ML API.
"""
import os
import logging
from flask import Flask, jsonify, request

from custom_ml_api.views.model_view import (
    ModelTrainingView,
    ModelFilesView
)
from custom_ml_api.views.model_testing_view import (
    ModelTestingView,
    TestResultFilesView
)
from custom_ml_api.utils.error_handler import handle_exception, MLApiError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_api.log')
    ]
)

logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Set up routes
    app.add_url_rule(
        '/api/custom_train_model',
        view_func=ModelTrainingView.as_view('custom_train_model')
    )
 
    app.add_url_rule(
        '/api/model_files/<model_id>/<file_type>',
        view_func=ModelFilesView.as_view('model_files')
    )

    # Set up routes for testing
    app.add_url_rule(
        '/api/custom_test_model',
        view_func=ModelTestingView.as_view('custom_test_model')
    )
    
    app.add_url_rule(
        '/api/test_files/<model_id>/<file_type>',
        view_func=TestResultFilesView.as_view('test_files')
    )
    
    # Register error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "error": True,
            "message": "Endpoint not found"
        }), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            "error": True,
            "message": f"Method {request.method} not allowed for this endpoint"
        }), 405
    
    @app.errorhandler(500)
    def internal_server_error(error):
        logger.error(f"Internal server error: {str(error)}")
        return jsonify({
            "error": True,
            "message": "Internal server error"
        }), 500
    
    @app.errorhandler(MLApiError)
    def handle_ml_api_error(error):
        response, status_code = handle_exception(error)
        return jsonify(response), status_code
    
    # Add a health check endpoint
    @app.route('/health')
    def health_check():
        return jsonify({
            "status": "ok",
            "service": "ml_api"
        }), 200
    
    logger.info("Flask application initialized with API endpoints")
    return app

# Create the Flask app
app = create_app()

if __name__ == '__main__':
    # Run the app in debug mode when executed directly
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)