"""
Error handling utilities for the ML API.
"""
import logging
import traceback
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class MLApiError(Exception):
    """Base exception class for ML API errors."""
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the error to a dictionary representation for API responses."""
        result = {
            "error": True,
            "message": self.message,
            "status_code": self.status_code
        }
        if self.details:
            result["details"] = self.details
        return result


class ValidationError(MLApiError):
    """Raised when input validation fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=400, details=details)


class ResourceNotFoundError(MLApiError):
    """Raised when a requested resource is not found."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=404, details=details)


class TrainingError(MLApiError):
    """Raised when model training fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)

class TestingError(MLApiError):
    """Raised when model testing fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)

class FileValidationError(MLApiError):
    """Raised when output file validation fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


def handle_exception(e: Exception) -> Tuple[Dict[str, Any], int]:
    """
    Convert exceptions to API-friendly error responses.
    
    Args:
        e: The exception to handle
        
    Returns:
        Tuple of (error_dict, status_code)
    """
    logger.error(f"Exception occurred: {str(e)}", exc_info=True)
    
    if isinstance(e, MLApiError):
        return e.to_dict(), e.status_code
    
    # Handle unexpected exceptions
    error_detail = {
        "error": True,
        "message": "An unexpected error occurred",
        "status_code": 500,
        "details": {
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
    }
    
    # Add traceback in debug mode (should be configured based on environment)
    if logger.level <= logging.DEBUG:
        error_detail["details"]["traceback"] = traceback.format_exc()
    
    return error_detail, 500