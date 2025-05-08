# ML Model Training API

This project provides a Flask-based API for training machine learning models using the custom regression and classification pipelines.

## Features

- Train regression models via API endpoints
- Upload datasets for model training
- Track training jobs and their status
- Download trained models
- Access model metadata and performance metrics
- Secured with API key authentication

## Project Structure

```
.
├── api/                  # API implementation
│   ├── main.py           # Main Flask application
│   ├── auth.py           # Authentication middleware
│   ├── config.py         # Configuration
│   └── utils.py          # Utility functions
├── regression_pipeline.py     # Regression ML pipeline
├── classification_pipeline.py # Classification ML pipeline
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker container definition
├── docker-compose.yml    # Docker Compose configuration
└── client.py             # Example API client
```

## Installation

### With Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ml-training-api.git
   cd ml-training-api
   ```

2. Build and start the Docker container:
   ```bash
   docker-compose up -d
   ```

3. The API will be available at http://localhost:5000

### Without Docker

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ml-training-api.git
   cd ml-training-api
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the Flask application:
   ```bash
   python api/main.py
   ```

4. The API will be available at http://localhost:5000

## API Usage

### Upload a Dataset

```bash
curl -X POST \
  -F "file=@/path/to/your/dataset.csv" \
  -F "model_id=my_model" \
  http://localhost:5000/upload_dataset
```

### Train a Regression Model

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "my_model",
    "target": "target_column",
    "test_size": 0.2
  }' \
  http://localhost:5000/custom_ml_train_model/regression
```

### Check Job Status

```bash
curl -X GET http://localhost:5000/job_status/job_id_here
```

### Download Trained Model

```bash
curl -X GET -o model.pkl http://localhost:5000/models/my_model/download
```

## Example Client Usage

The included `client.py` provides a Python client for interacting with the API:

```bash
# Upload a dataset
python client.py --url http://localhost:5000 --upload /path/to/data.csv

# Train a model
python client.py --url http://localhost:5000 --model_id my_model --target price --train --wait

# Download a trained model
python client.py --url http://localhost:5000 --model_id my_model --download model.pkl
```

## Configuration

### API Configuration

Edit `api/config.py` to customize:
- API host and port
- Authentication settings
- File paths for model input/output

### Authentication

To enable API key authentication:

1. Set `AUTH_CONFIG["enabled"] = True` in `api/config.py`
2. Add your API keys to the `AUTH_CONFIG["valid_api_keys"]` list
3. Include the API key in requests with the header `X-API-Key: your-api-key-here`

## Docker Configuration

Edit `docker-compose.yml` to customize:
- Port mapping
- Volume configuration
- Environment variables

## Detailed Documentation

Refer to the [API Documentation](API_DOCUMENTATION.md) for complete details on all endpoints, request/response formats, and error handling.

## License

MIT