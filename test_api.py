import requests
import json
import time
import sys

# API endpoint
API_URL = "http://localhost:5000/api/custom_train_model"

def train_model(model_id, target="target"):
    """Send a training request to the API."""
    print(f"Requesting training for model {model_id}...")
    
    payload = {
        "model_id": model_id,
        "target": target
    }
    
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Training initiated for {model_id}:")
        print(json.dumps(result, indent=2))
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def check_model_status(model_id):
    """Check the status of a model by trying to get its metadata."""
    url = f"http://localhost:5000/api/model_files/{model_id}/metadata"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Model {model_id} training completed successfully!")
            return True
        else:
            print(f"Model {model_id} not ready yet (status: {response.status_code})")
            return False
    except Exception as e:
        print(f"Error checking status: {str(e)}")
        return False

# Add this to test_api.py:

def wait_for_completion(model_id, timeout=60):  # Reduce timeout to 60 seconds
    """Wait for model training to complete."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if check_model_status(model_id):
            return True
        print("Waiting 5 seconds...")
        time.sleep(5)
    
    print(f"Timeout waiting for model {model_id} - checking what files were created:")
    
    # Check what files actually exist
    try:
        output_dir = f"data/training/output/{model_id}"
        if os.path.exists(output_dir):
            print(f"Directory exists: {output_dir}")
            versions = [d for d in os.listdir(output_dir) if d.startswith('v')]
            if versions:
                latest_version = sorted(versions, key=lambda x: int(x[1:]))[-1]
                version_dir = os.path.join(output_dir, latest_version)
                print(f"Latest version directory: {version_dir}")
                
                if os.path.exists(version_dir):
                    files = os.listdir(version_dir)
                    print(f"Files found: {files}")
                    
                    # Check if there's a v1 subdirectory
                    v1_dir = os.path.join(version_dir, "v1")
                    if os.path.exists(v1_dir):
                        print(f"Found v1 subdirectory with files: {os.listdir(v1_dir)}")
    except Exception as e:
        print(f"Error checking files: {str(e)}")
    
    return False

def main():
    """Main function to test model training."""
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    else:
        model_id = "test-model-1"
    
    # Start training
    result = train_model(model_id)
    
    if result:
        # Wait for completion
        wait_for_completion(model_id)
        
        # Try to get logs
        log_url = f"http://localhost:5000/api/model_files/{model_id}/log"
        log_response = requests.get(log_url)
        
        if log_response.status_code == 200:
            print("\nTraining logs:")
            print("-" * 50)
            print(log_response.text[:500])  # Print just the first 500 chars
            print("-" * 50)
        
        # Try to get metadata
        metadata_url = f"http://localhost:5000/api/model_files/{model_id}/metadata"
        metadata_response = requests.get(metadata_url)
        
        if metadata_response.status_code == 200:
            print("\nModel metadata:")
            print("-" * 50)
            print(json.dumps(metadata_response.json(), indent=2))
            print("-" * 50)

if __name__ == "__main__":
    main()
