import requests
import json
import time
import threading

# API endpoint
API_URL = "http://localhost:5000/api/custom_train_model"

# Model IDs to test
MODEL_IDS = ["test-model-1", "test-model-2"]

def train_model(model_id):
    """Send a training request for a specific model."""
    print(f"Starting training for {model_id}...")
    
    payload = {
        "model_id": model_id,
        "target": "target"
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Training initiated for {model_id}: {result.get('status', 'unknown')}")
            
            # Wait and check status
            monitor_training(model_id)
            
        else:
            print(f"Failed to start training for {model_id}: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"Error training model {model_id}: {str(e)}")

def monitor_training(model_id, max_checks=20, check_interval=5):
    """Monitor training progress by checking for metadata file."""
    print(f"Monitoring training progress for {model_id}...")
    
    for i in range(max_checks):
        try:
            # Try to get metadata which should exist when training is done
            metadata_url = f"http://localhost:5000/api/model_files/{model_id}/metadata"
            response = requests.get(metadata_url)
            
            if response.status_code == 200:
                print(f"Model {model_id} training COMPLETED!")
                
                # Get log file to check sklearn version
                log_url = f"http://localhost:5000/api/model_files/{model_id}/log"
                log_response = requests.get(log_url)
                
                if log_response.status_code == 200:
                    # Look for scikit-learn version in logs
                    logs = log_response.text
                    print(f"Log snippet for {model_id}:")
                    
                    # Look for the sklearn version line
                    import re
                    version_match = re.search(r"Using scikit-learn version: ([0-9.]+)", logs)
                    if version_match:
                        version = version_match.group(1)
                        print(f"✅ Model {model_id} used scikit-learn version: {version}")
                    else:
                        print("Could not find scikit-learn version in logs")
                    
                return
            else:
                print(f"Model {model_id} still training... (attempt {i+1}/{max_checks})")
        
        except Exception as e:
            print(f"Error checking status for {model_id}: {str(e)}")
        
        time.sleep(check_interval)
    
    print(f"❌ Monitoring timed out for {model_id}")

def main():
    """Run parallel training tests."""
    print("Starting parallel model training test")
    
    # Create threads for each model
    threads = []
    for model_id in MODEL_IDS:
        thread = threading.Thread(target=train_model, args=(model_id,))
        threads.append(thread)
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print("All training jobs completed")

if __name__ == "__main__":
    main()
