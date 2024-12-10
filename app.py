import os
import re
import json
import time
import boto3
import random
import threading
from flask_cors import CORS
from environment import ColorEnv
from stable_baselines3 import PPO
from flask import Flask, jsonify, abort
from dotenv import load_dotenv

load_dotenv()

# Flask app initialization
app = Flask(__name__)
CORS(app)

# Environment variables
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
AWS_REGION = os.getenv('AWS_REGION')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# Local directories
filtered_recordings = "./filtered_recordings"
s3_recordings_dir = "./s3_recordings"
s3_filtered_recordings_dir = "./s3_filter_rec"
new_json_folder = "./new_files/"
os.makedirs(s3_recordings_dir, exist_ok=True)
os.makedirs(s3_filtered_recordings_dir, exist_ok=True)
os.makedirs(new_json_folder, exist_ok=True)

# S3 client initialization
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# Initialize RL environment
env = ColorEnv(json_folder=filtered_recordings)

# File sanitization
def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

# S3 download logic
def download_from_s3(bucket_name, prefix):
    print(f"Downloading files from S3 bucket '{bucket_name}' with prefix '{prefix}'...")

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if 'Contents' not in response:
        print(f"No objects found in S3 bucket with prefix: {prefix}")
        return

    for obj in response['Contents']:
        key = obj['Key']

        if key.endswith(".json"):  # Only process JSON files
            # Get precise file size using head_object
            file_metadata = s3.head_object(Bucket=bucket_name, Key=key)
            size = file_metadata['ContentLength']  # File size in bytes

            if size > 256000:  # Check if the file size is greater than 250KB
                sanitized_filename = sanitize_filename(os.path.basename(key))  # Sanitize file name
                local_file_path = os.path.join(s3_recordings_dir, sanitized_filename)
                print(f"Downloading: {key} ({size} bytes) to {local_file_path}")
                s3.download_file(bucket_name, key, local_file_path)
            else:
                print(f"Deleting: {key} ({size} bytes) as it is smaller than 250KB")
                s3.delete_object(Bucket=bucket_name, Key=key)  # Delete the small file
    print("All files processed successfully.")


# rrweb filtering logic
def filter_rrweb_data(events, colors, fonts):
    filtered_events = []

    def process_node(node, elements):
        attributes = node.get("attributes", {})
        class_list = attributes.get("class", "").split()

        if node.get("tagName") == "nav":
            elements.append({
                "type": "navbar",
                "id": node.get("id") or random.randint(200, 1200),
                "attributes": {
                    "style": {
                        "background-color": colors.get("navbar", "rgb(0, 0, 255)"),
                        "color": "rgb(255, 255, 255)",
                        "font-family": fonts.get("navbar", "Arial, sans-serif")
                    }
                }
            })
        elif node.get("tagName") in ["button", "a"] and "btn" in class_list:
            if colors.get("buttons"):
                button_color = colors["buttons"].pop(0)
            else:
                button_color = {"backgroundColor": "rgb(255, 0, 0)", "color": "rgb(255, 255, 255)"}
            
            elements.append({
                "type": "button",
                "id": node.get("id") or random.randint(100, 1100),
                "attributes": {
                    "style": {
                        "background-color": button_color.get("backgroundColor", "rgb(255, 0, 0)"),
                        "color": button_color.get("color", "rgb(255, 255, 255)"),
                        "font-family": fonts.get("button", "Arial, sans-serif")  # Default button font family
                    }
                }
            })
        elif node.get("tagName") in ["p", "span", "div"]:
            elements.append({
                "type": "text",
                "id": node.get("id") or random.randint(300, 1300),
                "attributes": {
                    "style": {
                        "color": colors.get("text", "rgb(0, 0, 0)"),  # Default text color
                        "font-family": fonts.get("text", "Arial, sans-serif")  # Default text font family
                    }
                }
            })
        elif "shepherd-header" in class_list and node.get("tagName") == "header":
            elements.append({
                "type": "shepherdHeader",
                "id": node.get("id") or random.randint(400, 1400),
                "attributes": {
                    "style": {
                        "background-color": colors.get("backgroundColor", "rgb(100, 100, 100)"),
                    }
                }
            })
        elif "shepherd-button" in class_list and node.get("tagName") == "button":
            elements.append({
                "type": "shepherdButtons",
                "id": node.get("id") or random.randint(500, 1500),
                "attributes": {
                    "style": {
                        "background-color": colors.get("backgroundColor", "rgb(0, 128, 0)"),
                    }
                }
            })
        elif "shepherd-button-secondary" in class_list and node.get("tagName") == "button":
            elements.append({
                "type": "shepherdSecondaryButtons",
                "id": node.get("id") or random.randint(600, 1600),
                "attributes": {
                    "style": {
                        "background-color": colors.get("backgroundColor", "rgb(200, 200, 200)"),
                    }
                }
            })

        if "childNodes" in node:
            for child in node["childNodes"]:
                process_node(child, elements)

    for event in events:
        if event.get("type") == 2:  # Full snapshot
            timestamp = event.get("timestamp")
            elements = []

            process_node(event.get("data", {}).get("node", {}), elements)

            elements.append({
                "type": "background",
                "id": 301,
                "attributes": {
                    "style": {
                        "background-color": colors.get("background", "rgb(245, 245, 245)")
                    }
                }
            })

            if elements:
                filtered_events.append({
                    "timestamp": timestamp,
                    "data": {"elements": elements}
                })

    return filtered_events

def filter_all_recordings():
    for filename in os.listdir(s3_recordings_dir):
        if filename.endswith(".json"):
            raw_file_path = os.path.join(s3_recordings_dir, filename)
            filtered_file_path = os.path.join(s3_filtered_recordings_dir, filename.replace(".json", "-filtered.json"))
            with open(raw_file_path, 'r') as raw_file:
                raw_data = json.load(raw_file)
                events = raw_data.get("events", [])
                colors = raw_data.get("colors", {})
                fonts = raw_data.get("font-family", {})
            filtered_events = filter_rrweb_data(events, colors, fonts)
            with open(filtered_file_path, 'w') as filtered_file:
                json.dump(filtered_events, filtered_file, indent=2)
            print(f"Filtered recording saved: {filtered_file_path}")

# RL training and testing
def train_model():
    try:
        print("Starting model training...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            n_steps=4096,
            batch_size=128,
            n_epochs=20,
            learning_rate=1e-4,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1
        )
        model.learn(total_timesteps=50000)
        model.save("saved_model/ppo_model")
        print("Model training complete and saved.")
        return model
    except Exception as e:
        print(f"Error during training: {e}")
        return None

def load_or_train_model():
    model_path = "saved_model/ppo_model.zip"
    if os.path.exists(model_path):
        print("Model found, loading the model...")
        return PPO.load(model_path)
    else:
        print("Model not found, starting training...")
        return train_model()

@app.route("/run-rl", methods=["GET"])
def run_rl_service():
    try:
        print("Running RL service...")
        model = load_or_train_model()
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        output_data = {
            "button_color": [int(obs[i] * 255) for i in range(0, 3)],
            "navbar_color": [int(obs[i] * 255) for i in range(3, 6)],
            "background_color": [int(obs[i] * 255) for i in range(6, 9)],
            "shepherd_header_color": [int(obs[i] * 255) for i in range(9, 12)],
            "shepherd_button_color": [int(obs[i] * 255) for i in range(12, 15)],
        }

        # Save to `new_files` folder
        current_time = int(time.time())
        filename = f"{current_time}_colors.json"
        with open(os.path.join(new_json_folder, filename), 'w') as f:
            json.dump(output_data, f)

        return jsonify({"message": "New color scheme generated", "data": output_data})
    except Exception as e:
        print(f"Error occurred: {e}")
        abort(500, description=str(e))


def run_full_cycle():
    while True:
        print("Cycle started: Downloading and filtering at 90 minutes...")
        # Step 1: Download from S3
        download_from_s3(S3_BUCKET_NAME, "events/")
        # Step 2: Filter recordings
        filter_all_recordings()
        print("Download and filtering completed.")

        # Wait until the 100th minute to train the model
        time.sleep(10 * 60)
        print("Starting model training...")
        train_model()

        # Wait until the 120th minute to hit RL API
        time.sleep(20 * 60)
        print("Hitting RL API to generate new color scheme...")
        response = run_rl_service()
        print(f"RL API Response: {response.json()}")

# Background thread for the RL API service
def start_rl_api():
    app.run(debug=False, port=5001)

if __name__ == "__main__":
    # Start the background thread for the full cycle
    threading.Thread(target=run_full_cycle, daemon=True).start()
    
    # Run the Flask app in the main thread
    ssl_context = ('./ssl/cert.pem', './ssl/key.pem')
    app.run(host="0.0.0.0", port=5001, debug=False,ssl_context=ssl_context) 