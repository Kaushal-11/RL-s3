from flask import Flask, jsonify, abort
from flask_cors import CORS
from stable_baselines3 import PPO
from environment import ColorEnv
import numpy as np
import json
import os
import time
import threading

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the environment
env = ColorEnv(json_folder="./filtered_recordings")

# Ensure the new files directory exists
new_json_folder = "./new_files/"
if not os.path.exists(new_json_folder):
    os.makedirs(new_json_folder)

rrweb_data_folder = os.path.abspath('./filtered_recordings')
print("Absolute path to filtered_recordings:", rrweb_data_folder)

# Function to run model training
def train_model():
    try:
        print("Starting training...")

        # Define PPO model with specified configuration
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
        model.learn(total_timesteps=50000)   # Adjust timesteps as needed
        model.save("saved_model/ppo_model")  # Save the trained model
        print("Training complete, model saved.")
        return model
    except Exception as e:
        print(f"Error during training: {e}")
        return None

# Function to test the model after training
def test_model(model):
    try:
        print("Starting testing...")
        obs, _ = env.reset()
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        output_data = {
            "button_color": [int(obs[i] * 255) for i in range(0, 3)],
            "navbar_color": [int(obs[i] * 255) for i in range(3, 6)],
            "background_color": [int(obs[i] * 255) for i in range(6, 9)],
            "shepherd_header_color": [int(obs[i] * 255) for i in range(9, 12)],
            "shepherd_button_color": [int(obs[i] * 255) for i in range(12, 15)]
        }

        return output_data
    except Exception as e:
        print(f"Error during testing: {e}")
        return None

# Check if the model exists and load it, or train if it doesn't exist
def load_or_train_model():
    model_path = "saved_model/ppo_model.zip"
    if os.path.exists(model_path):
        print("Model found, loading the model...")
        model = PPO.load(model_path)
        print("Model loaded.")
        return model
    else:
        print("Model not found, starting training...")
        return train_model()

# Background thread function to retrain the model every 2 minutes
def background_retrain_model():
    while True:
        print("Checking for model retraining...")
        model = train_model()
        time.sleep(120)  # Wait for 2 minutes before retraining again

@app.route("/run-rl", methods=["GET"])
def run_rl_service():
    try:
        print("Current working directory:", os.getcwd())
        rrweb_data = load_latest_rrweb_json(rrweb_data_folder)
        processed_data = extract_color_data_from_rrweb(rrweb_data)

        model = load_or_train_model()
        output_data = test_model(model)

        current_time = int(time.time())
        filename = f"{current_time}_colors.json"
        with open(os.path.join(new_json_folder, filename), 'w') as f:
            json.dump(output_data, f)

        return jsonify({"message": "New color scheme generated", "data": output_data})

    except Exception as e:
        print(f"Error occurred: {e}")
        abort(500, description=str(e))

def load_latest_rrweb_json(folder_path):
    try:
        files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)

        if not files:
            print("No JSON files found in the directory.")
            raise IndexError("No files found")

        latest_file = os.path.join(folder_path, files[0])
        with open(latest_file, 'r') as f:
            return json.load(f)

    except (IndexError, FileNotFoundError) as e:
        print(f"Error loading rrweb JSON: {e}")
        abort(404, description="No rrweb data files found in the directory.")

def extract_color_data_from_rrweb(rrweb_json):
    elements = rrweb_json[0]['data']['elements']

    button_color = [0, 0, 0]
    navbar_color = [0, 0, 0]
    background_color = [0, 0, 0]
    shepherd_header_color = [0, 0, 0]
    shepherd_button_color = [0, 0, 0]

    for element in elements:
        style = element.get('attributes', {}).get('style', {})
        if element['type'] == 'button':
            button_color = extract_rgb(style.get('background-color', 'rgb(0,0,0)'))
        elif element['type'] == 'navbar':
            navbar_color = extract_rgb(style.get('background-color', 'rgb(0,0,0)'))
        elif element['type'] == 'background':
            background_color = extract_rgb(style.get('background-color', 'rgb(255,255,255)'))
        elif element['type'] == 'shepherd_header':
            shepherd_header_color = extract_rgb(style.get('background-color', 'rgb(0,0,0)'))
        elif element['type'] == 'shepherd_button':
            shepherd_button_color = extract_rgb(style.get('background-color', 'rgb(0,0,0)'))

    return {
        "button_color": button_color,
        "navbar_color": navbar_color,
        "background_color": background_color,
        "shepherd_header_color": shepherd_header_color,
        "shepherd_button_color": shepherd_button_color
    }

def extract_rgb(color_str):
    color_str = color_str.replace('rgba(', '').replace('rgb(', '').replace(')', '')
    rgb_values = color_str.split(',')[:3]
    return [int(x.strip()) for x in rgb_values]

if __name__ == '__main__':
    threading.Thread(target=background_retrain_model, daemon=True).start()
    
    # Add the ssl_context parameter for HTTPS
    ssl_context = ('./ssl/cert.pem', './ssl/key.pem')
    app.run(debug=False, port=5001, ssl_context=ssl_context)