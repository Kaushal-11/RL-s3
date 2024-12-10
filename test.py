import time
import json
import os
from stable_baselines3 import PPO
from environment import ColorEnv

# Load the environment and the model
env = ColorEnv(json_folder="../Backend/filtered_recordings")
model = PPO.load("saved_model/ppo_model")

# Reset the environment
obs, _ = env.reset()

# Directory to save new JSON files
new_json_folder = "new_files"
os.makedirs(new_json_folder, exist_ok=True)

# Set the total time to run the agent (in seconds)
file_interval = 10  # Generate a new JSON file every 10 seconds

# Function to convert RGB values to rgb(x, y, z) format
def to_rgb_format(r, g, b):
    return f"rgb({int(r)}, {int(g)}, {int(b)})"

while True:
    current_time = time.time()

    # Predict the next action using the model
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    # Convert the normalized RGB values to actual 255-scale RGB and then to `rgb(x, y, z)` format
    button_rgb = to_rgb_format(obs[0] * 255, obs[1] * 255, obs[2] * 255)
    navbar_rgb = to_rgb_format(obs[3] * 255, obs[4] * 255, obs[5] * 255)
    background_rgb = to_rgb_format(obs[6] * 255, obs[7] * 255, obs[8] * 255)

    # Create the output data in RGB format
    output_data = {
        "button_color": button_rgb,
        "navbar_color": navbar_rgb,
        "background_color": background_rgb
    }

    # Save the output data to a new JSON file with a timestamp
    filename = f"{int(current_time)}_colors.json"
    with open(os.path.join(new_json_folder, filename), 'w') as f:
        json.dump(output_data, f)

    print(f"Generated {filename}")

    # Wait for 10 seconds before generating the next file
    time.sleep(file_interval)

    # Reset the environment if the episode ends (terminated or truncated)
    if terminated or truncated:
        obs, _ = env.reset()
        terminated = False
        truncated = False

    print("Testing complete!")
