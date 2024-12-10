import time
import os
from stable_baselines3 import PPO
from environment import ColorEnv
import subprocess  # Needed to run testing automatically

# Load or create the environment
json_folder = "../Backend/filtered_recordings"
env = ColorEnv(json_folder=json_folder)

# Load the existing model or create a new one
model_path = "saved_model/ppo_model"
if os.path.exists(model_path):
    model = PPO.load(model_path)
else:
    model = PPO("MlpPolicy", env, verbose=1)

# Training parameters
train_interval = 36  # Time interval in seconds
training_epochs = 10000

while True:
    # Train the model
    print("Training started...")
    model.learn(total_timesteps=training_epochs)
    
    # Save the model after training
    model.save(model_path)
    print("Training complete! Model saved.")

    # Automatically trigger testing after training is done
    subprocess.Popen(['python', 'test.py'])

    # Sleep for the train_interval before retraining
    time.sleep(train_interval)
