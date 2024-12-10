import numpy as np
import gymnasium as gym
import json
import os

class ColorEnv(gym.Env):
    def __init__(self, json_folder='./filtered_recordings'):
        super(ColorEnv, self).__init__()
        self.json_folder = json_folder
        self.files = os.listdir(json_folder)
        self.current_file_index = 0

        # Observation space: 18 elements (15 color values + 3 engagement metrics)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(18,), dtype=np.float32)

        # Action space: Adjust 15 color values (5 elements × 3 RGB channels)
        self.action_space = gym.spaces.Box(low=-0.05, high=0.05, shape=(15,), dtype=np.float32)

        # Historical engagement data (for reward calculation)
        self.engagement_data = {}

        # Load initial JSON
        self.seed()

    def load_json(self, file_index):
        """Load and parse a JSON file and extract color information."""
        filepath = os.path.join(self.json_folder, self.files[file_index])
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Ensure 'data' is a dictionary, if not, handle as list
        if isinstance(data, list):
            data = data[0]  # Assuming the first item in the list is relevant

        # Safeguard for missing or incorrect structure
        elements = data.get('data', {}).get('elements', [])
        button_color = [0, 0, 0]
        navbar_color = [200, 200, 200]
        background_color = [0, 0, 0]
        shepherd_header_color = [200, 200, 200]
        shepherd_button_color = [200, 200, 200]

        # Extract color information if elements are available
        for element in elements:
            style = element.get('attributes', {}).get('style', {})
            if element['type'] == 'button':
                button_color = self.extract_rgb(style.get('background-color', 'rgb(0,0,0)'))
            elif element['type'] == 'navbar':
                navbar_color = self.extract_rgb(style.get('background-color', 'rgb(0,0,0)'))
            elif element['type'] == 'background':
                background_color = self.extract_rgb(style.get('background-color', 'rgb(255,255,255)'))  # Default white
            elif element['type'] == 'shepherd-header':
                shepherd_header_color = self.extract_rgb(style.get('background-color', 'rgb(0,0,0)'))
            elif element['type'] == 'shepherd-button':
                shepherd_button_color = self.extract_rgb(style.get('background-color', 'rgb(0,0,0)'))

        # Normalize to [0, 1] and return as a flat array
        return np.array(
            button_color + navbar_color + background_color +
            shepherd_header_color + shepherd_button_color
        ) / 255.0

    def extract_rgb(self, color_str):
        """Extract RGB values from 'rgb(x,x,x)' or 'rgba(x,x,x,x)' string."""
        # Remove 'rgb(' or 'rgba(' and the closing ')'
        color_str = color_str.replace('rgba(', '').replace('rgb(', '').replace(')', '')

        # Split the color values by commas and take the first 3 values (R, G, B)
        rgb_values = color_str.split(',')[:3]  # Ignore alpha channel if present

        return [int(x.strip()) for x in rgb_values]

    def load_engagement_data(self, file_index):
        """Load historical engagement data for reward calculation (dummy implementation)."""
        return {
            'user_clicks': np.random.rand(),      # Random for now, but should be actual data
            'scroll_depth': np.random.rand(),     # Random for now, but should be actual data
            'bounce_rate': np.random.rand()       # Random for now, but should be actual data
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.current_file_index = (self.current_file_index + 1) % len(self.files)
        state = self.load_json(self.current_file_index)
        self.engagement_data = self.load_engagement_data(self.current_file_index)

        # Add engagement metrics to the observation
        engagement_metrics = [
            self.engagement_data['user_clicks'],
            self.engagement_data['scroll_depth'],
            self.engagement_data['bounce_rate']
        ]
        self.state = np.concatenate([state, engagement_metrics])

        return self.state, {}

    def calculate_reward(self, obs):
        """Calculate the reward based on the current observation."""
        reward = 0

        # Encourage brighter navbar colors
        if all(value > 0.3 for value in obs[3:6]):  # Navbar RGB > 77 (0.3 * 255)
            reward += 5
        else:
            reward -= 5  # Penalize dark colors

        # Ensure contrast between navbar and background
        navbar_avg = sum(obs[3:6]) / 3
        background_avg = sum(obs[6:9]) / 3
        if abs(navbar_avg - background_avg) < 0.2:  # Contrast too low
            reward -= 10

        # Encourage visually distinct shepherd button colors
        if all(value > 0.3 for value in obs[12:15]):
            reward += 5
        else:
            reward -= 5

        return reward

    def step(self, action):
        self.state[:15] = np.clip(self.state[:15] + action, 0, 1)

        # Compute reward
        reward = self.calculate_reward(self.state)

        # Add engagement metrics to reward
        reward += self.engagement_data['user_clicks'] * 0.5
        reward += self.engagement_data['scroll_depth'] * 0.3
        reward -= self.engagement_data['bounce_rate'] * 0.2

        # Fine-tune action reward
        reward += 0.5 - np.mean(np.abs(action))

        # Add engagement metrics to the state
        engagement_metrics = [
            self.engagement_data['user_clicks'],
            self.engagement_data['scroll_depth'],
            self.engagement_data['bounce_rate']
        ]
        self.state = np.concatenate([self.state[:15], engagement_metrics])

        terminated = np.all(np.abs(action) < 0.01)
        truncated = False
        return self.state, float(reward), terminated, truncated, {}

    def seed(self, seed=None):
        """Set the random seed for the environment."""
        np.random.seed(seed)