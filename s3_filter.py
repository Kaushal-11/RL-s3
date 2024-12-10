import boto3
import os
import json
from dotenv import load_dotenv
import random
import re

# Load environment variables from .env
load_dotenv()

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
AWS_REGION = os.getenv('AWS_REGION')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# Define local paths
recordings_dir = "./s3_recordings"  # For downloaded recordings
filtered_recordings_dir = "./s3_filter_rec"  # For filtered recordings

# Initialize S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

def sanitize_filename(filename):
    """
    Replace invalid characters in a filename with underscores.
    """
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def download_from_s3(bucket_name, prefix):
    """
    Download all files from the specified S3 bucket and prefix to the local recordings directory.
    """
    print(f"Downloading files from S3 bucket '{bucket_name}' with prefix '{prefix}'...")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    
    if 'Contents' not in response:
        print(f"No objects found in S3 bucket with prefix: {prefix}")
        return
    
    for obj in response['Contents']:
        key = obj['Key']
        if key.endswith(".json"):  # Only process JSON files
            sanitized_filename = sanitize_filename(os.path.basename(key))  # Sanitize file name
            local_file_path = os.path.join(recordings_dir, sanitized_filename)
            print(f"Downloading: {key} to {local_file_path}")
            s3.download_file(bucket_name, key, local_file_path)
    print("All files downloaded successfully.")


def filter_rrweb_data(events, colors, fonts):
    """
    Filter rrweb events and apply custom styles.
    """
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
    """
    Process all recordings in the local directory and save filtered versions.
    """
    for filename in os.listdir(recordings_dir):
        if filename.endswith(".json"):
            raw_file_path = os.path.join(recordings_dir, filename)
            filtered_file_path = os.path.join(filtered_recordings_dir, filename.replace(".json", "-filtered.json"))

            with open(raw_file_path, 'r') as raw_file:
                raw_data = json.load(raw_file)
                events = raw_data.get("events", [])
                colors = raw_data.get("colors", {})
                fonts = raw_data.get("font-family", {})

            filtered_events = filter_rrweb_data(events, colors, fonts)

            with open(filtered_file_path, 'w') as filtered_file:
                json.dump(filtered_events, filtered_file, indent=2)
            print(f"Filtered recording saved: {filtered_file_path}")

# Ensure local directories exist
os.makedirs(recordings_dir, exist_ok=True)
os.makedirs(filtered_recordings_dir, exist_ok=True)

if __name__ == "__main__":
    # Step 1: Download recordings from S3
    download_from_s3(S3_BUCKET_NAME, "events/")  # Updated prefix for S3 directory

    # Step 2: Filter all recordings
    filter_all_recordings()