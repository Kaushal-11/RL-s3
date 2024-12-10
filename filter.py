import os
import json
import random

# Define paths
recordings_dir = "../Backend/recordings"
filtered_recordings_dir = "./filtered_recordings"

def filter_rrweb_data(events, colors, fonts):
    filtered_events = []

    def process_node(node, elements):
        attributes = node.get("attributes", {})
        class_list = attributes.get("class", "").split()  # Split class names into a list

        if node.get("tagName") == "nav":
            elements.append({
                "type": "navbar",
                "id": node.get("id") or random.randint(200, 1200),
                "attributes": {
                    "style": {
                        "background-color": colors.get("navbar", "rgb(0, 0, 255)"),
                        "color": "rgb(255, 255, 255)",  # Default text color
                        "font-family": fonts.get("navbar", "Arial, sans-serif")  # Default font family
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

# Filter all recordings
def filter_all_recordings():
    for filename in os.listdir(recordings_dir):
        if filename.endswith(".json"):
            raw_file_path = os.path.join(recordings_dir, filename)
            filtered_file_path = os.path.join(filtered_recordings_dir, filename.replace(".json", "-filtered.json"))

            # Load raw data
            with open(raw_file_path, 'r') as raw_file:
                raw_data = json.load(raw_file)
                events = raw_data.get("events", [])
                colors = raw_data.get("colors", {})
                fonts = raw_data.get("font-family", {})

            # Filter data
            filtered_events = filter_rrweb_data(events, colors, fonts)

            # Save filtered data
            with open(filtered_file_path, 'w') as filtered_file:
                json.dump(filtered_events, filtered_file, indent=2)
            print(f"Filtered recording saved: {filtered_file_path}")

# Ensure the filtered recordings directory exists
os.makedirs(filtered_recordings_dir, exist_ok=True)

# Run the filtering process
if __name__ == "__main__":
    filter_all_recordings()