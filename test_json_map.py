import sys
import os
import requests
import json
import time

# Point script to My Thesis folder to access the map generation logic
sys.path.append(r"c:\Users\SAYAN\Desktop\My Thesis")
try:
    from ambulance_placement_V2 import load_system_from_json, generate_map
except ImportError:
    print("Could not import mapping functions from My Thesis folder.")
    sys.exit(1)

def test_json_to_map(json_source, output_html):
    """
    Downloads structural JSON from a URL (or reads a local file),
    reconstructs the system mapping, and uses folium (OpenStreetMap)
    to generate an interactive HTML map for local testing.
    """
    temp_json_path = "temp_testing_state.json"
    
    if json_source.startswith("http://") or json_source.startswith("https://"):
        print(f"Downloading JSON from: {json_source} ...")
        try:
            response = requests.get(json_source)
            response.raise_for_status()
            with open(temp_json_path, 'w') as f:
                f.write(response.text)
            print("Download successful.")
            source_file = temp_json_path
        except Exception as e:
            print(f"Failed to download JSON from URL: {e}")
            return
    else:
        print(f"Reading local JSON file: {json_source} ...")
        if not os.path.exists(json_source):
            print("Error: Local JSON file not found!")
            return
        source_file = json_source

    print("Reconstructing state from JSON format...")
    all_results = load_system_from_json(source_file)
    
    if all_results:
        print("Generating OpenStreetMap / Folium HTML map...")
        generate_map(all_results, output_html)
        print(f"✅ Success! Map saved to: {os.path.abspath(output_html)}")
    else:
        print("❌ Error: Failed to reconstruct the state from the JSON file. Ensure schema matches.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("================================================================")
        print("  TEST MAP VISUALIZATION SCRIPT")
        print("================================================================")
        print("Usage: python test_json_map.py <URL_OR_FILE_JSON> [OUTPUT_MAP_NAME.html]")
        print("\nExamples:")
        print("  python test_json_map.py \"https://your-server-json-url.json\" test_url_map.html")
        print("  python test_json_map.py local_output.json local_map.html")
    else:
        input_src = sys.argv[1]
        out_dest = sys.argv[2] if len(sys.argv) > 2 else f"test_map_output_{int(time.time())}.html"
        test_json_to_map(input_src, out_dest)
