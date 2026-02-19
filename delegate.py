import os
import time
import socket
from analysis import azar_surface_analysis

RESOLUTION = 2000
TCP_PORT = 5000
TCP_HOST = 'localhost'
WATCH_PATH = '/path/to/watch'

def analyze_file(filepath):
    with open(filepath, 'r') as f:
        data = f.read()
        HD, AH, SS = azar_surface_analysis(data, resolution=RESOLUTION, pixel_width=1e-6, rot90_value=0, figure_on=True)
        return HD, AH, SS

def send_tcp(data, host=TCP_HOST, port=TCP_PORT):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            s.sendall(f"{data[0]},{data[1]},{data[2]}\n".encode())
    except Exception as e:
        print(f"TCP error: {e}")

# Track processed files
processed = set()
watch_dir = WATCH_PATH

while True:
    current_files = {f for f in os.listdir(watch_dir) if f.endswith('.txt')}
    new_files = current_files - processed
    
    for filename in new_files:
        filepath = os.path.join(watch_dir, filename)
        try:
            result = analyze_file(filepath)
            send_tcp(result)
            processed.add(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    time.sleep(1)  # Check every second