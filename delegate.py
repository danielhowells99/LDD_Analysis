import os
import time
import socket
from analysis import azar_surface_analysis
import numpy as np
from pathlib import Path

RESOLUTION = 2000
TCP_PORT = 4192
TCP_HOST = "192.168.3.232"
WATCH_PATH = "E:/IPG OmniWELD Data/3D Viewer/AutomationTests_Feb2026/20260219"

def analyze_file(filepath):
    with open(filepath, 'r') as f:
        data = np.loadtxt(f,delimiter = ',')
        HD, AH, SS = azar_surface_analysis(data, resolution=RESOLUTION, pixel_width=1e-6, rot90_value=0, figure_on=False)
        return HD, AH, SS

def send_tcp(data, socket):
    try:
        socket.sendall(f"{data[0]},{data[1]},{data[2]}\r\n".encode('utf-8'))
        print(f"sending - HD:{data[0]}, AH:{data[1]}, SS:{data[2]}")
    except Exception as e:
        print(f"TCP error: {e}")


tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_socket.connect((TCP_HOST, TCP_PORT))
print("Connected to LabVIEW")

# Prompt for directory
watch_dir_input = input("Enter directory path to watch: ")

# Path handles both / and \ automatically
watch_dir = Path(watch_dir_input)

# Verify it exists
if not watch_dir.exists():
    print(f"Error: Directory '{watch_dir}' does not exist!")
    exit(1)

if not watch_dir.is_dir():
    print(f"Error: '{watch_dir}' is not a directory!")
    exit(1)

print(f"Watching: {watch_dir}")

processed = set()
i = 0
while True:
    print(f"watch loop itr: {i}")
    i = i + 1
    # Use glob to find .txt files
    current_files = {f.name for f in watch_dir.glob('*.txt')}
    new_files = current_files - processed
    
    if new_files:
        new_files_sorted = sorted(
            new_files,
            key=lambda f: (watch_dir / f).stat().st_mtime
        )
    
        for filename in new_files_sorted:
            filepath = watch_dir / filename
            try:
                print(f"analyzing {filename}")
                result = analyze_file(filepath)
                send_tcp(result,tcp_socket)
                processed.add(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    time.sleep(1)  # Check every second