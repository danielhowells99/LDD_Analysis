import os
import time
import socket
from analysis import azar_surface_analysis

RESOLUTION = 800#2000
TCP_PORT = 4192
TCP_HOST = "192.168.3.232"
WATCH_PATH = "E:/IPG OmniWELD Data/3D Viewer/AutomationTests_Feb2026/20260219"

def analyze_file(filepath):
    with open(filepath, 'r') as f:
        data = nd.loadtxt(f)
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

processed = set()
watch_dir = WATCH_PATH
i = 0
while True:
    print(f"watch loop itr: {i}")
    i = i + 1;
    current_files = {f for f in os.listdir(watch_dir) if f.endswith('.txt')}
    new_files = current_files - processed
    
    if (len(new_files) == 0):
           print("waiting for new file...")
    
    for filename in new_files:
        filepath = os.path.join(watch_dir, filename)
        try:
            print(f"analyzing {filename}")
            result = analyze_file(filepath)
            send_tcp(result,tcp_socket)
            processed.add(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    time.sleep(1)  # Check every second