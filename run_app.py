import os
import sys
import subprocess
import webbrowser
import time

# Get current folder path
base_dir = os.path.dirname(os.path.abspath(__file__))

# Path to app.py
app_path = os.path.join(base_dir, "app", "app.py")

# Start Streamlit
process = subprocess.Popen(
    ["streamlit", "run", app_path, "--server.headless", "true"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait a few seconds for server to start
time.sleep(3)

# Open browser automatically
webbrowser.open("http://localhost:8501")

process.wait()
