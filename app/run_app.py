# Save this as run_app.py

import subprocess
import time
import streamlit as st

def run_streamlit():
    # Start Streamlit in the background
    streamlit_process = subprocess.Popen(["streamlit", "run", "app.py"])
    return streamlit_process

def run_localtunnel():
    # Start localtunnel
    localtunnel_process = subprocess.Popen(["npx", "localtunnel", "--port", "8501"], 
                                           stdout=subprocess.PIPE, 
                                           stderr=subprocess.PIPE, 
                                           text=True)
    
    # Wait for localtunnel to generate the URL
    for line in localtunnel_process.stdout:
        if "your url is:" in line.lower():
            url = line.split()[-1]
            print(f"Your app is available at: {url}")
            break
    
    return localtunnel_process, url

if __name__ == "__main__":
    # Ensure localtunnel is installed
    subprocess.run(["npm", "install", "-g", "localtunnel"])
    
    # Run Streamlit
    streamlit_process = run_streamlit()
    
    # Wait for Streamlit to start
    time.sleep(10)
    
    # Run localtunnel
    localtunnel_process, url = run_localtunnel()
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        streamlit_process.terminate()
        localtunnel_process.terminate()