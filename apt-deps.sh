#!/bin/bash
# Install dependencies
apt update && apt upgrade -y
apt install -y libcamera libcamera-apps python3-picamera2 libcap-dev
# Create repository
python -m venv repo --system-site-packages
source repo/bin/activate
pip install -r requirements.txt