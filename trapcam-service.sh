#!/bin/bash


# Create service pointing to this Script
#
# [Unit]
# Description=Efficient Tracker Camera Service
# After=network.target
#
# [Service]
# ExecStart=/path/to/efficient-tracker/trapcam-service.sh
# WorkingDirectory=/path/to/efficient-tracker
# Restart=always
# User=your_user
# Environment=PYTHONUNBUFFERED=1
#
# [Install]
# WantedBy=multi-user.target

source repo/bin/activate
python mp.py >> trapcam.log 2>&1