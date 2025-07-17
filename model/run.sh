#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Change directory to the script's location to ensure paths are correct.
cd "$(dirname "$0")"

# Create the models directory if it doesn't exist.
mkdir -p models

# Install dependencies and run the training script.
pip install -r ../requirements.txt
python lstm_models.py 