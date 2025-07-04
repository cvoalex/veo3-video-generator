#!/bin/bash

# Create virtual environment
python3 -m venv veo3_env

# Activate virtual environment
source veo3_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Virtual environment setup complete!"
echo "To activate: source veo3_env/bin/activate"