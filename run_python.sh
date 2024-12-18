#!/bin/bash

# Define the name of the virtual environment
ENV_NAME="my_new_env"

# Define the path to the project root directory where 'pikan' is located
PROJECT_DIR="."  # Change this to the correct path

# Create a new virtual environment
python -m venv $ENV_NAME

# Activate the virtual environment
source $ENV_NAME/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Add the project directory to PYTHONPATH so the 'pikan' package can be found
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# Check if the Python file is provided as an argument
if [ "$#" -eq 1 ]; then
    # Execute the Python file
    python $1
else
    echo "Usage: $0 <python_file>"
fi

# Deactivate the virtual environment
deactivate
