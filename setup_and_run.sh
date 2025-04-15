#!/bin/bash

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
  echo "Error: The OPENAI_API_KEY environment variable is not set."
  echo "Please set it using 'export OPENAI_API_KEY=your_api_key_here' and try again."
  exit 1
fi

# Check if the "prompt-project" virtual environment already exists.
if [ ! -d "prompt-project" ]; then
  echo "1. Creating virtual environment 'prompt-project'..."
  python3 -m venv prompt-project
else
  echo "1. Virtual environment 'prompt-project' already exists."
fi

# Activate the virtual environment
echo "2. Activating virtual environment..."
source prompt-project/bin/activate

# Upgrade pip (optional but recommended)
pip install --upgrade pip

# Install required packages from requirements.txt
echo "3. Installing dependencies from requirements.txt..."
pip install -q -r requirements.txt

# Build the vector database using the RAG system.
# This script runs rag.py which should have its __main__ block set to rebuild the database (rebuild=True)
echo "4. Building the vector database (RAG)..."
python rag.py

# Start the Flask application by running main.py
echo "Starting Flask app..."
python main.py
