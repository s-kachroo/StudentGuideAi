#!/usr/bin/env bash
set -euo pipefail

# 0. make sure we run from the directory containing this script
cd "$(dirname "$0")"

# 1. check for OpenAI key
: "${OPENAI_API_KEY?Error: OPENAI_API_KEY is not set. Please 'export OPENAI_API_KEY=…' and retry.}"

# 2. check for python3
if ! command -v python3 &>/dev/null; then
  echo "Error: python3 not found in PATH." >&2
  exit 1
fi

# 3. create (or reuse) a virtual env in .venv/
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
else
  echo "Using existing virtual environment in $VENV_DIR."
fi

# 4. activate and install dependencies
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
python3 -m pip install --upgrade pip setuptools wheel
echo "Installing requirements…"
python3 -m pip install -r requirements.txt

# 5. build/update your vector database
echo "Building/updating RAG vector database…"
python3 ui/rag.py  # make sure rag.py’s __main__ rebuilds when run directly

# 6. launch your Flask app
echo "Starting Flask development server…"
export FLASK_APP=ui/main.py
export FLASK_ENV=development
flask run
