# Create venv and install dependencies
install: 
    python3 -m venv .venv
    source .venv/bin/activate && pip install uv && uv sync
