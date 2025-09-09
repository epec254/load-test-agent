#!/bin/bash

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Kill any existing processes on port 8010
kill $(lsof -t -i:8010) || true

# Start uvicorn with 8 workers for better concurrency
# Note: --reload doesn't work with multiple workers, so we remove it for production
uvicorn load_test_agent.main:app --host 0.0.0.0 --port 8010 --workers 4
