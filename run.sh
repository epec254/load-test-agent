#!/bin/bash

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

kill $(lsof -t -i:8010) || true
uvicorn load_test_agent.main:app --host 0.0.0.0 --port 8010 --reload
