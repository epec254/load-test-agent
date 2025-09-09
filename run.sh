#!/bin/bash
kill $(lsof -t -i:8010) || true
uvicorn load_test_agent.main:app --host 0.0.0.0 --port 8010
