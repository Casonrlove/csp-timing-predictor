#!/bin/bash

# Complete startup script for CSP timing web system

echo "======================================================================"
echo "CSP TIMING WEB SYSTEM - STARTUP"
echo "======================================================================"
echo ""

# Check if model exists
if [ ! -f "csp_model.pkl" ]; then
    echo "ERROR: No trained model found (csp_model.pkl)"
    echo "Please train the model first: python model_trainer.py"
    exit 1
fi

echo "✓ Model found: csp_model.pkl"
echo ""

# Start API server in background
echo "Starting API server on port 8000..."
python api_server.py > api_server.log 2>&1 &
API_PID=$!
echo "✓ API server started (PID: $API_PID)"
echo "  (Logs: api_server.log)"

# Wait for server to be ready
echo ""
echo "Waiting for API server to be ready..."
MAX_WAIT=10
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ API server is responding"
        break
    fi
    sleep 1
    WAITED=$((WAITED + 1))
    echo -n "."
done

if [ $WAITED -eq $MAX_WAIT ]; then
    echo ""
    echo "ERROR: API server not responding after ${MAX_WAIT}s"
    echo "Check api_server.log for errors"
    kill $API_PID 2>/dev/null
    exit 1
fi

echo ""
echo "======================================================================"
echo "API SERVER RUNNING"
echo "======================================================================"
echo ""
echo "Local access:       http://localhost:8000"
echo "API documentation:  http://localhost:8000/docs"
echo ""
echo "NOTE: GPU warning in logs is normal (using CPU for predictions)"
echo ""
echo "--------------------------------------------------------------------"
echo "NEXT STEPS TO ACCESS FROM WORK:"
echo "--------------------------------------------------------------------"
echo ""
echo "1. In a NEW terminal, run:"
echo "   ngrok http 8000"
echo ""
echo "2. Copy the ngrok URL (https://xxxxx.ngrok.io)"
echo ""
echo "3. Enable GitHub Pages (if not done yet):"
echo "   - Go to: https://github.com/Casonrlove/csp-timing-predictor"
echo "   - Settings → Pages → Source: main branch, / (root) → Save"
echo ""
echo "4. Open your GitHub Pages URL at work:"
echo "   https://casonrlove.github.io/csp-timing-predictor/"
echo ""
echo "5. Enter your ngrok URL in the web interface and get predictions!"
echo ""
echo "--------------------------------------------------------------------"
echo "To stop the server: kill $API_PID"
echo "======================================================================"
echo ""

# Keep script running
wait $API_PID
