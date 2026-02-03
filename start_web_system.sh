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
python api_server.py &
API_PID=$!
echo "✓ API server started (PID: $API_PID)"
sleep 2

# Test API
echo ""
echo "Testing API server..."
curl -s http://localhost:8000/ > /dev/null
if [ $? -eq 0 ]; then
    echo "✓ API server is responding"
else
    echo "ERROR: API server not responding"
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
echo "--------------------------------------------------------------------"
echo "NEXT STEPS TO ACCESS FROM WORK:"
echo "--------------------------------------------------------------------"
echo ""
echo "1. In a NEW terminal, run:"
echo "   ngrok http 8000"
echo ""
echo "2. Copy the ngrok URL (https://xxxxx.ngrok.io)"
echo ""
echo "3. Deploy frontend to GitHub Pages:"
echo "   - Create repo: csp-timing-predictor"
echo "   - Push web/index.html"
echo "   - Enable GitHub Pages in Settings"
echo ""
echo "4. Open your GitHub Pages URL at work"
echo "   Enter your ngrok URL in the settings"
echo "   Get predictions!"
echo ""
echo "--------------------------------------------------------------------"
echo "To stop the server: kill $API_PID"
echo "======================================================================"
echo ""

# Keep script running
wait $API_PID
