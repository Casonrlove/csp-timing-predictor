#!/bin/bash

# Quick start script for CSP Timing API server

echo "======================================================================"
echo "CSP TIMING PREDICTION SERVER - STARTUP"
echo "======================================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found. Please install Python 3.10+."
    exit 1
fi

# Check if models exist
if [ ! -f "csp_hybrid_model.pkl" ] && [ ! -f "csp_tabnet_nvda.pkl" ]; then
    echo "WARNING: No trained models found!"
    echo "Please train models first:"
    echo "  Option 1: python deep_learning_model.py"
    echo "  Option 2: python tabnet_trainer.py"
    echo ""
    read -p "Do you want to train the Hybrid model now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 deep_learning_model.py
    else
        echo "Exiting. Please train models first."
        exit 1
    fi
fi

echo ""
echo "Starting API server..."
echo ""
echo "Server will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "--------------------------------------------------------------------"
echo "NEXT STEPS:"
echo "1. Open a new terminal"
echo "2. Run: ngrok http 8000"
echo "3. Copy the ngrok URL (https://xxxxx.ngrok.io)"
echo "4. Use that URL in your web interface"
echo "--------------------------------------------------------------------"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 api_server.py
