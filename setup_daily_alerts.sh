#!/bin/bash

# Setup script for automated daily CSP alerts

echo "======================================================================"
echo "SETUP DAILY CSP ALERTS"
echo "======================================================================"
echo ""

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PATH=$(which python3)

echo "Script directory: $SCRIPT_DIR"
echo "Python path: $PYTHON_PATH"
echo ""

# Create cron job entry
CRON_TIME="30 16 * * 1-5"  # 4:30 PM EST, Monday-Friday (after market close)
CRON_JOB="$CRON_TIME cd $SCRIPT_DIR && $PYTHON_PATH daily_alerts.py >> csp_alerts.log 2>&1"

echo "This will add the following cron job:"
echo "$CRON_JOB"
echo ""
echo "Alerts will run: Monday-Friday at 4:30 PM (after market close)"
echo ""

read -p "Do you want to install this cron job? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check if cron job already exists
    if crontab -l 2>/dev/null | grep -q "daily_alerts.py"; then
        echo "Cron job already exists. Removing old entry..."
        crontab -l | grep -v "daily_alerts.py" | crontab -
    fi

    # Add new cron job
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

    echo "✓ Cron job installed successfully!"
    echo ""
    echo "To view your cron jobs: crontab -l"
    echo "To remove this job: crontab -e (then delete the line)"
    echo ""
else
    echo "Installation cancelled."
    echo ""
    echo "To run manually:"
    echo "  python3 daily_alerts.py"
    echo ""
fi

echo "======================================================================"
echo "CONFIGURATION OPTIONS"
echo "======================================================================"
echo ""
echo "Edit daily_alerts.py to customize:"
echo "  - WATCHLIST: Add/remove tickers"
echo "  - CONFIDENCE_THRESHOLD: Minimum confidence (default 55%)"
echo "  - ALERT_METHOD: 'print', 'email', or 'file'"
echo "  - Email settings (if using email alerts)"
echo ""
echo "For email alerts:"
echo "  1. Enable 2FA on your Gmail account"
echo "  2. Generate app-specific password"
echo "  3. Update EMAIL_CONFIG in daily_alerts.py"
echo ""
echo "======================================================================"
