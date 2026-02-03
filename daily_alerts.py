"""
Automated daily CSP timing alerts
Run this via cron job or task scheduler
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictor import CSPPredictor


# Configuration
WATCHLIST = ['NVDA', 'AMD', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN']
CONFIDENCE_THRESHOLD = 0.55  # Only alert on 55%+ confidence
ALERT_METHOD = 'print'  # Options: 'print', 'email', 'file'

# Email configuration (if using email alerts)
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'your-email@gmail.com',
    'sender_password': 'your-app-password',  # Use app-specific password
    'recipient_email': 'your-email@gmail.com'
}


def format_alert_message(opportunities):
    """Format opportunities into readable message"""
    if not opportunities:
        return "No high-confidence CSP opportunities found today."

    message = f"CSP TIMING ALERT - {datetime.now().strftime('%Y-%m-%d')}\n"
    message += "="*60 + "\n\n"
    message += f"Found {len(opportunities)} high-confidence opportunities:\n\n"

    for i, opp in enumerate(opportunities, 1):
        message += f"{i}. {opp['ticker']} - ${opp['current_price']:.2f}\n"
        message += f"   Recommendation: {opp['prediction']}\n"
        message += f"   Confidence: {opp['confidence']:.1%}\n"
        message += f"   Probability Good: {opp['prob_good']:.1%}\n"
        message += "\n"

    message += "="*60 + "\n"
    message += "Check full details at your web interface\n"

    return message


def send_email_alert(message):
    """Send alert via email"""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['sender_email']
        msg['To'] = EMAIL_CONFIG['recipient_email']
        msg['Subject'] = f"CSP Timing Alert - {len(opportunities)} Opportunities"

        msg.attach(MIMEText(message, 'plain'))

        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
        server.send_message(msg)
        server.quit()

        print("✓ Email alert sent successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to send email: {e}")
        return False


def save_to_file(message):
    """Save alert to file"""
    filename = f"csp_alerts_{datetime.now().strftime('%Y%m%d')}.txt"
    try:
        with open(filename, 'w') as f:
            f.write(message)
        print(f"✓ Alert saved to {filename}")
        return True
    except Exception as e:
        print(f"✗ Failed to save to file: {e}")
        return False


def scan_opportunities():
    """Scan watchlist for CSP opportunities"""
    print(f"\nScanning {len(WATCHLIST)} tickers for CSP opportunities...")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    try:
        # Load predictor
        predictor = CSPPredictor('csp_model.pkl')

        opportunities = []

        for ticker in WATCHLIST:
            try:
                print(f"\nChecking {ticker}...", end=' ')
                result = predictor.predict(ticker, show_details=False)

                # Check if it's a good opportunity
                if result['prediction'] == 'GOOD' and result['confidence'] >= CONFIDENCE_THRESHOLD:
                    opportunities.append(result)
                    print(f"✓ OPPORTUNITY ({result['confidence']:.1%})")
                else:
                    print(f"  ({result['prediction']}, {result['confidence']:.1%})")

            except Exception as e:
                print(f"✗ Error: {str(e)}")
                continue

        return opportunities

    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        return []


def main():
    """Main alert function"""
    print("\n" + "="*60)
    print("CSP TIMING DAILY ALERT SYSTEM")
    print("="*60)

    # Scan for opportunities
    opportunities = scan_opportunities()

    # Format message
    message = format_alert_message(opportunities)

    print("\n" + "="*60)
    print("ALERT MESSAGE")
    print("="*60)
    print(message)

    # Send alerts based on configuration
    if ALERT_METHOD == 'email':
        send_email_alert(message)
    elif ALERT_METHOD == 'file':
        save_to_file(message)
    elif ALERT_METHOD == 'print':
        print("\n✓ Alert displayed (configured for print mode)")

    # Always save to log
    log_file = "csp_alerts.log"
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(message)

    print(f"\n✓ Alert logged to {log_file}")

    return len(opportunities)


if __name__ == "__main__":
    opportunities_found = main()
    sys.exit(0 if opportunities_found >= 0 else 1)
