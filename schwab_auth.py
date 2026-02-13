"""
Schwab OAuth2 Authentication Handler
Handles authorization code flow and token refresh
"""

import json
import base64
import requests
import webbrowser
from datetime import datetime, timedelta
from urllib.parse import urlencode, urlparse, parse_qs
import os

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'schwab_config.json')
REQUEST_TIMEOUT_SECONDS = 20


def load_config():
    """Load configuration from file, returns empty dict if file is missing."""
    if not os.path.exists(CONFIG_FILE):
        return {}
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)


def save_config(config):
    """Save configuration to file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)


def get_authorization_url():
    """Generate the authorization URL for user to visit"""
    config = load_config()
    params = {
        'response_type': 'code',
        'client_id': config['client_id'],
        'scope': 'readonly',
        'redirect_uri': config['redirect_uri']
    }
    url = f"{config['auth_url']}?{urlencode(params)}"
    return url


def extract_code_from_url(redirect_url):
    """Extract authorization code from the redirect URL"""
    parsed = urlparse(redirect_url)
    params = parse_qs(parsed.query)
    if 'code' in params:
        return params['code'][0]
    raise ValueError("No authorization code found in URL")


def exchange_code_for_tokens(auth_code):
    """Exchange authorization code for access and refresh tokens"""
    config = load_config()

    # Create Basic auth header
    credentials = f"{config['client_id']}:{config['client_secret']}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()

    headers = {
        'Authorization': f'Basic {encoded_credentials}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    data = {
        'grant_type': 'authorization_code',
        'code': auth_code,
        'redirect_uri': config['redirect_uri']
    }

    response = requests.post(
        config['token_url'],
        headers=headers,
        data=data,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if response.status_code != 200:
        raise Exception(f"Token exchange failed: {response.status_code} - {response.text}")

    tokens = response.json()

    # Save tokens to config
    config['access_token'] = tokens['access_token']
    config['refresh_token'] = tokens['refresh_token']
    # Token typically expires in 30 minutes
    expires_in = tokens.get('expires_in', 1800)
    config['token_expires_at'] = (datetime.now() + timedelta(seconds=expires_in)).isoformat()
    save_config(config)

    print(f"✓ Tokens saved successfully!")
    print(f"  Access token expires: {config['token_expires_at']}")

    return tokens


def refresh_access_token():
    """Refresh the access token using the refresh token"""
    config = load_config()

    if not config.get('refresh_token'):
        raise Exception("No refresh token available. Please re-authorize.")

    # Create Basic auth header
    credentials = f"{config['client_id']}:{config['client_secret']}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()

    headers = {
        'Authorization': f'Basic {encoded_credentials}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    data = {
        'grant_type': 'refresh_token',
        'refresh_token': config['refresh_token']
    }

    response = requests.post(
        config['token_url'],
        headers=headers,
        data=data,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if response.status_code != 200:
        raise Exception(f"Token refresh failed: {response.status_code} - {response.text}")

    tokens = response.json()

    # Save new tokens
    config['access_token'] = tokens['access_token']
    if 'refresh_token' in tokens:
        config['refresh_token'] = tokens['refresh_token']
    expires_in = tokens.get('expires_in', 1800)
    config['token_expires_at'] = (datetime.now() + timedelta(seconds=expires_in)).isoformat()
    save_config(config)

    print(f"✓ Access token refreshed! Expires: {config['token_expires_at']}")

    return tokens['access_token']


def get_valid_access_token():
    """Get a valid access token, refreshing if necessary"""
    config = load_config()

    if not config.get('access_token'):
        raise Exception("No access token. Please authorize first using: python schwab_auth.py")

    # Check if token is expired or about to expire (within 5 minutes)
    if config.get('token_expires_at'):
        expires_at = datetime.fromisoformat(config['token_expires_at'])
        if datetime.now() > expires_at - timedelta(minutes=5):
            print("Access token expired or expiring soon, refreshing...")
            return refresh_access_token()

    return config['access_token']


def authorize():
    """Interactive authorization flow"""
    print("\n" + "="*70)
    print("SCHWAB API AUTHORIZATION")
    print("="*70)

    config = load_config()

    if config.get('client_secret') == 'YOUR_CLIENT_SECRET_HERE':
        print("\n⚠️  First, edit schwab_config.json and add your client_secret!")
        print("   Then run this script again.\n")
        return False

    auth_url = get_authorization_url()

    print("\n1. Opening browser for Schwab login...")
    print(f"\n   If browser doesn't open, visit this URL:\n   {auth_url}\n")

    try:
        webbrowser.open(auth_url)
    except:
        pass

    print("2. After logging in and authorizing, you'll be redirected.")
    print("   Copy the ENTIRE URL from your browser's address bar.\n")

    redirect_url = input("3. Paste the redirect URL here: ").strip()

    try:
        auth_code = extract_code_from_url(redirect_url)
        print(f"\n✓ Authorization code extracted!")

        print("\n4. Exchanging code for tokens...")
        exchange_code_for_tokens(auth_code)

        print("\n" + "="*70)
        print("✅ AUTHORIZATION COMPLETE!")
        print("="*70)
        print("\nYou can now use the Schwab API. Tokens are saved in schwab_config.json")
        print("The access token will auto-refresh when needed.\n")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == "__main__":
    authorize()
