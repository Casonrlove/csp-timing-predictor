#!/bin/bash

# Helper script to set up GitHub remote

echo "======================================================================"
echo "GITHUB SETUP - CSP Timing Predictor"
echo "======================================================================"
echo ""

# Check if git repo exists
if [ ! -d ".git" ]; then
    echo "ERROR: Not a git repository"
    echo "Run: git init"
    exit 1
fi

# Check if already has remote
if git remote | grep -q "origin"; then
    echo "Remote 'origin' already exists:"
    git remote -v
    echo ""
    read -p "Do you want to replace it? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git remote remove origin
    else
        echo "Keeping existing remote. Exiting."
        exit 0
    fi
fi

echo "First, create your GitHub repository:"
echo "  1. Go to: https://github.com/new"
echo "  2. Repository name: csp-timing-predictor"
echo "  3. Make it PUBLIC (for free GitHub Pages)"
echo "  4. Do NOT initialize with README"
echo "  5. Click 'Create repository'"
echo ""

read -p "Have you created the repository? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Create the repository first, then run this script again."
    exit 0
fi

# Get GitHub username
echo ""
read -p "Enter your GitHub username: " github_user

if [ -z "$github_user" ]; then
    echo "ERROR: Username required"
    exit 1
fi

# Choose authentication method
echo ""
echo "Choose authentication method:"
echo "  1. HTTPS (requires Personal Access Token)"
echo "  2. SSH (requires SSH key setup)"
echo ""
read -p "Enter choice (1 or 2): " -n 1 auth_choice
echo ""

if [ "$auth_choice" = "2" ]; then
    # SSH authentication
    remote_url="git@github.com:$github_user/csp-timing-predictor.git"

    echo ""
    echo "Testing SSH connection to GitHub..."
    if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
        echo "✓ SSH connection works!"
    else
        echo ""
        echo "⚠ WARNING: SSH connection test failed"
        echo ""
        echo "To set up SSH authentication:"
        echo "  1. Generate SSH key (if you don't have one):"
        echo "     ssh-keygen -t ed25519 -C \"your_email@example.com\""
        echo ""
        echo "  2. Copy your public key:"
        echo "     cat ~/.ssh/id_ed25519.pub"
        echo ""
        echo "  3. Add it to GitHub:"
        echo "     https://github.com/settings/keys"
        echo ""
        echo "  4. Test connection:"
        echo "     ssh -T git@github.com"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Exiting. Set up SSH and run this script again."
            exit 1
        fi
    fi
else
    # HTTPS authentication
    remote_url="https://github.com/$github_user/csp-timing-predictor.git"
fi

# Add remote
git remote add origin "$remote_url"

echo ""
echo "✓ Remote added: $remote_url"
echo ""

# Show current branch and commit status
echo "Current status:"
git log --oneline -1
echo ""

# Offer to push
read -p "Do you want to push to GitHub now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Pushing to GitHub..."

    if [ "$auth_choice" = "1" ]; then
        echo "You'll be prompted for credentials:"
        echo "  - Username: $github_user"
        echo "  - Password: Use a Personal Access Token (not your password)"
        echo ""
        echo "Get token at: https://github.com/settings/tokens"
        echo ""
    else
        echo "Using SSH authentication..."
        echo ""
    fi

    git push -u origin main

    if [ $? -eq 0 ]; then
        echo ""
        echo "======================================================================"
        echo "✓ SUCCESS! Code pushed to GitHub"
        echo "======================================================================"
        echo ""
        echo "Your repo: https://github.com/$github_user/csp-timing-predictor"
        echo ""
        echo "NEXT STEPS:"
        echo "  1. Enable GitHub Pages:"
        echo "     - Go to repo Settings → Pages"
        echo "     - Source: main branch, / (root)"
        echo "     - Save"
        echo ""
        echo "  2. Your site will be live at:"
        echo "     https://$github_user.github.io/csp-timing-predictor/"
        echo ""
        echo "  3. Start using:"
        echo "     - Start API: ./start_web_system.sh"
        echo "     - Start ngrok: ngrok http 8000"
        echo "     - Open your GitHub Pages URL"
        echo "     - Enter ngrok URL"
        echo ""
        echo "See GITHUB_DEPLOY.md for detailed instructions"
        echo "======================================================================"
    else
        echo ""
        echo "Push failed. Common issues:"
        echo "  - Wrong credentials (use Personal Access Token)"
        echo "  - Repository doesn't exist"
        echo "  - Network issues"
        echo ""
        echo "See GITHUB_DEPLOY.md for troubleshooting"
    fi
else
    echo ""
    echo "Skipped push. To push later, run:"
    echo "  git push -u origin main"
fi

echo ""
