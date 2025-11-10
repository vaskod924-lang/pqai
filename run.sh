#!/bin/bash

# Exit on error
set -e

# Step 1: Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv venv

# Step 2: Activate virtual environment
echo "🚀 Activating virtual environment..."
source venv/bin/activate

# Step 3: Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Step 4: Install dependencies
echo "📦 Installing required packages..."
pip install fastapi uvicorn requests sqlalchemy python-dotenv

# Step 5: Run the app
echo "🧠 Starting chatbot app..."
uvicorn chatbot:app --host 0.0.0.0 --port 5000 --reload