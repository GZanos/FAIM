#!/usr/bin/env python3
"""
Setup script for NASA POWER Wildfire Risk Assessment Web App
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running: {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Setting up NASA POWER Wildfire Risk Assessment Web App")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✅ Python version: {sys.version}")
    
    # Install requirements
    print("\n📦 Installing required packages...")
    if not run_command("pip install -r requirements.txt"):
        print("❌ Failed to install requirements")
        return False
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    try:
        import streamlit
        import folium
        import pandas
        import geopandas
        print("✅ All packages installed successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    print("\n🎉 Setup complete!")
    print("\n" + "=" * 60)
    print("To run the application:")
    print("  streamlit run wildfire_risk_app.py")
    print("\nThe app will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    
    return True

if __name__ == "__main__":
    main()