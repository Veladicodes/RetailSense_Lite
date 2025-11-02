# launch_dashboard.py - RetailSense Dashboard Launcher
"""
RetailSense Phase 4: Dashboard Launcher
Automated setup and launch script for the Streamlit dashboard
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

# Ensure UTF-8 capable output (prevents UnicodeEncodeError on Windows terminals)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Ensure required directories exist
required_dirs = [
    os.path.join("data", "processed"),
    os.path.join("data", "uploaded"),
    os.path.join("data", "predictions"),
    "outputs",
    "notebooks"
]

for directory in required_dirs:
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), directory), exist_ok=True)

# -------------------------------
# Step 1: Check requirements
# -------------------------------
def check_requirements(auto_install=True):
    """Check and install required packages if missing"""
    print("🔍 Checking Python package requirements...\n")

    # Map pip package name -> import name
    required_packages = {
        "streamlit": "streamlit",
        "pandas": "pandas",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "plotly": "plotly",
        "scikit-learn": "sklearn",   # ✅ Correct mapping
        "joblib": "joblib",
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "optuna": "optuna",
        "prophet": "prophet",
        "statsmodels": "statsmodels",
        "papermill": "papermill",    # For notebook execution
    }

    missing = []
    for pip_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {pip_name} installed")
        except ImportError:
            print(f"❌ {pip_name} missing")
            missing.append(pip_name)

    if missing and auto_install:
        print("\n📦 Installing missing packages...")
        for pkg in missing:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                print(f"✅ Installed {pkg}")
            except Exception as e:
                print(f"❌ Failed to install {pkg}: {e}")

        # 🔁 Re-check after installation
        missing_after = []
        for pip_name, import_name in required_packages.items():
            try:
                __import__(import_name)
            except ImportError:
                missing_after.append(pip_name)

        if missing_after:
            print(f"\n❌ Still missing: {missing_after}")
            return False
        else:
            print("\n✅ All dependencies installed successfully!")
            return True

    return not missing

# -------------------------------
# Step 2: Check data files
# -------------------------------
def check_data_files():
    """Check Phase 2 & 3 output files"""
    print("\n📊 Checking data files...\n")

    required_files = [
        "data/processed/data_with_all_features.csv",
        "business_inventory_alerts.csv",
        "business_sales_anomalies.csv",
        "business_seasonal_insights.csv",
        "business_pricing_opportunities.csv",
        "phase3_completion_report.csv"
    ]

    all_found = True
    for f in required_files:
        if os.path.exists(f):
            print(f"✅ Found: {f}")
        else:
            print(f"⚠️ Missing: {f}")
            all_found = False

    return all_found

# -------------------------------
# Step 3: Check Streamlit App
# -------------------------------
def check_streamlit_app():
    """Ensure app.py exists"""
    print("\n📱 Checking Streamlit dashboard app...\n")

    if os.path.exists("app.py"):
        print("✅ app.py found")
        return True
    else:
        print("❌ app.py not found")
        print("💡 Please create Phase 4 dashboard in app.py")
        return False

# -------------------------------
# Step 4: Launch dashboard
# -------------------------------
def launch_dashboard():
    """Launch Streamlit dashboard"""
    print("\n🚀 Launching RetailSense Dashboard...")
    print("📅", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("📱 Dashboard will open at: http://localhost:8501\n")

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless=false"])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")


# -------------------------------
# Main
# -------------------------------
def main(auto=True):
    print("\n🛍️ RETAILSENSE DASHBOARD LAUNCHER")
    print("="*60)
    
    if auto:
        print("✅ Auto-install enabled: Missing packages will be installed automatically")
    else:
        print("⚠️ Auto-install disabled: You must install missing packages manually")
    print()

    # Requirements
    if not check_requirements(auto_install=auto):
        print("❌ Missing dependencies. Please install manually.")
        return

    # Data
    if not check_data_files():
        print("\n⚠️ Some data files missing. The dashboard may still run but with limited features.")

    # App
    if not check_streamlit_app():
        return

    # Launch
    launch_dashboard()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RetailSense Dashboard Launcher")
    parser.add_argument("--no-auto-install", action="store_true", help="Disable automatic package installation")
    args = parser.parse_args()

    # Auto-install is ON by default, disable only if --no-auto-install flag is used
    main(auto=not args.no_auto_install)
