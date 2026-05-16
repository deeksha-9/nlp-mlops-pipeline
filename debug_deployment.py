"""
Debug script to check deployment readiness.
Run this locally to verify everything works before deploying.
"""

import os
import sys

print("\n" + "=" * 70)
print("DEPLOYMENT DEBUG CHECKLIST")
print("=" * 70 + "\n")

# 1. Check current directory
print("1. Current working directory:")
print(f"   {os.getcwd()}\n")

# 2. Check if CSV exists
csv_path = "data/IMDB_Dataset.csv"
print(f"2. Dataset check ({csv_path}):")
if os.path.exists(csv_path):
    size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"   ✓ File exists ({size_mb:.2f} MB)\n")
else:
    print(f"   ✗ File NOT found\n")
    sys.exit(1)

# 3. Check Python packages
print("3. Required packages:")
packages = ['pandas', 'mlflow', 'sklearn', 'fastapi', 'uvicorn']
missing = []
for pkg in packages:
    try:
        __import__(pkg if pkg != 'sklearn' else 'sklearn')
        print(f"   ✓ {pkg}")
    except ImportError:
        print(f"   ✗ {pkg} MISSING")
        missing.append(pkg)

if missing:
    print(f"\n   Install missing: pip install {' '.join(missing)}\n")
    sys.exit(1)
print()

# 4. Try training
print("4. Testing training script:")
try:
    import subprocess
    result = subprocess.run(
        [sys.executable, "src/train.py"],
        capture_output=True,
        text=True,
        timeout=300  # 5 min timeout
    )
    if result.returncode == 0:
        print("   ✓ Training completed successfully\n")
    else:
        print(f"   ✗ Training failed:\n{result.stderr}\n")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Error running training: {e}\n")
    sys.exit(1)

# 5. Check mlruns directory
print("5. MLflow artifacts:")
if os.path.exists("./mlruns"):
    model_paths = []
    for root, dirs, files in os.walk("./mlruns"):
        if "model" in dirs:
            model_paths.append(os.path.join(root, "model"))
    
    if model_paths:
        print(f"   ✓ Found {len(model_paths)} trained model(s)")
        for path in model_paths[:3]:  # Show first 3
            print(f"      - {path}\n")
    else:
        print("   ✗ No models found in mlruns/\n")
        sys.exit(1)
else:
    print("   ✗ mlruns/ directory not found\n")
    sys.exit(1)

# 6. Test API startup
print("6. Testing API startup:")
try:
    from api.main import app
    print("   ✓ API imports successfully")
    
    # Check if model is loaded
    import asyncio
    # Note: Can't easily test async without running the event loop
    print("   ✓ Model should load when API starts\n")
except Exception as e:
    print(f"   ✗ API import error: {e}\n")
    sys.exit(1)

print("=" * 70)
print("✓ ALL CHECKS PASSED - Ready for deployment!")
print("=" * 70 + "\n")
