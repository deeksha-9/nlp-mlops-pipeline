#!/usr/bin/env python3
"""Verify the entire setup before deployment."""

import os
import sys
import subprocess

print("\n" + "="*70)
print("SETUP VERIFICATION")
print("="*70 + "\n")

# 1. Check working directory
print("1. Working directory:")
cwd = os.getcwd()
print(f"   {cwd}\n")

# 2. Check dataset
print("2. Dataset:")
if os.path.exists("data/IMDB_Dataset.csv"):
    size = os.path.getsize("data/IMDB_Dataset.csv") / (1024**2)
    print(f"   ✓ data/IMDB_Dataset.csv exists ({size:.2f} MB)\n")
else:
    print(f"   ✗ data/IMDB_Dataset.csv NOT FOUND\n")
    sys.exit(1)

# 3. Check if training runs
print("3. Running training...")
result = subprocess.run(
    [sys.executable, "src/train.py"],
    capture_output=True,
    text=True,
    cwd=cwd
)

if result.returncode != 0:
    print(f"   ✗ Training failed!")
    print(f"\nSTDOUT:\n{result.stdout}")
    print(f"\nSTDERR:\n{result.stderr}")
    sys.exit(1)
else:
    print(f"   ✓ Training completed successfully\n")
    print(f"Output:\n{result.stdout}\n")

# 4. Check mlruns structure
print("4. MLflow artifacts:")
if os.path.exists("mlruns"):
    print("   ✓ mlruns/ exists")
    model_paths = []
    for root, dirs, files in os.walk("mlruns"):
        for d in dirs:
            if d == "model":
                model_paths.append(os.path.join(root, d))

    if model_paths:
        print(f"   ✓ Found {len(model_paths)} model(s)")
        for p in model_paths[:3]:
            print(f"      - {p}")
    else:
        print("   ✗ No model artifacts found")
        print("\n   mlruns/ structure:")
        for root, dirs, files in os.walk("mlruns"):
            level = root.replace("mlruns", "").count(os.sep)
            indent = "   " + "  " * level
            print(f"{indent}{os.path.basename(root)}/")
            for f in files[:3]:
                print(f"{indent}  {f}")
    print()
else:
    print("   ✗ mlruns/ does not exist\n")
    sys.exit(1)

# 5. Test API imports
print("5. API startup test:")
try:
    from api.main import app, model, load_model
    print("   ✓ API imports successfully\n")
except Exception as e:
    print(f"   ✗ API import failed: {e}\n")
    sys.exit(1)

print("="*70)
print("✓ ALL CHECKS PASSED")
print("="*70 + "\n")
