#!/bin/bash
set -e

echo "=== Current directory ==="
pwd

echo "=== Checking dataset ==="
ls -la data/IMDB_Dataset.csv || { echo "ERROR: Dataset not found!"; exit 1; }

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Training model ==="
python src/train.py

echo "=== Training complete ==="
ls -la mlruns/
