#!/bin/bash
# Setup DVC to use Hugging Face as remote storage (free option)
# Run this once locally, then push changes to Git

# Option A: Use a public S3 bucket (if you have one)
# dvc remote add -d myremote s3://your-bucket-name/dvc-storage

# Option B: Use local .git directory as remote (works for small projects)
dvc remote add -d myremote .dvc/remote_storage

# Then run: dvc push
# And commit the changes
