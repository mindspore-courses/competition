#!/bin/bash
set -e
echo "--- Initializing Conda environment ---"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms

echo "--- Conda environment 'ms' activated. Executing command: $@ ---"

exec "$@"
