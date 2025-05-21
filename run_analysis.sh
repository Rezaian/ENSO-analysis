#!/bin/bash

# Run ENSO Analysis
# This script runs the ENSO analysis with default settings

# Default settings
SST_FILE="./data/era5_sst_1940_2024.nc"
OUTPUT_DIR="./output/$(date +%Y%m%d_%H%M%S)"
START_YEAR=1940
END_YEAR=2024

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print information
echo "===== ENSO Analysis ====="
echo "SST File: $SST_FILE"
echo "Output Directory: $OUTPUT_DIR"
echo "Analysis Period: $START_YEAR-$END_YEAR"
echo "========================="

# Run the analysis
python -m enso_analysis \
  --sst-file "$SST_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --start-year "$START_YEAR" \
  --end-year "$END_YEAR" \
  --log-level "INFO"

# Check if the analysis was successful
if [ $? -eq 0 ]; then
  echo "===== Analysis Complete ====="
  echo "Results saved to: $OUTPUT_DIR"
  echo "============================"
else
  echo "===== Analysis Failed ====="
  echo "Check the log file for details."
  echo "==========================="
  exit 1
fi
