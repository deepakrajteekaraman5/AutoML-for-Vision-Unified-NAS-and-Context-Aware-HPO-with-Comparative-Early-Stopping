#!/bin/bash

# AutoML Pipeline Runner Script
# Compatible with Git Bash on Windows

echo "==================================="
echo "AutoML Pipeline Runner"
echo "==================================="
echo ""

# Ask user for dataset name
echo "Available datasets:"
echo "  - emotions"
echo "  - flowers" 
echo "  - skin_cancer"
echo "  - fashion"
echo ""
read -p "Enter dataset name (x): " dataset_name

# Validate input
if [ -z "$dataset_name" ]; then
    echo "Error: Dataset name cannot be empty!"
    exit 1
fi

echo ""
echo "Starting AutoML pipeline for dataset: $dataset_name"
echo "==================================="

# Validate input
if [ -z "$dataset_name" ]; then
    echo "Error: Dataset name cannot be empty!"
    exit 1
fi

# Ask user for dataset name
echo "Time Budget"
echo ""
read -p "Enter time buget in the followign format (y) (eg: 2 Hours = 2.0): " time_budget

# Validate input
if [ -z "$time_budget" ]; then
    echo "Error: Time Budget cannot be empty!"
    exit 1
fi



echo ""
echo "Starting AutoML pipeline for dataset: $dataset_name with timebudget: $time_budget"
echo "==================================="

# Change to the AutoML directory
cd automl-exam-ss25-vision-freiburg-template/

# Check if directory exists
if [ ! -d "src/automl" ]; then
    echo "Error: AutoML source directory not found!"
    echo "Make sure you're running this script from the correct location."
    exit 1
fi

# Run the AutoML pipeline
echo "Running: python -m src.automl.run_automl --dataset $dataset_name --time_budget $time_budget"
echo ""

python -m src.automl.run_automl --dataset "$dataset_name" --time_budget "$time_budget"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "==================================="
    echo "AutoML pipeline completed successfully!"
    echo "==================================="
else
    echo ""
    echo "==================================="
    echo "AutoML pipeline failed with errors."
    echo "Check the logs above for details."
    echo "==================================="
    exit 1
fi
