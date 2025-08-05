#!/bin/bash

# Script to generate predictions for skin cancer test set
# For Phase II submission

echo "============================================================"
echo "SKIN CANCER TEST PREDICTION GENERATOR"
echo "============================================================"

# Check if checkpoints directory exists
if [ ! -d "automl-exam-ss25-vision-freiburg-template/checkpoints" ]; then
    echo "❌ ERROR: Checkpoints directory not found!"
    echo "Please run your AutoML pipeline first to train models."
    exit 1
fi

# List available checkpoints
echo "Available trained models:"
ls -la automl-exam-ss25-vision-freiburg-template/checkpoints/*.pt 2>/dev/null || {
    echo "❌ ERROR: No trained models found in checkpoints directory!"
    echo "Please run your AutoML pipeline first to train models."
    exit 1
}

echo ""
echo "============================================================"

# Check which model to use (prefer the final models)
if [ -f "automl-exam-ss25-vision-freiburg-template/checkpoints/final_resnet18.pt" ]; then
    CHECKPOINT="automl-exam-ss25-vision-freiburg-template/checkpoints/final_resnet18.pt"
    echo "Using ResNet18 model: $CHECKPOINT"
elif [ -f "automl-exam-ss25-vision-freiburg-template/checkpoints/final_efficientnet_b0.pt" ]; then
    CHECKPOINT="automl-exam-ss25-vision-freiburg-template/checkpoints/final_efficientnet_b0.pt"
    echo "Using EfficientNet-B0 model: $CHECKPOINT"
else
    # Use the first available checkpoint
    CHECKPOINT=$(ls automl-exam-ss25-vision-freiburg-template/checkpoints/*.pt | head -n 1)
    echo "Using first available model: $CHECKPOINT"
fi

# Generate predictions
echo "Generating predictions..."
python generate_predictions.py \
    --checkpoint "$CHECKPOINT" \
    --data_root "automl-exam-ss25-vision-freiburg-template/data" \
    --output "skin_cancer_predictions.csv" \
    --batch_size 32 \
    --image_size 450

# Check if prediction was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ PREDICTION GENERATION COMPLETED SUCCESSFULLY!"
    echo "============================================================"
    echo "📁 Submission file: skin_cancer_predictions.csv"
    echo "📊 Probabilities file: skin_cancer_predictions_probabilities.csv"
    echo ""
    echo "Next steps for Phase II submission:"
    echo "1. Review the prediction distribution in the output above"
    echo "2. Submit 'skin_cancer_predictions.csv' via GitHub"
    echo "3. Wait for test score from grading system"
    echo "============================================================"
else
    echo ""
    echo "❌ ERROR: Prediction generation failed!"
    echo "Please check the error messages above and try again."
    exit 1
fi
