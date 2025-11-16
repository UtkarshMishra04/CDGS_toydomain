#!/bin/bash
# Script to identify and run missing horizon-resampling matrix experiments

# Exit on error
set -e

OUTPUT_DIR="profile/sweep_horizon_resampling_matrix"
mkdir -p "$OUTPUT_DIR"

# Generate all expected combinations
echo "Generating all expected horizon-resampling combinations..."
HORIZON_LENGTHS=($(seq 2 100))  # 99 values
RESAMPLING_STEPS=(1 5 10 15 20)  # 5 values

EXPECTED_COMBINATIONS=()
for horizon in "${HORIZON_LENGTHS[@]}"; do
    for resampling in "${RESAMPLING_STEPS[@]}"; do
        EXPECTED_COMBINATIONS+=("$horizon,$resampling")
    done
done

echo "Total expected combinations: ${#EXPECTED_COMBINATIONS[@]}"

# Check which combinations already have results
echo "Checking existing results..."

# First, quickly build a list of existing combinations
EXISTING_COMBOS=()
if [ -d "$OUTPUT_DIR" ]; then
    echo "Reading existing JSON files..."
    for json_file in "$OUTPUT_DIR"/*.json; do
        if [ -f "$json_file" ]; then
            # Extract both values in one pass
            HORIZON_VAL=$(grep -o '"horizon_length": [0-9]*' "$json_file" | head -1 | awk '{print $2}')
            RESAMPLING_VAL=$(grep -o '"num_resampling_steps": [0-9]*' "$json_file" | head -1 | awk '{print $2}')
            if [ -n "$HORIZON_VAL" ] && [ -n "$RESAMPLING_VAL" ]; then
                EXISTING_COMBOS+=("$HORIZON_VAL,$RESAMPLING_VAL")
            fi
        fi
    done
fi

echo "Found ${#EXISTING_COMBOS[@]} existing results"
echo "Comparing against ${#EXPECTED_COMBINATIONS[@]} expected combinations..."

# Now find missing combinations
MISSING_COMBINATIONS=()
for combo in "${EXPECTED_COMBINATIONS[@]}"; do
    FOUND=0
    for existing in "${EXISTING_COMBOS[@]}"; do
        if [ "$combo" = "$existing" ]; then
            FOUND=1
            break
        fi
    done
    if [ $FOUND -eq 0 ]; then
        MISSING_COMBINATIONS+=("$combo")
    fi
done

echo "Found ${#MISSING_COMBINATIONS[@]} missing combinations"

if [ ${#MISSING_COMBINATIONS[@]} -eq 0 ]; then
    echo "All experiments complete! No missing combinations."
    exit 0
fi

echo "Missing combinations (showing first 20):"
for i in $(seq 0 $((${#MISSING_COMBINATIONS[@]} < 20 ? ${#MISSING_COMBINATIONS[@]} - 1 : 19))); do
    combo="${MISSING_COMBINATIONS[$i]}"
    HORIZON=$(echo $combo | cut -d',' -f1)
    RESAMPLING=$(echo $combo | cut -d',' -f2)
    echo "  horizon=$HORIZON, resampling=$RESAMPLING"
done
if [ ${#MISSING_COMBINATIONS[@]} -gt 20 ]; then
    echo "  ... and $((${#MISSING_COMBINATIONS[@]} - 20)) more"
fi

# Ask for confirmation
read -p "Run ${#MISSING_COMBINATIONS[@]} missing experiments? (y/n) " -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Run missing experiments
echo "Starting to run ${#MISSING_COMBINATIONS[@]} missing experiments..."
echo ""
COMPLETED=0
FAILED=0

for combo in "${MISSING_COMBINATIONS[@]}"; do
    HORIZON=$(echo $combo | cut -d',' -f1)
    RESAMPLING=$(echo $combo | cut -d',' -f2)

    echo "[$((COMPLETED + FAILED + 1))/${#MISSING_COMBINATIONS[@]}] Running: horizon=$HORIZON, resampling=$RESAMPLING"

    set +e  # Temporarily disable exit on error
    uv run sampling_time_single.py \
        --horizon-length $HORIZON \
        --enable-pruning False \
        --num-resampling-steps $RESAMPLING \
        --output-directory "$OUTPUT_DIR" 0<&-
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        COMPLETED=$((COMPLETED + 1))
        echo "  ✓ Success (total completed: $COMPLETED)"
    else
        FAILED=$((FAILED + 1))
        echo "  ✗ Failed with exit code $EXIT_CODE (total failed: $FAILED)"
    fi

    set -e  # Re-enable exit on error
    echo ""
done

echo ""
echo "===================="
echo "Summary:"
echo "  Completed: $COMPLETED"
echo "  Failed: $FAILED"
echo "  Total: ${#MISSING_COMBINATIONS[@]}"
echo "===================="
