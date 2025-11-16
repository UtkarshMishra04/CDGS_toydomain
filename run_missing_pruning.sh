#!/bin/bash
# Script to identify and run missing pruning experiments

# Exit on error
set -e

OUTPUT_DIR="profile/sweep_pruning"
mkdir -p "$OUTPUT_DIR"

# Generate all expected (start, end) pairs where end > start
echo "Generating all expected pruning pairs..."
EXPECTED_PAIRS=()
for start in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    for end in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
        if awk "BEGIN {exit !($end > $start)}"; then
            EXPECTED_PAIRS+=("$start,$end")
        fi
    done
done

echo "Total expected pairs: ${#EXPECTED_PAIRS[@]}"

# Check which pairs already have results
echo "Checking existing results..."

# First, quickly build a list of existing pairs
EXISTING_PAIRS=()
if [ -d "$OUTPUT_DIR" ]; then
    echo "Reading existing JSON files..."
    for json_file in "$OUTPUT_DIR"/*.json; do
        if [ -f "$json_file" ]; then
            # Extract both values in one pass
            START_VAL=$(grep -o '"pruning_start": [0-9.]*' "$json_file" | head -1 | awk '{print $2}')
            END_VAL=$(grep -o '"pruning_end": [0-9.]*' "$json_file" | head -1 | awk '{print $2}')
            if [ -n "$START_VAL" ] && [ -n "$END_VAL" ]; then
                EXISTING_PAIRS+=("$START_VAL,$END_VAL")
            fi
        fi
    done
fi

echo "Found ${#EXISTING_PAIRS[@]} existing results"
echo "Comparing against ${#EXPECTED_PAIRS[@]} expected pairs..."

# Now find missing pairs
MISSING_PAIRS=()
for pair in "${EXPECTED_PAIRS[@]}"; do
    FOUND=0
    for existing in "${EXISTING_PAIRS[@]}"; do
        if [ "$pair" = "$existing" ]; then
            FOUND=1
            break
        fi
    done
    if [ $FOUND -eq 0 ]; then
        MISSING_PAIRS+=("$pair")
    fi
done

echo "Found ${#MISSING_PAIRS[@]} missing pairs"

if [ ${#MISSING_PAIRS[@]} -eq 0 ]; then
    echo "All experiments complete! No missing pairs."
    exit 0
fi

echo "Missing pairs:"
for pair in "${MISSING_PAIRS[@]}"; do
    echo "  $pair"
done

# Ask for confirmation
read -p "Run ${#MISSING_PAIRS[@]} missing experiments? (y/n) " -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Run missing experiments
echo "Starting to run ${#MISSING_PAIRS[@]} missing experiments..."
echo "MISSING_PAIRS array contains: ${MISSING_PAIRS[@]}"
echo ""
COMPLETED=0
FAILED=0

for pair in "${MISSING_PAIRS[@]}"; do
    PRUNING_START=$(echo $pair | cut -d',' -f1)
    PRUNING_END=$(echo $pair | cut -d',' -f2)

    echo "========================================="
    echo "[$((COMPLETED + FAILED + 1))/${#MISSING_PAIRS[@]}] Running: start=$PRUNING_START, end=$PRUNING_END"
    echo "========================================="

    set +e  # Temporarily disable exit on error
    uv run sampling_time_single.py \
        --horizon-length 10 \
        --num-resampling-steps 10 \
        --enable-pruning True \
        --pruning-start $PRUNING_START \
        --pruning-end $PRUNING_END \
        --output-directory "$OUTPUT_DIR" 0<&-
    EXIT_CODE=$?

    echo "Experiment completed with exit code: $EXIT_CODE"

    if [ $EXIT_CODE -eq 0 ]; then
        COMPLETED=$((COMPLETED + 1))
        echo "  ✓ Success (total completed: $COMPLETED)"
    else
        FAILED=$((FAILED + 1))
        echo "  ✗ Failed with exit code $EXIT_CODE (total failed: $FAILED)"
    fi

    set -e  # Re-enable exit on error

    echo "Moving to next experiment..."
    echo ""
done

echo "Loop finished!"

echo ""
echo "===================="
echo "Summary:"
echo "  Completed: $COMPLETED"
echo "  Failed: $FAILED"
echo "  Total: ${#MISSING_PAIRS[@]}"
echo "===================="
