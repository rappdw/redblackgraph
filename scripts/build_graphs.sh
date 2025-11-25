#!/bin/bash
# Build graphs for hop counts 4 through 14

set -e  # Exit on error

VENV_PYTHON="/home/rappdw/dev/redblackgraph/.venv-rbg-gpu/bin/python"
BUILDER="/home/rappdw/dev/redblackgraph/.venv-rbg-gpu/bin/rbg-graph-builder"
CANONICALIZER="/home/rappdw/dev/redblackgraph/scripts/compute_canonical_forms.py"
EXPORTER="/home/rappdw/dev/redblackgraph/scripts/export_canonical_graph.py"
source "/home/rappdw/dev/redblackgraph/.venv-rbg-gpu/bin/activate"

echo "Building graphs for hop counts 4-14..."
echo "========================================"

for hopcount in {2..14}; do
    echo ""
    echo "Processing hop count: $hopcount"
    echo "----------------------------------------"
    
    start_time=$(date +%s)
    
    if $BUILDER -c $hopcount; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "✓ Hop count $hopcount completed in ${duration}s"
    else
        echo "✗ Hop count $hopcount failed"
        exit 1
    fi
    
    start_time=$(date +%s)

    if $CANONICALIZER --hop-count $hopcount; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "✓ Hop count $hopcount completed in ${duration}s"
    else
        echo "✗ Hop count $hopcount failed"
        exit 1
    fi

    start_time=$(date +%s)

    if $EXPORTER --hop-count $hopcount; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "✓ Hop count $hopcount completed in ${duration}s"
    else
        echo "✗ Hop count $hopcount failed"
        exit 1
    fi

done

echo ""
echo "========================================"
echo "All graphs built successfully!"
