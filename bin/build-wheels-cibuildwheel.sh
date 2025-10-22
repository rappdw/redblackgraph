#!/usr/bin/env bash
#
# Build wheels using cibuildwheel for RedBlackGraph
# This script builds wheels for the current platform only
# For multi-platform builds, use the GitHub Actions workflow
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}RedBlackGraph Wheel Builder${NC}"
echo "======================================"

# Check if cibuildwheel is installed
if ! command -v cibuildwheel &> /dev/null; then
    echo -e "${YELLOW}cibuildwheel not found. Installing...${NC}"
    pip install cibuildwheel
fi

# Clean previous builds
echo -e "${GREEN}Cleaning previous builds...${NC}"
rm -rf ./wheelhouse
rm -rf ./build
rm -rf ./dist

# Create output directory
mkdir -p ./wheelhouse

# Run cibuildwheel
echo -e "${GREEN}Building wheels with cibuildwheel...${NC}"
echo "This will build wheels for the current platform only."
echo ""

# Build wheels for current platform
cibuildwheel --platform auto --output-dir wheelhouse

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Wheels built successfully!${NC}"
    echo ""
    echo "Built wheels:"
    ls -lh wheelhouse/*.whl 2>/dev/null || echo "No wheels found"
    echo ""
    echo "To test a wheel locally:"
    echo "  pip install wheelhouse/<wheel-name>.whl"
    echo ""
    echo "To upload to Test PyPI:"
    echo "  twine upload --repository testpypi wheelhouse/*"
    echo ""
    echo "To upload to PyPI:"
    echo "  twine upload wheelhouse/*"
else
    echo -e "${RED}✗ Wheel build failed${NC}"
    exit 1
fi
