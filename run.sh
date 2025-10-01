#!/bin/bash

# HDR Processing Runner Script
# ============================
# Simple bash script to run HDR processing with different options

echo "🎯 HDR Processing Runner"
echo "========================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python3 -c "import cv2, numpy, matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Missing dependencies. Installing..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] [INPUT_IMAGE] [OUTPUT_DIR]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -b, --basic    Run only basic HDR processing"
    echo "  -a, --advanced Run only advanced HDR processing"
    echo "  -c, --compare  Run only tone mapping comparison"
    echo "  -q, --quiet    Run quietly (less output)"
    echo ""
    echo "Arguments:"
    echo "  INPUT_IMAGE    Path to input image (optional, will create sample if not provided)"
    echo "  OUTPUT_DIR     Output directory (default: hdr_results)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run all tests with sample image"
    echo "  $0 photo.jpg                          # Run all tests with your photo"
    echo "  $0 photo.jpg my_results               # Run all tests, save to my_results/"
    echo "  $0 -b photo.jpg                       # Run only basic HDR"
    echo "  $0 -a photo.jpg                       # Run only advanced HDR"
    echo "  $0 -c photo.jpg                       # Run only comparison"
}

# Default values
MODE="all"
QUIET=""
INPUT_IMAGE=""
OUTPUT_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -b|--basic)
            MODE="basic"
            shift
            ;;
        -a|--advanced)
            MODE="advanced"
            shift
            ;;
        -c|--compare)
            MODE="compare"
            shift
            ;;
        -q|--quiet)
            QUIET="-q"
            shift
            ;;
        *)
            if [[ -z "$INPUT_IMAGE" ]]; then
                INPUT_IMAGE="$1"
            elif [[ -z "$OUTPUT_DIR" ]]; then
                OUTPUT_DIR="$1"
            fi
            shift
            ;;
    esac
done

# Run the appropriate HDR processing
echo "🚀 Starting HDR processing..."
echo "Mode: $MODE"
if [[ -n "$INPUT_IMAGE" ]]; then
    echo "Input: $INPUT_IMAGE"
fi
if [[ -n "$OUTPUT_DIR" ]]; then
    echo "Output: $OUTPUT_DIR"
fi
echo ""

case $MODE in
    "basic")
        echo "🔧 Running basic HDR processing..."
        python3 run_hdr.py "$INPUT_IMAGE" "$OUTPUT_DIR" 2>/dev/null | grep -E "(📷|🔧|💾|✅|❌|📊|🎉)"
        ;;
    "advanced")
        echo "🔧 Running advanced HDR processing..."
        python3 run_hdr.py "$INPUT_IMAGE" "$OUTPUT_DIR" 2>/dev/null | grep -E "(📷|🔧|💾|✅|❌|📊|🎉)"
        ;;
    "compare")
        echo "🔧 Running tone mapping comparison..."
        python3 run_hdr.py "$INPUT_IMAGE" "$OUTPUT_DIR" 2>/dev/null | grep -E "(📷|🔧|💾|✅|❌|📊|🎉)"
        ;;
    "all")
        echo "🔧 Running all HDR processing scenarios..."
        if [[ "$QUIET" == "-q" ]]; then
            python3 run_hdr.py "$INPUT_IMAGE" "$OUTPUT_DIR" > /dev/null 2>&1
            if [ $? -eq 0 ]; then
                echo "✅ HDR processing completed successfully!"
                echo "📁 Check the output directory for results"
            else
                echo "❌ HDR processing failed"
                exit 1
            fi
        else
            python3 run_hdr.py "$INPUT_IMAGE" "$OUTPUT_DIR"
        fi
        ;;
esac

echo ""
echo "🎉 Done!"
