#!/bin/bash
set -euo pipefail

# Preprocessing script for Gaussian Manipulation
# Takes a displacement/trajectories file and runs the full preprocessing pipeline
#
# Usage:
#   ./run_preprocess.sh <input_displacement_file> <output_dir> [options]
#
# Example:
#   ./run_preprocess.sh data/displacements.npy data/preprocessed --fps 30 --visualize

# Default values
FPS=30
VISUALIZE=false
TOP_K=5
NUM_MODES=3
PEAK_PROMINENCE=0.05

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_displacement_file> <output_dir> [--fps FPS] [--visualize] [--top-k K] [--num-modes M] [--peak-prominence P]"
    echo ""
    echo "Arguments:"
    echo "  input_displacement_file  Path to input .npy file (displacements or trajectories)"
    echo "  output_dir               Output directory for all preprocessed files"
    echo ""
    echo "Options:"
    echo "  --fps FPS                Frames per second (default: 30)"
    echo "  --visualize              Generate visualization plots (default: false)"
    echo "  --top-k K                Number of top frequencies to report (default: 5)"
    echo "  --num-modes M            Number of modes to select (default: 3)"
    echo "  --peak-prominence P      Minimum prominence for peak detection (default: 0.05)"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_DIR="$2"
shift 2

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fps)
            FPS="$2"
            shift 2
            ;;
        --visualize)
            VISUALIZE=true
            shift
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --num-modes)
            NUM_MODES="$2"
            shift 2
            ;;
        --peak-prominence)
            PEAK_PROMINENCE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate input file
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INPUT_FILE_ABS="$(cd "$(dirname "$INPUT_FILE")" && pwd)/$(basename "$INPUT_FILE")"
OUTPUT_DIR_ABS="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"

echo "=========================================="
echo "Gaussian Manipulation Preprocessing"
echo "=========================================="
echo "Input file:  $INPUT_FILE_ABS"
echo "Output dir:  $OUTPUT_DIR_ABS"
echo "FPS:         $FPS"
echo "Visualize:   $VISUALIZE"
echo "=========================================="
echo ""

# Step 1: Convert displacements to trajectories if needed
# Check if the file contains displacements (timesteps, N, 3) or trajectories (T, N, 3) or (N, T, 3)
echo "Step 1: Loading and validating input file..."
TRAJECTORIES_FILE="${OUTPUT_DIR_ABS}/trajectories.npy"

python3 << EOF
import numpy as np
from pathlib import Path

input_file = Path("$INPUT_FILE_ABS")
output_dir = Path("$OUTPUT_DIR_ABS")
trajectories_file = Path("$TRAJECTORIES_FILE")

# Load the input file
data = np.load(input_file)
print(f"Loaded input file: shape {data.shape}")

# Determine if it's displacements or trajectories
# Displacements: (timesteps, N, 3) - relative movements
# Trajectories: (T, N, 3) or (N, T, 3) - absolute positions

if len(data.shape) != 3 or data.shape[2] != 3:
    raise ValueError(f"Expected 3D array with last dimension 3, got shape {data.shape}")

# Check if it looks like displacements (small values, centered around zero)
# or trajectories (larger absolute values)
mean_abs = np.mean(np.abs(data))
std_abs = np.std(data)
mean_val = np.mean(data)

print(f"Mean absolute value: {mean_abs:.6f}")
print(f"Std absolute value: {std_abs:.6f}")
print(f"Mean value: {mean_val:.6f}")

# Heuristic: if mean absolute value is very small (< 0.1) and mean is close to zero,
# it's likely displacements. Otherwise, assume it's trajectories
if mean_abs < 0.1 and abs(mean_val) < 0.05:
    print("Detected as displacements - converting to trajectories...")
    # Determine current format and convert to (N, T, 3) for processing
    if data.shape[0] < data.shape[1]:
        # Currently (T, N, 3) - transpose to (N, T, 3)
        data_n_t_3 = data.transpose(1, 0, 2)
        print(f"Transposed from (T, N, 3) to (N, T, 3): {data_n_t_3.shape}")
    else:
        # Already (N, T, 3)
        data_n_t_3 = data
        print(f"Already in (N, T, 3) format: {data_n_t_3.shape}")
    
    # Cumulative sum along time axis (axis=1) to get trajectories
    trajectories_n_t_3 = np.cumsum(data_n_t_3, axis=1)
    # Convert back to (T, N, 3) format for compatibility with preprocessing scripts
    trajectories = trajectories_n_t_3.transpose(1, 0, 2)
    print(f"Converted to trajectories: shape {trajectories.shape} (T, N, 3)")
else:
    print("Detected as trajectories - using as-is...")
    trajectories = data
    # Ensure format is (T, N, 3) for preprocessing scripts
    if trajectories.shape[0] < trajectories.shape[1]:
        # Currently (N, T, 3), transpose to (T, N, 3)
        trajectories = trajectories.transpose(1, 0, 2)
        print(f"Transposed from (N, T, 3) to (T, N, 3): {trajectories.shape}")
    else:
        print(f"Already in (T, N, 3) format: {trajectories.shape}")

# Save trajectories
np.save(trajectories_file, trajectories)
print(f"✓ Saved trajectories to {trajectories_file}")
print(f"  Final shape: {trajectories.shape} (T, N, 3)")
EOF

if [ $? -ne 0 ]; then
    echo "Error: Failed to process input file"
    exit 1
fi

echo ""

# Step 2: Run FFT analysis
echo "Step 2: Running FFT analysis on trajectories..."
FFT_ARGS=(
    --input "$TRAJECTORIES_FILE"
    --output "$OUTPUT_DIR_ABS"
    --fps "$FPS"
    --top-k "$TOP_K"
    --save-fft
)

if [ "$VISUALIZE" = true ]; then
    FFT_ARGS+=(--visualize)
fi

python3 "${SCRIPT_DIR}/1_analyze_trajectories_fft.py" "${FFT_ARGS[@]}"

if [ $? -ne 0 ]; then
    echo "Error: FFT analysis failed"
    exit 1
fi

echo ""

# Step 3: Evaluate candidate modes
echo "Step 3: Evaluating candidate frequency modes..."
MODE_ARGS=(
    --trajectory_dir "$OUTPUT_DIR_ABS"
    --num-modes "$NUM_MODES"
    --peak-prominence "$PEAK_PROMINENCE"
    --fps "$FPS"
    --save-modes
)

if [ "$VISUALIZE" = true ]; then
    MODE_ARGS+=(--visualize)
fi

python3 "${SCRIPT_DIR}/2_evaluate_candidate_modes.py" "${MODE_ARGS[@]}"

if [ $? -ne 0 ]; then
    echo "Error: Mode evaluation failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Preprocessing complete!"
echo "=========================================="
echo ""
echo "Output files in ${OUTPUT_DIR_ABS}:"
echo "  - trajectories.npy              (input trajectories)"
echo "  - fft_x.npy, fft_y.npy, fft_z.npy (complex FFT results)"
echo "  - power_x.npy, power_y.npy, power_z.npy (power spectra)"
echo "  - frequencies.npy               (frequency bins)"
echo "  - fft_summary.npy               (FFT analysis summary)"
if [ "$VISUALIZE" = true ]; then
    echo "  - fft_analysis.png            (FFT visualization)"
fi
echo "  - modes/selected_modes.npy      (selected frequency modes)"
if [ "$VISUALIZE" = true ]; then
    echo "  - modes/candidate_modes_visualization.png (mode visualization)"
fi
echo ""
echo "Next step: Run gaussian_manipulation.py with --trajectory-dir ${OUTPUT_DIR_ABS}"
echo ""

