#!/bin/bash
set -euo pipefail

INPUT_VIDEO="/hdd_data/nakul/soham/video-data/videos/00133.mp4"
OUTPUT_DIR="/hdd_data/nakul/soham/video-data/out_dir"
VIDEO_NAME="$(basename "$INPUT_VIDEO")"
VIDEO_STEM="${VIDEO_NAME%.*}"

if [[ "$OUTPUT_DIR" = /* ]]; then
        OUTPUT_BASE="$OUTPUT_DIR"
else
        OUTPUT_BASE="data/${OUTPUT_DIR#data/}"
fi

FLOW_ROOT="${OUTPUT_BASE}/${VIDEO_STEM}"
TRAJECTORY_DIR="${FLOW_ROOT}/trajectories"

mkdir -p "${FLOW_ROOT}"
mkdir -p "${TRAJECTORY_DIR}"

python video-manipulation/preprocess/1_preprocess_video.py \
        --input "$INPUT_VIDEO" \
        --output "$OUTPUT_DIR" \
        --visualize

python video-manipulation/preprocess/2_track_trajectories.py \
        --input "$FLOW_ROOT" \
        --use-flow-magnitude \
        --magnitude-threshold 0.5

python video-manipulation/preprocess/3_analyze_trajectories_fft.py \
        --input "${TRAJECTORY_DIR}/trajectories.npy" \
        --fps 150 \
        --visualize \
        --save-fft

python video-manipulation/preprocess/4_evaluate_candidate_modes.py \
        --input "${TRAJECTORY_DIR}" \
        --visualize \
        --save-modes