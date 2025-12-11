#!/usr/bin/env python3
"""
Convert all images in a folder to a video.

Usage:
    python convert_video.py <input_folder> <output_video> [--fps FPS] [--pattern PATTERN]
    
Example:
    python convert_video.py /path/to/images output.mp4 --fps 30
    python convert_video.py /path/to/images output.mp4 --fps 30 --pattern "*.png"
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from glob import glob


def images_to_video(input_folder, output_video, fps=30, pattern="*.png", sort=True):
    """
    Convert all images in a folder to a video.
    
    Args:
        input_folder: Path to folder containing images
        output_video: Path to output video file
        fps: Frames per second for the video
        pattern: Glob pattern to match image files (default: "*.png")
        sort: Whether to sort images by filename (default: True)
    """
    input_folder = Path(input_folder)
    
    if not input_folder.exists():
        print(f"Error: Input folder '{input_folder}' does not exist!")
        return False
    
    # Find all image files
    image_pattern = str(input_folder / pattern)
    image_files = glob(image_pattern)
    
    # Also try common image extensions
    if not image_files:
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]:
            image_files = glob(str(input_folder / ext))
            if image_files:
                break
    
    if not image_files:
        print(f"Error: No images found in '{input_folder}' with pattern '{pattern}'")
        return False
    
    # Sort images by filename
    if sort:
        image_files = sorted(image_files)
    
    print(f"Found {len(image_files)} images")
    print(f"First image: {Path(image_files[0]).name}")
    print(f"Last image: {Path(image_files[-1]).name}")
    
    # Read first image to get dimensions
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"Error: Could not read first image '{image_files[0]}'")
        return False
    
    height, width, channels = first_image.shape
    print(f"Image dimensions: {width}x{height}")
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for '{output_video}'")
        return False
    
    # Write all images to video
    print(f"Writing video at {fps} fps...")
    for i, image_path in enumerate(image_files):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image '{image_path}', skipping...")
            continue
        
        # Resize if dimensions don't match
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        out.write(img)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(image_files)} images...")
    
    # Release everything
    out.release()
    print(f"Video saved to: {output_video}")
    print(f"Total frames: {len(image_files)}")
    print(f"Duration: {len(image_files) / fps:.2f} seconds")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert all images in a folder to a video"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        help="Input folder containing images"
    )
    parser.add_argument(
        "--output_video",
        type=str,
        help="Output video file path (e.g., output.mp4)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second for the video (default: 30)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="Glob pattern to match image files (default: *.png)"
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Don't sort images by filename"
    )
    
    args = parser.parse_args()
    
    success = images_to_video(
        args.input_folder,
        args.output_video,
        fps=args.fps,
        pattern=args.pattern,
        sort=not args.no_sort
    )
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
