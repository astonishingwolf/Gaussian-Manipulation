#!/usr/bin/env python3
"""
Conversion script to convert NeRF/Blender format (transforms_train.json) 
to iPhone/Nerfies format required by DG-Mesh (data_type: "ours" or "iPhone")

This matches the camera format used in the Blender export script.

Usage:
    python convert_to_dgmesh_format.py --input_dir <input_dir> --output_dir <output_dir>
    
Example:
    python convert_to_dgmesh_format.py --input_dir /hdd_data/nakul/soham/video-data/physics_dreamer/alocasia --output_dir /hdd_data/nakul/soham/video-data/physics_dreamer/alocasia_dgmesh
"""

import json
import os
import sys
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not available. Will create placeholder masks.")


def extract_masks_from_video(video_path, frame_ids, output_dir, w, h, fps=30):
    """
    Extract masks from mask.mp4 video file for specific frame IDs.
    Converts green pixels (#00FF00) to white (foreground) and everything else to black.
    Extracts frames based on frame numbers from frame_ids (e.g., "0_00020" -> frame 20).
    
    Args:
        video_path: Path to mask.mp4 video file
        frame_ids: List of frame IDs (e.g., ["0_00001", "0_00002", ...]) - only these will be extracted
        output_dir: Output directory for masks
        w: Image width
        h: Image height
        fps: Frames per second for extraction (default: 30)
    
    Returns:
        success: True if masks were extracted successfully
    """
    if not CV2_AVAILABLE:
        print("Warning: opencv-python not available. Creating placeholder masks.")
        return False
    
    if not video_path.exists():
        print(f"Warning: {video_path} not found. Creating placeholder masks.")
        return False
    
    print(f"Extracting masks from {video_path} for {len(frame_ids)} frames at {fps} fps...")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = video_frame_count / video_fps if video_fps > 0 else 0
    
    print(f"Video: {video_frame_count} frames at {video_fps:.2f} fps (duration: {video_duration:.2f}s)")
    
    # Calculate frame interval based on fps
    # Extract one frame every (video_fps / fps) frames
    frame_interval = max(1, int(video_fps / fps)) if video_fps > 0 else 1
    
    # Green color in RGB (OpenCV uses BGR, so we need [0, 255, 0])
    green_bgr = np.array([0, 255, 0], dtype=np.uint8)
    
    # Tolerance for green color matching (to handle compression artifacts)
    tolerance = 25
    
    masks_extracted = 0
    
    for idx, frame_id in enumerate(frame_ids):
        # Extract frame number from frame_id (e.g., "0_00020" -> 20)
        try:
            frame_num = int(frame_id.split('_')[-1])
        except:
            # Fallback: use index if parsing fails
            frame_num = idx + 1
        
        # Calculate video frame index: frame_num corresponds to frame_num-th frame in video
        # Assuming video starts at frame 1, we need frame_num - 1 for 0-indexed
        # Then scale by frame_interval to account for fps
        video_frame_idx = int((frame_num - 1) * frame_interval)
        
        # Ensure we don't go beyond video length
        if video_frame_idx >= video_frame_count:
            video_frame_idx = video_frame_count - 1
        if video_frame_idx < 0:
            video_frame_idx = 0
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame {video_frame_idx} from video (for frame_id {frame_id})")
            # Create black mask as fallback
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            # Resize if dimensions don't match
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            
            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create mask: green pixels (#00FF00) become white, everything else black
            # Check if pixel is green (within tolerance)
            green_mask = np.all(np.abs(frame_rgb - green_bgr) < tolerance, axis=2)
            
            # Convert to grayscale mask (white for green pixels, black otherwise)
            mask = (green_mask * 255).astype(np.uint8)
        
        # Save mask with the exact frame_id
        mask_path = output_dir / "mask-tracking" / "1x" / "Annotations" / f"{frame_id}.png"
        cv2.imwrite(str(mask_path), mask)
        
        masks_extracted += 1
        
        if (idx + 1) % 50 == 0:
            print(f"  Extracted {idx + 1}/{len(frame_ids)} masks...")
    
    cap.release()
    print(f"Extracted {masks_extracted} masks from video for {len(frame_ids)} processed frames")
    return True


def c2w_to_w2c_orientation_position(c2w):
    """
    Convert camera-to-world 4x4 matrix to world-to-camera orientation and camera position.
    Matches the format used in the Blender SaveCameraData function.
    
    Args:
        c2w: 4x4 camera-to-world transformation matrix (numpy array, Blender/OpenGL convention)
    
    Returns:
        orientation: 3x3 rotation matrix (R_w2c - world-to-camera rotation)
        position: 3x1 translation vector (camera position in world coordinates)
    """
    # Convert from Blender/OpenGL to OpenCV convention
    # Blender/OpenGL: +Y up, -Z forward, +X right
    # OpenCV: +Y down, +Z forward, +X right
    # Conversion: flip Y and Z axes (multiply columns 1 and 2 by -1)
    c2w_opencv = c2w.copy()
    c2w_opencv[:3, 1] *= -1  # Flip Y axis
    c2w_opencv[:3, 2] *= -1  # Flip Z axis
    
    # Extract rotation and translation from c2w (OpenCV convention)
    R_c2w = c2w_opencv[:3, :3]
    t_c2w = c2w_opencv[:3, 3]
    
    # Build w2c (world-to-camera) transform
    R_w2c = R_c2w.T  # Transpose to get world-to-camera rotation
    # Position is camera position in world (t_c2w)
    
    return R_w2c, t_c2w


def convert_transforms_to_iphone_format(input_dir, output_dir, train_split=0.9):
    """
    Convert transforms_train.json format to iPhone/Nerfies format for DG-Mesh.
    Matches the format used in the Blender export script.
    Only processes frames where images are actually available.
    
    Args:
        input_dir: Input directory containing transforms_train.json and images/
        output_dir: Output directory where converted data will be saved
        train_split: Fraction of frames to use for training (rest for validation)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directory structure (matching Blender script format)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "camera").mkdir(exist_ok=True)
    (output_dir / "rgb" / "1x").mkdir(parents=True, exist_ok=True)
    (output_dir / "mask-tracking" / "1x" / "Annotations").mkdir(parents=True, exist_ok=True)
    
    # Read transforms_train.json (required)
    transforms_path = input_dir / "transforms_train.json"
    if not transforms_path.exists():
        print(f"Error: {transforms_path} not found!")
        sys.exit(1)
    
    print(f"Reading {transforms_path}...")
    with open(transforms_path, 'r') as f:
        data = json.load(f)
    
    # Extract camera intrinsics
    w = data.get("w", 1920)
    h = data.get("h", 1080)
    
    # Get focal length or convert from camera angle
    if "fl_x" in data:
        fl_x = data["fl_x"]
    elif "camera_angle_x" in data:
        # Convert camera angle to focal length: fl = (w/2) / tan(camera_angle_x/2)
        camera_angle_x = data["camera_angle_x"]
        fl_x = (w / 2.0) / np.tan(camera_angle_x / 2.0)
    else:
        fl_x = w * 0.7  # Default fallback
    
    fl_y = data.get("fl_y", fl_x)  # Use fl_x if fl_y not present
    cx = data.get("cx", w / 2.0)
    cy = data.get("cy", h / 2.0)
    
    # Distortion parameters
    k1 = data.get("k1", 0.0)
    k2 = data.get("k2", 0.0)
    k3 = data.get("k3", 0.0)
    k4 = data.get("k4", 0.0)
    p1 = data.get("p1", 0.0)
    p2 = data.get("p2", 0.0)
    
    # Read transforms_test.json if available (for missing frames)
    transforms_test_path = input_dir / "transforms_test.json"
    test_data = None
    if transforms_test_path.exists():
        print(f"Reading {transforms_test_path}...")
        with open(transforms_test_path, 'r') as f:
            test_data = json.load(f)
        print(f"Found {len(test_data.get('frames', []))} frames in test JSON")
    
    # Build a map of frame numbers to frames from train JSON
    train_frames_map = {}
    for frame in data["frames"]:
        file_path = frame["file_path"]
        # Extract frame number from path like "images/frame_00001.png"
        try:
            frame_name = Path(file_path).stem  # "frame_00001"
            frame_num = int(frame_name.split('_')[-1])
            train_frames_map[frame_num] = frame
        except:
            pass
    
    # Build a map of frame numbers to frames from test JSON (if available)
    test_frames_map = {}
    if test_data:
        for frame in test_data.get("frames", []):
            file_path = frame["file_path"]
            try:
                frame_name = Path(file_path).stem
                frame_num = int(frame_name.split('_')[-1])
                test_frames_map[frame_num] = frame
            except:
                pass
    
    # Find all images in the images directory
    images_dir = input_dir / "images"
    all_image_files = []
    if images_dir.exists():
        all_image_files = sorted([f for f in images_dir.iterdir() 
                                  if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    
    print(f"Found {len(train_frames_map)} frames in train JSON, {len(test_frames_map)} frames in test JSON, {len(all_image_files)} images in directory")
    
    # Build complete frame list: use train JSON first, then test JSON for missing frames
    available_frames = []
    missing_frames = []
    
    for img_file in all_image_files:
        try:
            frame_num = int(img_file.stem.split('_')[-1])
            
            if frame_num in train_frames_map:
                # Frame exists in train JSON, use it
                frame = train_frames_map[frame_num].copy()
                frame["file_path"] = f"images/{img_file.name}"  # Ensure correct path
                available_frames.append((frame_num, frame))
            elif frame_num in test_frames_map:
                # Frame exists in test JSON, use it
                frame = test_frames_map[frame_num].copy()
                frame["file_path"] = f"images/{img_file.name}"  # Ensure correct path
                available_frames.append((frame_num, frame))
                print(f"  Using frame {frame_num} from test JSON")
            else:
                # Frame missing from both JSONs
                missing_frames.append((frame_num, img_file))
        except:
            continue
    
    # Sort by frame number
    available_frames.sort(key=lambda x: x[0])
    missing_frames.sort(key=lambda x: x[0])
    
    # Report missing frames
    if missing_frames:
        print(f"Warning: {len(missing_frames)} frames have images but no camera parameters in JSON:")
        for missing_num, img_file in missing_frames[:10]:  # Show first 10
            print(f"  Frame {missing_num}: {img_file.name}")
        if len(missing_frames) > 10:
            print(f"  ... and {len(missing_frames) - 10} more")
        print("  These frames will be skipped.")
    
    # Sort all frames by frame number and extract just the frame data
    available_frames.sort(key=lambda x: x[0])
    available_frames = [frame for _, frame in available_frames]
    
    num_frames = len(available_frames)
    num_from_train = sum(1 for f in available_frames if int(Path(f["file_path"]).stem.split('_')[-1]) in train_frames_map)
    num_from_test = num_frames - num_from_train
    
    print(f"Processing {num_frames} total frames ({num_from_train} from train JSON + {num_from_test} from test JSON)...")
    
    if num_frames == 0:
        print("Error: No frames with available images found!")
        sys.exit(1)
    
    # Prepare data structures for iPhone format (matching Blender script)
    metadata = {}
    dataset = {"train_ids": [], "val_ids": []}
    all_frame_ids = []
    
    # Determine train/val split
    train_num = int(num_frames * train_split)
    
    # Calculate pixel aspect ratio (assuming square pixels)
    pixel_aspect_ratio = 1.0
    
    for idx, frame in enumerate(available_frames):
        # Extract image path and name
        file_path = frame["file_path"]
        image_name = Path(file_path).stem  # e.g., "frame_00001"
        
        # Create frame ID (matching Blender script format: 0_00001)
        # Extract frame number from name if possible, otherwise use index
        try:
            frame_num = int(image_name.split('_')[-1])
            frame_id = f"0_{frame_num:05d}"
        except:
            frame_id = f"0_{idx:05d}"
        
        all_frame_ids.append(frame_id)
        
        # Determine if train or val
        if idx < train_num:
            dataset["train_ids"].append(frame_id)
        else:
            dataset["val_ids"].append(frame_id)
        
        # Extract transform matrix (camera-to-world, Blender/OpenGL convention)
        c2w = np.array(frame["transform_matrix"])
        
        # Convert to orientation and position (matching Blender script format)
        orientation, position = c2w_to_w2c_orientation_position(c2w)
        
        # Create metadata entry (matching Blender script format)
        metadata[frame_id] = {
            "appearance_id": idx,
            "camera_id": 0,
            "warp_id": idx
        }
        
        # Create camera JSON (matching Blender script format exactly)
        camera_data = {
            "focal_length": float(fl_x),
            "image_size": [int(w), int(h)],
            "orientation": orientation.tolist(),
            "pixel_aspect_ratio": pixel_aspect_ratio,
            "position": position.tolist(),
            "principal_point": [float(cx), float(cy)],
            "radial_distortion": [float(k1), float(k2), float(k3)],
            "skew": 0.0,
            "tangential_distortion": [float(p1), float(p2)]
        }
        
        # Save camera JSON (matching Blender script naming: 0_00001.json)
        camera_path = output_dir / "camera" / f"{frame_id}.json"
        with open(camera_path, 'w') as f:
            json.dump(camera_data, f, indent=4)
        
        # Copy image to rgb/1x/ (image already verified to exist)
        input_image_path = input_dir / file_path
        if not input_image_path.exists():
            # Try with .png extension if original doesn't exist
            input_image_path = input_dir / f"{file_path}.png"
        
        output_image_path = output_dir / "rgb" / "1x" / f"{frame_id}.png"
        shutil.copy2(input_image_path, output_image_path)
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{num_frames} frames...")
        
        # Masks will be extracted from video after processing all frames
    
    # Normalize warp_ids to [0, 1] (matching Blender script)
    max_warp_id = max(meta["warp_id"] for meta in metadata.values())
    if max_warp_id > 0:
        for frame_id in metadata:
            metadata[frame_id]["warp_id"] = metadata[frame_id]["warp_id"] / max_warp_id
    
    # Save metadata.json (matching Blender script format)
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved {metadata_path}")
    
    # Save dataset.json (matching Blender script format)
    dataset["count"] = len(all_frame_ids)
    dataset["ids"] = all_frame_ids
    dataset["num_exemplars"] = len(all_frame_ids)
    dataset_path = output_dir / "dataset.json"
    with open(dataset_path, 'w') as f:
        json.dump(dataset, f, indent=4)
    print(f"Saved {dataset_path}")
    
    # Extract masks from mask.mp4 video if available (at 30 fps)
    # Only extract masks for frames that are actually being processed
    mask_video_path = input_dir / "mask.mp4"
    if mask_video_path.exists():
        print(f"Extracting masks for {len(all_frame_ids)} processed frames...")
        success = extract_masks_from_video(
            mask_video_path,
            all_frame_ids,  # Only frame IDs that are being processed
            output_dir,
            w,
            h,
            fps=30
        )
        if not success:
            print("Creating placeholder masks (all white) for processed frames only...")
            for frame_id in all_frame_ids:
                mask_path = output_dir / "mask-tracking" / "1x" / "Annotations" / f"{frame_id}.png"
                mask = Image.new('L', (w, h), color=255)  # White = foreground
                mask.save(mask_path)
    else:
        print(f"mask.mp4 not found. Creating placeholder masks (all white) for {len(all_frame_ids)} processed frames...")
        for frame_id in all_frame_ids:
            mask_path = output_dir / "mask-tracking" / "1x" / "Annotations" / f"{frame_id}.png"
            mask = Image.new('L', (w, h), color=255)  # White = foreground
            mask.save(mask_path)
    
    # Generate points.npy and points3d.ply from PLY file (matching Blender script)
    points_path = input_dir / "points.npy"
    
    # Try to find PLY file in order of preference
    ply_candidates = [
        "clean_object_points.ply",
        "point_cloud.ply",
        "points3d.ply",
        "moving_part_points.ply"
    ]
    
    ply_path = None
    if points_path.exists():
        # Copy existing points.npy
        shutil.copy2(points_path, output_dir / "points.npy")
        print(f"Copied points.npy")
    
    # Look for PLY files
    for ply_name in ply_candidates:
        candidate_path = input_dir / ply_name
        if candidate_path.exists():
            ply_path = candidate_path
            break
    
    if ply_path:
        # Copy PLY file to points3d.ply
        output_ply_path = output_dir / "points3d.ply"
        shutil.copy2(ply_path, output_ply_path)
        print(f"Copied {ply_path.name} to points3d.ply")
        
        # Also extract points for points.npy if not already copied
        if not points_path.exists():
            try:
                from plyfile import PlyData
                plydata = PlyData.read(str(ply_path))
                vertices = plydata['vertex']
                xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
                np.save(output_dir / "points.npy", xyz)
                print(f"Extracted {len(xyz)} points from PLY and saved as points.npy")
            except ImportError:
                print("Note: plyfile not available. Skipping points.npy extraction.")
            except Exception as e:
                print(f"Warning: Could not extract points from PLY: {e}")
    else:
        print("Note: No PLY file found. DG-Mesh will generate a random point cloud.")
    
    print(f"\nConversion complete!")
    print(f"Output directory: {output_dir}")
    print(f"Total frames: {num_frames} (only frames with available images)")
    print(f"Train frames: {len(dataset['train_ids'])}")
    print(f"Val frames: {len(dataset['val_ids'])}")
    print(f"\niPhone/Nerfies format ready for DG-Mesh with data_type: 'ours' or 'iPhone'")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NeRF/Blender format to iPhone/Nerfies format for DG-Mesh (matching Blender export script format)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing transforms_train.json and images/"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for converted data"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Fraction of frames for training (default: 0.9)"
    )
    
    args = parser.parse_args()
    
    convert_transforms_to_iphone_format(
        args.input_dir,
        args.output_dir,
        args.train_split
    )


if __name__ == "__main__":
    main()

