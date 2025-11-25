"""
Track pixel trajectories from optical flow and save as numpy arrays.

Usage:
    # Track from moving pixels
    python video-manipulation/preprocess/2_track_trajectories.py \
        --input data/optical_flows/00376 \
        --use-flow-magnitude --magnitude-threshold 0.5
    
    # Track from grid
    python video-manipulation/preprocess/2_track_trajectories.py \
        --input data/optical_flows/00376 \
        --sample-grid 20 20
    
    # Visualize trajectories
    python video-manipulation/preprocess/2_track_trajectories.py \
        --input data/optical_flows/00376 \
        --use-flow-magnitude --visualize
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_flow_files(flow_dir):
    """Load all optical flow .pt files from a directory."""
    flow_dir = Path(flow_dir)
    flow_files = sorted(flow_dir.glob("*.pt"))
    
    if not flow_files:
        raise FileNotFoundError(f"No .pt files found in {flow_dir}")
    
    flows = []
    for flow_file in tqdm(flow_files, desc="Loading flows"):
        flow = torch.load(flow_file, map_location='cpu')
        flows.append(flow)
    
    return flows, [f.stem for f in flow_files]


def flow_to_numpy(flows):
    """Convert list of flow tensors to numpy array."""
    flow_array = []
    for flow in flows:
        if isinstance(flow, torch.Tensor):
            flow_np = flow.numpy()
        else:
            flow_np = flow
        
        # Ensure shape is (H, W, 2) - [dx, dy]
        if flow_np.ndim == 3 and flow_np.shape[0] == 2:
            flow_np = np.transpose(flow_np, (1, 2, 0))
        
        flow_array.append(flow_np)
    
    return np.array(flow_array)  # Shape: (T, H, W, 2)


def compute_flow_magnitude(flows):
    """Compute magnitude of optical flow vectors."""
    if isinstance(flows, list):
        flows = flow_to_numpy(flows)
    
    # flows shape: (T, H, W, 2)
    magnitude = np.sqrt(flows[..., 0]**2 + flows[..., 1]**2)
    return magnitude  # Shape: (T, H, W)


def track_pixel_trajectories(args,flows, initial_positions=None, sample_grid=None, 
                            use_flow_magnitude=False, magnitude_threshold=0.5, max_points=1000):
    """
    Track pixel trajectories by integrating optical flow over time.
    
    Args:
        flows: numpy array of shape (T, H, W, 2) or list of flow tensors
        initial_positions: array of shape (N, 2) with initial [y, x] positions
        sample_grid: tuple (grid_h, grid_w) to sample points uniformly
        use_flow_magnitude: If True, select initial points based on flow magnitude in first frame
        magnitude_threshold: Minimum flow magnitude to consider (in pixels)
        max_points: Maximum number of points to track when using flow magnitude
    
    Returns:
        trajectories: array of shape (N, T+1, 2) containing [y, x] positions over time
        initial_positions: array of shape (N, 2) with starting positions
    """
    if isinstance(flows, list):
        flows = flow_to_numpy(flows)
    
    T, H, W, _ = flows.shape
    
    # Create initial positions if not provided
    if initial_positions is None:
        if use_flow_magnitude:
            # Select points based on flow magnitude in first frame
            first_flow = flows[0]  # Shape: (H, W, 2)
            magnitude = np.sqrt(first_flow[:, :, 0]**2 + first_flow[:, :, 1]**2)
            
            # Find pixels above threshold
            mask = magnitude > magnitude_threshold
            y_coords, x_coords = np.where(mask)
            
            if len(y_coords) == 0:
                print(f"Warning: No pixels with magnitude > {magnitude_threshold}. Using lower threshold.")
                magnitude_threshold = np.percentile(magnitude, 50)  # Use median
                mask = magnitude > magnitude_threshold
                y_coords, x_coords = np.where(mask)
            
            print(len(y_coords), "pixels found with magnitude >", magnitude_threshold)
            # If too many points, sample based on magnitude
            if len(y_coords) > max_points and args.allow_sampling:
                # Sample points weighted by magnitude
                magnitudes_at_points = magnitude[y_coords, x_coords]
                probabilities = magnitudes_at_points / magnitudes_at_points.sum()
                selected_indices = np.random.choice(
                    len(y_coords), 
                    size=max_points, 
                    replace=False,
                    p=probabilities
                )
                y_coords = y_coords[selected_indices]
                x_coords = x_coords[selected_indices]
            
            initial_positions = np.stack([y_coords, x_coords], axis=1).astype(float)
            print(f"Selected {len(initial_positions)} points with flow magnitude > {magnitude_threshold:.2f}")
            print(f"  Mean magnitude: {magnitude[y_coords, x_coords].mean():.2f} pixels")
            print(f"  Max magnitude: {magnitude[y_coords, x_coords].max():.2f} pixels")
        else:
            # Use uniform grid sampling
            if sample_grid is None:
                sample_grid = (10, 10)  # Default 10x10 grid
            
            grid_h, grid_w = sample_grid
            y_coords = np.linspace(H // (2 * grid_h), H - H // (2 * grid_h), grid_h)
            x_coords = np.linspace(W // (2 * grid_w), W - W // (2 * grid_w), grid_w)
            yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
            initial_positions = np.stack([yy.ravel(), xx.ravel()], axis=1)
    
    N = len(initial_positions)
    trajectories = np.zeros((N, T + 1, 2))  # +1 for initial position
    trajectories[:, 0, :] = initial_positions
    
    print(f"Tracking {N} points across {T} frames...")
    
    for t in tqdm(range(T), desc="Computing trajectories"):
        current_positions = trajectories[:, t, :]
        
        # Get flow at current positions using bilinear interpolation
        for i, (y, x) in enumerate(current_positions):
            # Clamp to image boundaries
            y = np.clip(y, 0, H - 1.001)
            x = np.clip(x, 0, W - 1.001)
            
            # Bilinear interpolation
            y0, x0 = int(y), int(x)
            y1, x1 = min(y0 + 1, H - 1), min(x0 + 1, W - 1)
            
            wy1, wx1 = y - y0, x - x0
            wy0, wx0 = 1 - wy1, 1 - wx1
            
            # Interpolate flow at this position
            flow_here = (wy0 * wx0 * flows[t, y0, x0] +
                        wy0 * wx1 * flows[t, y0, x1] +
                        wy1 * wx0 * flows[t, y1, x0] +
                        wy1 * wx1 * flows[t, y1, x1])
            
            # Update position: flow is [dx, dy]
            new_x = x + flow_here[0]
            new_y = y + flow_here[1]
            
            # Store new position (keep in bounds)
            trajectories[i, t + 1, 0] = np.clip(new_y, 0, H - 1)
            trajectories[i, t + 1, 1] = np.clip(new_x, 0, W - 1)
    
    return trajectories, initial_positions


def compute_trajectory_displacement(trajectories):
    """
    Compute displacement and velocity from trajectories.
    
    Args:
        trajectories: array of shape (N, T, 2)
    
    Returns:
        displacement: Cumulative displacement from start (N, T)
        velocity: Frame-to-frame velocity magnitude (N, T-1)
        total_distance: Total path length for each trajectory (N,)
    """
    # Displacement from initial position
    displacement = np.sqrt(
        (trajectories[:, :, 0] - trajectories[:, 0:1, 0]) ** 2 +
        (trajectories[:, :, 1] - trajectories[:, 0:1, 1]) ** 2
    )
    
    # Frame-to-frame velocity
    diff = np.diff(trajectories, axis=1)  # Shape: (N, T-1, 2)
    velocity = np.sqrt(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2)
    
    # Total path length
    total_distance = np.sum(velocity, axis=1)
    
    return displacement, velocity, total_distance


def visualize_trajectories(trajectories, initial_positions, H, W, output_dir=None, max_trajectories=50):
    """Visualize pixel trajectories."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Only plot a subset if too many
    n_plot = min(len(trajectories), max_trajectories)
    indices = np.linspace(0, len(trajectories) - 1, n_plot, dtype=int)
    
    # Create colormap for trajectories
    colors = plt.cm.rainbow(np.linspace(0, 1, n_plot))
    
    for i, idx in enumerate(indices):
        traj = trajectories[idx]
        # Plot trajectory: x, y
        ax.plot(traj[:, 1], traj[:, 0], '-', alpha=0.6, linewidth=1.5, color=colors[i])
        # Mark start point
        ax.plot(traj[0, 1], traj[0, 0], 'o', markersize=8, color=colors[i], 
                markeredgecolor='black', markeredgewidth=1)
        # Mark end point
        ax.plot(traj[-1, 1], traj[-1, 0], 's', markersize=8, color=colors[i], 
                markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # Invert y-axis to match image coordinates
    ax.set_aspect('equal')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'Pixel Trajectories (showing {n_plot}/{len(trajectories)} trajectories)\n● Start  ■ End')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'trajectories.png', dpi=150)
        print(f"Saved trajectories to {output_dir / 'trajectories.png'}")
    else:
        plt.show()
    
    plt.close()


def visualize_trajectory_statistics(trajectories, output_dir=None):
    """Visualize trajectory displacement and velocity statistics."""
    displacement, velocity, total_distance = compute_trajectory_displacement(trajectories)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Displacement over time (mean + std)
    mean_disp = np.mean(displacement, axis=0)
    std_disp = np.std(displacement, axis=0)
    axes[0, 0].plot(mean_disp, 'b-', linewidth=2, label='Mean')
    axes[0, 0].fill_between(range(len(mean_disp)), 
                            mean_disp - std_disp, 
                            mean_disp + std_disp, 
                            alpha=0.3, label='±1 std')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Displacement from Start (pixels)')
    axes[0, 0].set_title('Mean Displacement Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Velocity over time (mean + std)
    mean_vel = np.mean(velocity, axis=0)
    std_vel = np.std(velocity, axis=0)
    axes[0, 1].plot(mean_vel, 'r-', linewidth=2, label='Mean')
    axes[0, 1].fill_between(range(len(mean_vel)), 
                            mean_vel - std_vel, 
                            mean_vel + std_vel, 
                            alpha=0.3, label='±1 std')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Velocity (pixels/frame)')
    axes[0, 1].set_title('Mean Velocity Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Histogram of total distances
    axes[1, 0].hist(total_distance, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(np.mean(total_distance), color='r', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(total_distance):.1f}')
    axes[1, 0].set_xlabel('Total Distance Traveled (pixels)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Distribution of Total Path Lengths')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Final displacement vs total distance
    final_disp = displacement[:, -1]
    axes[1, 1].scatter(total_distance, final_disp, alpha=0.5, s=20)
    axes[1, 1].plot([0, max(total_distance)], [0, max(total_distance)], 
                    'r--', linewidth=2, label='y=x (straight line)')
    axes[1, 1].set_xlabel('Total Path Length (pixels)')
    axes[1, 1].set_ylabel('Final Displacement (pixels)')
    axes[1, 1].set_title('Path Efficiency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'trajectory_statistics.png', dpi=150)
        print(f"Saved trajectory statistics to {output_dir / 'trajectory_statistics.png'}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Track pixel trajectories from optical flow")
    parser.add_argument("--input", required=True, 
                       help="Path to optical flow directory (e.g., data/optical_flows/00376)")
    parser.add_argument("--flow-subdir", default="flow", 
                       help="Subdirectory containing flow files")
    parser.add_argument("--output", default=None, 
                       help="Output directory (default: input/trajectories)")
    
    # Trajectory tracking options
    parser.add_argument("--use-flow-magnitude", action="store_true", 
                       help="Initialize trajectories from pixels with significant flow in first frame")
    parser.add_argument("--magnitude-threshold", type=float, default=0.5, 
                       help="Minimum flow magnitude threshold in pixels (default: 0.5)")
    parser.add_argument("--max-points", type=int, default=10000, 
                       help="Maximum number of points to track (default: 1000)")
    parser.add_argument("--sample-grid", type=int, nargs=2, default=[10, 10], 
                       metavar=('H', 'W'), 
                       help="Grid size for sampling initial points (default: 10 10)")
    parser.add_argument("--allow-sampling", action="store_true", 
                       help="Allow sampling of points if too many points are found")
    # Visualization options
    parser.add_argument("--visualize", action="store_true", 
                       help="Visualize trajectories and statistics")
    parser.add_argument("--max-trajectories", type=int, default=50, 
                       help="Max trajectories to visualize (default: 50)")
    
    args = parser.parse_args()
    
    # Construct paths
    input_path = Path(args.input)
    flow_dir = input_path / args.flow_subdir
    
    if not flow_dir.exists():
        print(f"Error: Flow directory not found at {flow_dir}")
        return
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_path / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load flows
    print(f"Loading optical flows from {flow_dir}...")
    flows, frame_names = load_flow_files(flow_dir)
    print(f"Loaded {len(flows)} flow frames")
    
    # Convert to numpy
    flows_np = flow_to_numpy(flows)
    print(f"Flow array shape: {flows_np.shape}") 


    magnitude = compute_flow_magnitude(flows_np)
    print(f"\nFlow Statistics:")
    print(f"  Mean magnitude: {np.mean(magnitude):.4f} pixels")
    print(f"  Max magnitude: {np.max(magnitude):.4f} pixels")
    print(f"  Min magnitude: {np.min(magnitude):.4f} pixels")
    print(f"  Std magnitude: {np.std(magnitude):.4f} pixels")
    
    # Track trajectories
    if args.use_flow_magnitude:
        print(f"\nTracking trajectories from pixels with flow magnitude > {args.magnitude_threshold}...")
    else:
        print(f"\nTracking trajectories with {args.sample_grid[0]}x{args.sample_grid[1]} grid...")
    
    trajectories, initial_positions = track_pixel_trajectories(
        args,
        flows_np, 
        sample_grid=tuple(args.sample_grid),
        use_flow_magnitude=args.use_flow_magnitude,
        magnitude_threshold=args.magnitude_threshold,
        max_points=args.max_points
    )
    print(f"Tracked {len(trajectories)} trajectories")
    print(f"Trajectory array shape: {trajectories.shape}")  # (N, T+1, 2)
    
    # Compute trajectory statistics
    displacement, velocity, total_distance = compute_trajectory_displacement(trajectories)
    print(f"\nTrajectory Statistics:")
    print(f"  Mean total distance: {np.mean(total_distance):.2f} pixels")
    print(f"  Max total distance: {np.max(total_distance):.2f} pixels")
    print(f"  Mean final displacement: {np.mean(displacement[:, -1]):.2f} pixels")
    print(f"  Mean velocity: {np.mean(velocity):.2f} pixels/frame")
    
    # Save trajectories
    output_file_traj = output_dir / "trajectories.npy"
    output_file_init = output_dir / "initial_positions.npy"
    
    np.save(output_file_traj, trajectories)
    np.save(output_file_init, initial_positions)
    
    print(f"\n✓ Saved trajectories to {output_file_traj}")
    print(f"✓ Saved initial positions to {output_file_init}")
    print(f"\nTrajectory shape: (N={trajectories.shape[0]}, T={trajectories.shape[1]}, XY=2)")
    print(f"To load: trajectories = np.load('{output_file_traj}')")
    
    # Visualize trajectories
    if args.visualize:
        print("\nVisualizing trajectories...")
        H, W = flows_np.shape[1:3]
        visualize_trajectories(trajectories, initial_positions, H, W, output_dir, args.max_trajectories)
        visualize_trajectory_statistics(trajectories, output_dir)


if __name__ == "__main__":
    main()
