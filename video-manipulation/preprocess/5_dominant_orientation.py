"""
Calculate dominant orientation (rotation angle) for selected frequency modes.

The improvement calculates a single, unified rotation angle for modes that are
present in both the X and Y selections (same frequency), by summing their
complex components. Modes present in only X or only Y are rotated using their
single component.

Usage:
    python video-manipulation/preprocess/5_dominant_orientation.py \
        --trajectory_dir data/optical_flows/00376/trajectories \
        --mode_dir data/optical_flows/00376/trajectories/modes \
        --output data/optical_flows/00376/trajectories/modes
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from scipy.interpolate import griddata


def find_frequency_indices(frequencies, mode_frequencies, tol=1e-6):

    """
    Find the indices in the frequency array that correspond to the selected mode frequencies.
    """
    indices = []
    for freq in mode_frequencies:
        # Find the closest frequency index
        idx = np.argmin(np.abs(frequencies - freq))
        if np.abs(frequencies[idx] - freq) > tol:
            print(f"Warning: Frequency {freq:.6f} not found exactly, using closest: {frequencies[idx]:.6f}")
        indices.append(idx)
    return np.array(indices)


def dominant_orientation(fft_x, fft_y, frequencies, selected_modes, output_dir):

    """
    Calculates a unified dominant orientation (rotation angle) for each selected mode.
    
    The angle is calculated as -np.angle(total_complex_sum) to align the dominant phase 
    to the positive real axis (phase 0).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find frequency indices for x and y modes
    x_indices = find_frequency_indices(frequencies, selected_modes['x_modes'])
    y_indices = find_frequency_indices(frequencies, selected_modes['y_modes'])
    
    # Find all unique frequency indices that are part of the selected modes
    all_selected_indices = np.unique(np.concatenate([x_indices, y_indices]))
    
    # Store the unified rotation angle for each unique frequency index
    unified_rotation_map = {} # {freq_idx: angle}
    
    x_set = set(x_indices)
    y_set = set(y_indices)
    
    print("Calculating unified rotation angles...")
    for freq_idx in all_selected_indices:
        
        mode_is_x = freq_idx in x_set
        mode_is_y = freq_idx in y_set
        
        total_complex_sum = 0
        
        # 1. Sum the complex mode shape components for rotation calculation
        if mode_is_x:
            total_complex_sum += np.sum(fft_x[:, freq_idx])
        if mode_is_y:
            total_complex_sum += np.sum(fft_y[:, freq_idx])

        # 2. Calculate rotation angle
        if np.abs(total_complex_sum) < 1e-9:
             angle = 0.0
        else:
             # Angle calculation: Align the phase of the total complex sum to the real axis (0)
             angle = -np.angle(total_complex_sum)
        
        unified_rotation_map[freq_idx] = angle
        
    # Prepare data structure for saving
    rotation_data = {
        'unified_map': unified_rotation_map,
        'x_mode_indices': x_indices,
        'y_mode_indices': y_indices,
        'x_modes': selected_modes['x_modes'],
        'y_modes': selected_modes['y_modes']
    }
    
    np.save(output_dir / "rotation_data_unified.npy", rotation_data, allow_pickle=True)
    print(f"✓ Unified rotation data saved to {output_dir / 'rotation_data_unified.npy'}")
    print(f"  Calculated {len(unified_rotation_map)} unique rotation angles.")
    
    return rotation_data, {'x_indices': x_indices, 'y_indices': y_indices}


def rotate_modes(fft_x, fft_y, mode_indices, rotation_data):

    """
    Applies the unified rotation angle to the respective X and Y mode shapes.
    """

    rotated_fft_x = fft_x.copy()
    rotated_fft_y = fft_y.copy()
    
    # Unified map is {freq_idx: angle}
    unified_rotation_map = rotation_data['unified_map']
    
    # Apply rotation to x modes using the unified map
    for mode_idx in mode_indices['x_indices']:
        angle = unified_rotation_map[mode_idx]
        rotation_factor = np.exp(1j * angle)
        rotated_fft_x[:, mode_idx] *= rotation_factor
        
    # Apply rotation to y modes using the unified map
    for mode_idx in mode_indices['y_indices']:
        angle = unified_rotation_map[mode_idx]
        rotation_factor = np.exp(1j * angle)
        rotated_fft_y[:, mode_idx] *= rotation_factor
        
    return rotated_fft_x, rotated_fft_y


def create_flow_map_from_fft(fft_x, fft_y, initial_positions, frequency_idx, grid_size=(256, 256)):

    """
    Create a flow map from FFT coefficients at a specific frequency.
    """
    fft_x_at_freq = fft_x[:, frequency_idx]
    fft_y_at_freq = fft_y[:, frequency_idx]
    
    # Separate real and imaginary parts
    real_x = np.real(fft_x_at_freq)
    imag_x = np.imag(fft_x_at_freq)
    real_y = np.real(fft_y_at_freq)
    imag_y = np.imag(fft_y_at_freq)
    
    # Get spatial coordinates (note: initial_positions are [y, x])
    y_coords = initial_positions[:, 0]
    x_coords = initial_positions[:, 1]
    
    # Create output grid and interpolate
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()
    
    # Add small padding
    padding = 0.02
    y_range = y_max - y_min
    x_range = x_max - x_min
    y_min -= y_range * padding
    y_max += y_range * padding
    x_min -= x_range * padding
    x_max += x_range * padding
    
    grid_y, grid_x = np.mgrid[y_min:y_max:grid_size[0]*1j, x_min:x_max:grid_size[1]*1j]
    
    flow_x_real = griddata((y_coords, x_coords), real_x, (grid_y, grid_x), method='cubic', fill_value=0)
    flow_y_real = griddata((y_coords, x_coords), real_y, (grid_y, grid_x), method='cubic', fill_value=0)
    flow_x_imag = griddata((y_coords, x_coords), imag_x, (grid_y, grid_x), method='cubic', fill_value=0)
    flow_y_imag = griddata((y_coords, x_coords), imag_y, (grid_y, grid_x), method='cubic', fill_value=0)
    
    # Combine into flow maps (H, W, 2)
    flow_map_real = np.stack([flow_x_real, flow_y_real], axis=-1)
    flow_map_imag = np.stack([flow_x_imag, flow_y_imag], axis=-1)
    
    return flow_map_real, flow_map_imag


def normalize_flow_map(flow_map):
    """
    Normalize flow map to [0, 1] range for visualization.
    """
    magnitude = np.sqrt(flow_map[:, :, 0]**2 + flow_map[:, :, 1]**2)
    max_mag = magnitude.max()
    if max_mag > 0:
        normalized_flow = flow_map / max_mag
    else:
        normalized_flow = flow_map
    
    return normalized_flow, magnitude


def create_composite_visualization(fft_x, fft_y, initial_positions, rotated_fft_x, rotated_fft_y, 
                                   mode_info_list, output_dir):
    """
    Creates a single composite figure (3 modes x 4 plots) to demonstrate phase correction.
    
    Layout: 
    Row 1: Mode 1 (Real Mag BEFORE | Imag Mag BEFORE | Real Mag AFTER | Imag Mag AFTER)
    Row 2: Mode 2 (Real Mag BEFORE | Imag Mag BEFORE | Real Mag AFTER | Imag Mag AFTER)
    Row 3: Mode 3 (Real Mag BEFORE | Imag Mag BEFORE | Real Mag AFTER | Imag Mag AFTER)
    """
    
    num_modes_to_plot = len(mode_info_list)
    if num_modes_to_plot == 0:
        print("No modes selected for composite visualization.")
        return

    # 3 rows (for 3 modes), 4 columns (for 4 magnitude plots per mode)
    fig, axes = plt.subplots(num_modes_to_plot, 4, figsize=(18, 5 * num_modes_to_plot)) 
    
    # Set main title for the figure
    fig.suptitle('Phase Correction Demonstration: Real vs. Imaginary Magnitude (Top 3 Modes)', fontsize=16, y=1.02)
    
    col_titles = ['Real Mag BEFORE', 'Imag Mag BEFORE', 'Real Mag AFTER (MAXIMIZED)', 'Imag Mag AFTER (MINIMIZED)']
    
    for row_idx, (freq, freq_idx, label_str) in enumerate(mode_info_list):
        
        # --- Data Generation for BEFORE ---
        flow_map_real_b, flow_map_imag_b = create_flow_map_from_fft(
            fft_x, fft_y, initial_positions, freq_idx
        )
        _, mag_real_b = normalize_flow_map(flow_map_real_b)
        _, mag_imag_b = normalize_flow_map(flow_map_imag_b)

        # --- Data Generation for AFTER ---
        flow_map_real_a, flow_map_imag_a = create_flow_map_from_fft(
            rotated_fft_x, rotated_fft_y, initial_positions, freq_idx
        )
        _, mag_real_a = normalize_flow_map(flow_map_real_a)
        _, mag_imag_a = normalize_flow_map(flow_map_imag_a)

        # --- Quantitative Check (AFTER) ---
        total_real_mag_a = np.sum(np.sqrt(flow_map_real_a[:, :, 0]**2 + flow_map_real_a[:, :, 1]**2))
        total_imag_mag_a = np.sum(np.sqrt(flow_map_imag_a[:, :, 0]**2 + flow_map_imag_a[:, :, 1]**2))

        if total_real_mag_a > 1e-9:
            minimization_ratio = total_imag_mag_a / total_real_mag_a
            ratio_text = f"Ratio: {minimization_ratio:.4f}"
        else:
            ratio_text = "Ratio: INF"
            
        # --- Plotting Row ---
        
        # Col 0: Real Mag BEFORE
        ax = axes[row_idx, 0]
        im = ax.imshow(mag_real_b, cmap='hot', origin='upper')
        if row_idx == 0: ax.set_title(col_titles[0], fontsize=10)
        ax.set_ylabel(f'Mode {row_idx+1}\n({label_str} {freq:.2f} Hz)', fontsize=10)
        ax.axis('off')

        # Col 1: Imag Mag BEFORE
        ax = axes[row_idx, 1]
        im = ax.imshow(mag_imag_b, cmap='cool', origin='upper')
        if row_idx == 0: ax.set_title(col_titles[1], fontsize=10)
        ax.axis('off')

        # Col 2: Real Mag AFTER (MAXIMIZED)
        ax = axes[row_idx, 2]
        im = ax.imshow(mag_real_a, cmap='hot', origin='upper')
        if row_idx == 0: ax.set_title(col_titles[2], fontsize=10)
        ax.set_xlabel(f"Real Mag: {np.max(mag_real_a):.2f}", fontsize=8)
        ax.axis('off')

        # Col 3: Imag Mag AFTER (MINIMIZED)
        ax = axes[row_idx, 3]
        im = ax.imshow(mag_imag_a, cmap='cool', origin='upper')
        if row_idx == 0: ax.set_title(col_titles[3], fontsize=10)
        ax.set_xlabel(f"{ratio_text}", fontsize=8)
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to make space for suptitle
    
    output_file = output_dir / f'composite_phase_correction_{num_modes_to_plot}_modes.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved composite visualization to {output_file}")


def visualize_frequency_flow_maps(fft_x, fft_y, initial_positions, frequencies, selected_modes, 
                                   mode_indices, rotated_fft_x=None, rotated_fft_y=None,
                                   output_dir=None, num_modes=6):
    """
    Prepares data for visualization, prioritizing the creation of a composite image
    for the top 3 modes to demonstrate phase correction.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_indices_map = {} # {freq_idx: freq}
    for freq, idx in zip(selected_modes['x_modes'], mode_indices['x_indices']):
        all_indices_map[idx] = freq
    for freq, idx in zip(selected_modes['y_modes'], mode_indices['y_indices']):
        all_indices_map[idx] = freq
    
    # Convert map to sorted list of (freq, freq_idx) tuples
    all_modes_sorted = sorted([(freq, idx) for idx, freq in all_indices_map.items()], key=lambda x: x[0])
    
    # --- Preparation for Composite Plot (Top 3 Modes) ---
    modes_for_composite = all_modes_sorted[:3]
    mode_info_list = []
    
    for freq, freq_idx in modes_for_composite:
        original_label_x = np.where(mode_indices['x_indices'] == freq_idx)[0]
        original_label_y = np.where(mode_indices['y_indices'] == freq_idx)[0]
        label_str = ""
        if original_label_x.size > 0: label_str += f"X:{original_label_x[0]+1}"
        if original_label_y.size > 0 and original_label_x.size > 0: label_str += "/"
        if original_label_y.size > 0: label_str += f"Y:{original_label_y[0]+1}"
        mode_info_list.append((freq, freq_idx, label_str))

    print("\nGenerating composite visualization (Top 3 unique modes)...")

    if rotated_fft_x is not None and rotated_fft_y is not None:
        create_composite_visualization(
            fft_x, fft_y, initial_positions, rotated_fft_x, rotated_fft_y,
            mode_info_list, output_dir
        )
    else:
        print("Skipping composite visualization: Rotated FFT data not provided.")


def main():

    parser = argparse.ArgumentParser(description="Calculate dominant orientation (rotation angle) for selected frequency modes")
    parser.add_argument("--trajectory_dir", type=str, required=True, help="Directory containing fft_x.npy, fft_y.npy, frequencies.npy, and initial_positions.npy")
    parser.add_argument("--mode_dir", type=str, help="Directory containing selected_modes.npy (default: trajectory_dir)")
    parser.add_argument("--output", type=str, help="Output directory for rotation angles and visualizations (default: mode_dir)")
    parser.add_argument("--num-visualize", type=int, default=6, help="Number of frequency modes to visualize (this is now only used for file saving, visualization is composite for top 3)")
    args = parser.parse_args()
    
    trajectory_dir = Path(args.trajectory_dir)
    mode_dir = Path(args.mode_dir) if args.mode_dir else trajectory_dir 
    output_dir = Path(args.output) if args.output else mode_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading FFT data from {trajectory_dir}...")
    try:
        fft_x = np.load(trajectory_dir / "fft_x.npy")
        fft_y = np.load(trajectory_dir / "fft_y.npy")
        frequencies = np.load(trajectory_dir / "frequencies.npy")
        selected_modes = np.load(mode_dir / "selected_modes.npy", allow_pickle=True).item()
        # Load initial positions for flow map visualization
        initial_positions = np.load(trajectory_dir / "initial_positions.npy")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print(f"Please ensure all required files exist in {trajectory_dir}")
        return
    
    print(f"FFT shapes: fft_x={fft_x.shape}, fft_y={fft_y.shape}, frequencies={frequencies.shape}")
    print(f"Initial positions shape: {initial_positions.shape}")
    print(f"Selected modes: {len(selected_modes['x_modes'])} x-modes, {len(selected_modes['y_modes'])} y-modes")
    
    # Calculate rotation angles
    print("\nCalculating unified rotation angles...")
    rotation_data, mode_indices = dominant_orientation(
        fft_x, fft_y, frequencies, selected_modes, output_dir
    )
    rotated_fft_x, rotated_fft_y = rotate_modes(fft_x, fft_y, mode_indices, rotation_data)
    np.save(output_dir / "rotated_fft_x.npy", rotated_fft_x)
    np.save(output_dir / "rotated_fft_y.npy", rotated_fft_y)
    print(f"✓ Rotated FFT data saved to {output_dir}")
    
    # Visualize frequency flow maps
    visualize_frequency_flow_maps(
        fft_x, fft_y, initial_positions, frequencies, selected_modes, mode_indices,
        rotated_fft_x=rotated_fft_x, rotated_fft_y=rotated_fft_y,
        output_dir=output_dir, num_modes=args.num_visualize
    )
    
    print(f"\n✓ Complete! All results saved to {output_dir}")


if __name__ == "__main__":
    main()