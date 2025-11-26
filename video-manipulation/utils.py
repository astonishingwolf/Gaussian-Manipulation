"""Utility functions for printing, visualization, and helper functions."""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from scipy.interpolate import griddata
import numpy as np
from scipy.ndimage import gaussian_filter
def save_displacements(displacements, output_dir):
    """Save displacements to a file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'displacements.npy', displacements)
    
def print_fft_summary(fft_data):
    """Print summary of FFT results."""
    summary = fft_data['summary']
    frequencies = fft_data['frequencies']
    power_x = fft_data['power_x']
    power_y = fft_data['power_y']
    
    print("\n" + "=" * 60)
    print("FFT Analysis Results")
    print("=" * 60)
    
    print("\n📊 Data Shapes:")
    print(f"  FFT X (complex):     {fft_data['fft_x'].shape}")
    print(f"  FFT Y (complex):     {fft_data['fft_y'].shape}")
    print(f"  Power X:             {power_x.shape}")
    print(f"  Power Y:             {power_y.shape}")
    print(f"  Frequencies:         {frequencies.shape}")
    
    print("\n📈 Summary:")
    print(f"  FPS:                 {summary['fps']}")
    print(f"  Number of points:    {summary['n_trajectories']}")
    print(f"  Number of frames:    {summary['n_frames']}")
    
    print("\n🎯 Top Dominant Frequencies:")
    for i, (freq, power) in enumerate(zip(summary['top_frequencies'], summary['top_powers']), 1):
        period = 1.0 / freq if freq > 0 else float('inf')
        print(f"  {i}. Frequency: {freq:.4f} Hz")
        print(f"     Period:    {period:.4f} seconds")
        print(f"     Power:     {power:.2e}")
    
    print("\n📉 Frequency Range:")
    print(f"  Min frequency: {frequencies.min():.4f} Hz")
    print(f"  Max frequency: {frequencies.max():.4f} Hz")
    
    print("\n💪 Power Statistics:")
    total_power = power_x + power_y
    print(f"  Mean power:    {np.mean(total_power):.2e}")
    print(f"  Max power:     {np.max(total_power):.2e}")
    print(f"  Total power:   {np.sum(total_power):.2e}")
    
    print("\n" + "=" * 60)


def print_selected_modes_info(selected_modes):
    """Print information about loaded selected modes."""
    if selected_modes is None:
        return
    
    print(f"\n✓ Loaded pre-selected modes")
    print(f"  Format: {list(selected_modes.keys())}")
    if 'modes' in selected_modes:
        print(f"  Number of modes: {len(selected_modes['modes'])}")
        print(f"  Frequency range: {selected_modes['modes'].min():.4f} - {selected_modes['modes'].max():.4f} Hz")
        if 'sum_powers' in selected_modes:
            print(f"  Total power range: {selected_modes['sum_powers'].min():.2e} - {selected_modes['sum_powers'].max():.2e}")


def print_frequency_selection(selected, selected_modes):
    """Print selected frequencies information."""
    print("\n🎯 Using pre-selected frequency modes")
    mode_indices = selected['indices']
    print(f"  Selected {len(mode_indices)} modes with indices: {mode_indices}")
    print(f"  Corresponding frequencies: {selected_modes['modes']}")
    
    print(f"\n🎯 Selected Frequencies:")
    for i, (freq, power) in enumerate(zip(selected['frequencies'], selected['total_power']), 1):
        period = 1.0 / freq if freq > 0 else float('inf')
        print(f"  {i:2d}. {freq:8.4f} Hz  |  Period: {period:8.4f} sec  |  Power: {power:.2e}")
    
    if selected_modes is not None and 'sum_powers' in selected_modes:
        print(f"\n📊 Pre-selected Mode Information:")
        for i, (freq, x_pow, y_pow, sum_pow) in enumerate(zip(
            selected_modes['modes'], 
            selected_modes['x_powers'],
            selected_modes['y_powers'],
            selected_modes['sum_powers']
        ), 1):
            print(f"  {i:2d}. {freq:8.4f} Hz  |  X-power: {x_pow:.2e}  |  Y-power: {y_pow:.2e}  |  Sum: {sum_pow:.2e}")


def print_fft_matrix_info(fft_matrix):
    """Print FFT matrix information."""
    N, K = fft_matrix.shape[:2]
    print(f"\n{'='*60}")
    print(f"Built FFT Matrix")
    print(f"{'='*60}")
    print(f"  Shape: {fft_matrix.shape} (N={N}, K={K}, 2)")
    print(f"  Dtype: {fft_matrix.dtype}")
    print(f"  Memory: {fft_matrix.nbytes / 1024 / 1024:.2f} MB")
    
    print(f"\n📊 Matrix Statistics:")
    print(f"  Real part - mean: {fft_matrix.real.mean():.2e}, std: {fft_matrix.real.std():.2e}")
    print(f"  Imag part - mean: {fft_matrix.imag.mean():.2e}, std: {fft_matrix.imag.std():.2e}")
    print(f"  Magnitude - mean: {np.abs(fft_matrix).mean():.2e}, max: {np.abs(fft_matrix).max():.2e}")


def print_transition_matrices_info(transition_matrices, frequencies, dt, damping_factor):
    """Print transition matrices information."""
    K = len(frequencies)
    omegas = 2.0 * np.pi * frequencies
    
    print(f"\n{'='*60}")
    print(f"Created Transition Matrices")
    print(f"{'='*60}")
    print(f"  Shape: {transition_matrices.shape} (K={K}, 2, 2)")
    print(f"  Dtype: {transition_matrices.dtype}")
    print(f"  dt: {dt:.6f} seconds")
    print(f"  Damping factor: {damping_factor:.6f}")
    
    print(f"\n📊 Sample Transition Matrices (first 3 frequencies):")
    for i in range(min(3, K)):
        print(f"\n  Frequency {i+1}: {frequencies[i]:.4f} Hz (ω = {omegas[i]:.4f} rad/s)")
        print(f"    [[{transition_matrices[i, 0, 0]:8.5f}, {transition_matrices[i, 0, 1]:8.5f}],")
        print(f"     [{transition_matrices[i, 1, 0]:8.5f}, {transition_matrices[i, 1, 1]:8.5f}]]")


def print_state_info(state_y):
    """Print state vector information."""
    K = state_y.shape[0]
    print(f"\n{'='*60}")
    print(f"Initialized State Vector")
    print(f"{'='*60}")
    print(f"  Shape: {state_y.shape} (K={K}, 2)")
    print(f"  State components: [position, velocity]")


def print_initial_positions_info(initial_positions):
    """Print initial positions information."""
    if initial_positions is None:
        print(f"\n⚠ No initial positions available")
        return
    
    print(f"\n✓ Loaded initial positions from trajectories")
    print(f"  Shape: {initial_positions.shape} (N={initial_positions.shape[0]}, 2)")
    print(f"  X range: [{initial_positions[:, 0].min():.2f}, {initial_positions[:, 0].max():.2f}]")
    print(f"  Y range: [{initial_positions[:, 1].min():.2f}, {initial_positions[:, 1].max():.2f}]")


def print_brightness_filter_info(valid_indices, initial_positions, pixel_threshold):
    """Print brightness filtering information."""
    print(f"\nFiltered trajectories by brightness:")
    print(f"  Pixel threshold: {pixel_threshold:.2f}")
    print(f"  Valid trajectories: {len(valid_indices)} / {len(initial_positions)}")


def visualize_results(displacements, initial_positions, sample_indices, 
                     manipulation_params, frame, output_dir):
    """Generate visualization plots for simulation results."""
    output_dir = Path(output_dir)
    
    if initial_positions is not None:
        absolute_positions = displacements + initial_positions[None, :, :]
    else:
        absolute_positions = displacements
    
    video_path = output_dir / "displacements_overlay.mp4"
    T = displacements.shape[0]
    H, W = frame.shape[:2]
    fps = 20

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    for t in range(T):
        fig = plt.figure(figsize=(12, 8))
        plt.imshow(frame)
        plt.axis("off")

        for i in sample_indices:
            dx = absolute_positions[t, i, 1] - initial_positions[i, 1]
            dy = absolute_positions[t, i, 0] - initial_positions[i, 0]
            plt.arrow(
                initial_positions[i, 1], initial_positions[i, 0],
                dx, dy,
                color="red", head_width=5, head_length=5, alpha=0.6, length_includes_head=True
            )

        if manipulation_params is not None:
            p = manipulation_params["p"]
            d = manipulation_params["d"]
            plt.arrow(
                initial_positions[p, 1], initial_positions[p, 0],
                d[1] * 50, d[0] * 50,
                color="blue", head_width=20, head_length=20, linewidth=3, length_includes_head=True
            )

        plt.title(f"Displacements Overlay - t={t}")

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 3)
        plt.close(fig)

        img_resized = cv2.resize(img, (W, H))
        writer.write(cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"\n✓ Saved displacement overlay video to {video_path}")

    plt.figure(figsize=(12, 8))
    plt.imshow(frame)

    for i in sample_indices:
        plt.arrow(initial_positions[i, 1], initial_positions[i, 0],
                 absolute_positions[-1, i, 1] - initial_positions[i, 1],
                 absolute_positions[-1, i, 0] - initial_positions[i, 0],
                 color='red', head_width=5, head_length=5, alpha=0.6)
    
    if manipulation_params is not None:
        p = manipulation_params['p']
        d = manipulation_params['d']
        plt.arrow(initial_positions[p, 1], initial_positions[p, 0],
                 d[1]*50, d[0]*50,
                 color='blue', head_width=20, head_length=20, linewidth=3)
    
    plt.title('Displacements on First Frame (Red: trajectories, Blue: manipulation direction)')
    plt.savefig(output_dir / 'displacements_on_frame.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    plt.xlim(0, frame.shape[1])
    plt.ylim(frame.shape[0], 0)
    
    for i in sample_indices[:20]:
        plt.plot(absolute_positions[:, i, 1], absolute_positions[:, i, 0], 
                alpha=0.7, linewidth=1)
    
    plt.title('Trajectory Paths over Time (Sampled Points)')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'xy_displacements_sampled.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved visualizations to {output_dir}/")


def print_save_results(output_dir, num_modes, initial_positions):
    """Print save results information."""
    print(f"\n✓ Saved simulation results to {output_dir}/")
    print(f"  - selected_frequencies.npy  [{num_modes} modes]")
    print(f"  - selected_fft_matrix.npy  [N×{num_modes}×2 matrix]")
    print(f"  - state_y_init.npy  [{num_modes}×2 initial state]")
    print(f"  - transition_matrices.npy  [{num_modes}×2×2 matrices]")
    print(f"  - simulation_params.npy")
    if initial_positions is not None:
        print(f"  - initial_positions.npy  [N×2 positions]")


def interpolate_sparse_displacements(displacements, initial_positions, frame_shape, sigma=3.0):

    """
    Interpolate sparse per-trajectory displacements onto a dense (H, W, 2) grid 
    using a fast Gaussian Splatting technique.
    """
    
    if displacements.ndim != 3 or displacements.shape[-1] != 2:
        raise ValueError("displacements must have shape (T, N, 2)")
    if initial_positions is None or len(initial_positions) != displacements.shape[1]:
        raise ValueError("initial_positions must be provided with shape (N, 2) matching displacements")
    
    T, N, _ = displacements.shape
    H, W = frame_shape
    
    coords_y = np.clip(np.round(initial_positions[:, 0]).astype(int), 0, H - 1)
    coords_x = np.clip(np.round(initial_positions[:, 1]).astype(int), 0, W - 1)
    dense = np.zeros((T, H, W, 2), dtype=np.float32)
    
    for t in range(T):
        # --- A. Splatting ---
        # Create sparse map for displacements (sum) and a map for counts (weight)
        disp_sum_map = np.zeros((H, W, 2), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)
        
        # Vectorized addition/splatting
        np.add.at(disp_sum_map, (coords_y, coords_x), displacements[t])
        np.add.at(count_map, (coords_y, coords_x), 1)

        # --- B. Gaussian Smoothing (Interpolation) ---
        # Apply highly optimized Gaussian filter to both sum and count maps        
        blurred_disp_x = gaussian_filter(disp_sum_map[..., 0], sigma=sigma)
        blurred_disp_y = gaussian_filter(disp_sum_map[..., 1], sigma=sigma)        
        blurred_count = gaussian_filter(count_map, sigma=sigma)
        
        # --- C. Normalization ---
        # Normalize the blurred sums by the blurred weights
        # (Equivalent to a weighted average interpolation)
        
        # Avoid division by zero
        blurred_count[blurred_count == 0] = 1e-6         
        dense_frame = np.stack((blurred_disp_x / blurred_count, blurred_disp_y / blurred_count), axis=-1)
        
        dense[t] = dense_frame
        
    return dense


def interpolate_sparse_displacements_legacy(displacements, initial_positions, frame_shape, method='linear'):
    """
    Interpolate sparse per-trajectory displacements onto a dense (H, W, 2) grid for each frame.
    """
    if displacements.ndim != 3 or displacements.shape[-1] != 2:
        raise ValueError("displacements must have shape (T, N, 2)")
    if initial_positions is None or len(initial_positions) != displacements.shape[1]:
        raise ValueError("initial_positions must be provided with shape (N, 2) matching displacements")
    
    H, W = frame_shape
    points = np.column_stack((initial_positions[:, 1], initial_positions[:, 0]))  # (N, 2) as (x, y)
    grid_x, grid_y = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32)
    )
    
    dense = np.zeros((displacements.shape[0], H, W, 2), dtype=np.float32)
    
    for t in range(displacements.shape[0]):
        values = displacements[t]
        dense_frame = griddata(points, values, (grid_x, grid_y), method=method)
        
        if dense_frame is None or np.isnan(dense_frame).any():
            dense_nearest = griddata(points, values, (grid_x, grid_y), method='nearest')
            if dense_frame is None:
                dense_frame = dense_nearest
            else:
                nan_mask = np.isnan(dense_frame)
                dense_frame[nan_mask] = dense_nearest[nan_mask]
        
        dense[t] = dense_frame.astype(np.float32)
    
    return dense


def render_displacements(displacements, base_frame, output_dir, fps=20, initial_positions=None):
    """Render a video by remapping a base frame with dense displacements using cv2.remap."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(base_frame, (str, Path)):
        frame_bgr = cv2.imread(str(base_frame))
        if frame_bgr is None:
            raise FileNotFoundError(f"Could not read base frame from {base_frame}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = np.array(base_frame)
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("base_frame must be an RGB image with shape (H, W, 3)")

    H, W = frame_rgb.shape[:2]

    if displacements.ndim != 4 or displacements.shape[-1] != 2:
        raise ValueError("displacements must have shape (T, H, W, 2)")
    if displacements.shape[1] != H or displacements.shape[2] != W:
        raise ValueError(
            f"Displacement dimensions {displacements.shape[1:3]} do not match base frame {(H, W)}"
        )

    video_path = output_dir / "displacements_render.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    grid_x, grid_y = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32)
    )

    for flow in displacements:
        flow = flow.astype(np.float32)
        map_x = grid_x + flow[:, :, 0]
        map_y = grid_y + flow[:, :, 1]

        warped_rgb = cv2.remap(
            frame_rgb,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101
        )
        warped_bgr = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR)
        writer.write(warped_bgr)

    writer.release()
    print(f"✓ Rendered displacement video saved to {video_path}")
