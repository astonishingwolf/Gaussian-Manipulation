"""
Modal Simulation for Video Manipulation

This script performs frequency-domain manipulation of trajectories using FFT analysis.
It loads pre-computed FFT results, selects dominant frequency modes, and simulates
controlled displacements using damped harmonic oscillator dynamics.

Usage:
    python video-manipulation/preprocess/5_simulation.py \
        --trajectory-dir data/optical_flows/ball/trajectories \
        --top-k 16 \
        --damping-factor 0.03
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


def load_fft_results(trajectories_dir):
    """
    Load all FFT analysis results from a directory.
    
    Args:
        trajectories_dir: Path to directory containing FFT results
    
    Returns:
        dict: Dictionary containing all FFT data
    """
    trajectories_dir = Path(trajectories_dir)
    
    # Check if files exist
    required_files = ['fft_x.npy', 'fft_y.npy', 'power_x.npy', 'power_y.npy', 
                     'frequencies.npy', 'fft_summary.npy']
    
    for file in required_files:
        if not (trajectories_dir / file).exists():
            raise FileNotFoundError(f"Required file not found: {trajectories_dir / file}")
    
    print(f"Loading FFT results from {trajectories_dir}...")
    
    # Load all files
    fft_data = {
        'fft_x': np.load(trajectories_dir / 'fft_x.npy'),
        'fft_y': np.load(trajectories_dir / 'fft_y.npy'),
        'power_x': np.load(trajectories_dir / 'power_x.npy'),
        'power_y': np.load(trajectories_dir / 'power_y.npy'),
        'frequencies': np.load(trajectories_dir / 'frequencies.npy'),
        'summary': np.load(trajectories_dir / 'fft_summary.npy', allow_pickle=True).item()
    }
    
    # Load trajectories if available (for initial positions)
    trajectories_file = trajectories_dir / 'trajectories.npy'
    if trajectories_file.exists():
        fft_data['trajectories'] = np.load(trajectories_file)
        print(f"✓ Loaded trajectories: {fft_data['trajectories'].shape}")
    else:
        print(f"⚠ No trajectories.npy found, initial positions will not be available")
        fft_data['trajectories'] = None
    
    print("✓ Successfully loaded all FFT results")
    
    return fft_data


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


def load_selected_modes(trajectories_dir):
    """
    Load pre-selected modes from selected_modes.npy file.
    
    Args:
        trajectories_dir: Path to directory containing selected_modes.npy
    
    Returns:
        dict: Dictionary containing selected modes info or None if not found
    """
    trajectories_dir = Path(trajectories_dir)
    modes_file = trajectories_dir / 'selected_modes.npy'
    
    if modes_file.exists():
        selected_modes = np.load(modes_file, allow_pickle=True).item()
        print(f"\n✓ Loaded pre-selected modes from {modes_file}")
        print(f"  Number of modes: {len(selected_modes['modes'])}")
        return selected_modes
    else:
        print(f"\n⚠ No pre-selected modes found at {modes_file}")
        return None


def select_top_k_frequencies(fft_x, fft_y, power_x, power_y, frequencies, top_k=16, selected_modes=None):
    """
    Select top K dominant frequencies and their corresponding FFT coefficients.
    Can use pre-selected modes from 4_evaluate_candidate_modes.py if available.
    
    Args:
        fft_x: Complex FFT array for x-coordinates (N, T)
        fft_y: Complex FFT array for y-coordinates (N, T)
        power_x: Power spectrum for x (N, T)
        power_y: Power spectrum for y (N, T)
        frequencies: Frequency bins (T,)
        top_k: Number of top frequencies to select (ignored if selected_modes provided)
        selected_modes: Pre-selected modes dict with 'indices' key (optional)
    
    Returns:
        dict: Dictionary containing selected frequencies and coefficients
    """
    if selected_modes is not None and 'indices' in selected_modes:
        # Use pre-selected mode indices
        print("\n🎯 Using pre-selected frequency modes from 4_evaluate_candidate_modes.py")
        top_k_indices = selected_modes['indices']
        top_k = len(top_k_indices)
    else:
        # Calculate total power across all trajectories
        print("\n🎯 Selecting frequencies based on power spectrum")
        total_power = np.mean(power_x + power_y, axis=0)
        
        # Only consider positive frequencies
        pos_mask = frequencies > 0
        pos_freqs = frequencies[pos_mask]
        pos_power = total_power[pos_mask]
        
        # Get indices of top K frequencies
        top_k_indices_pos = np.argsort(pos_power)[-top_k:][::-1]
        
        # Get actual indices in the full frequency array
        pos_indices = np.where(pos_mask)[0]
        top_k_indices = pos_indices[top_k_indices_pos]
    
    # Extract top K frequencies and their data
    selected_frequencies = frequencies[top_k_indices]
    selected_fft_x = fft_x[:, top_k_indices]
    selected_fft_y = fft_y[:, top_k_indices]
    selected_power_x = power_x[:, top_k_indices]
    selected_power_y = power_y[:, top_k_indices]
    
    # Calculate total power for selected frequencies
    total_power_all = np.mean(power_x + power_y, axis=0)
    total_power_selected = total_power_all[top_k_indices]
    
    result = {
        'frequencies': selected_frequencies,
        'fft_x': selected_fft_x,
        'fft_y': selected_fft_y,
        'power_x': selected_power_x,
        'power_y': selected_power_y,
        'indices': top_k_indices,
        'total_power': total_power_selected
    }
    
    return result


def build_fft_matrix(selected):
    """
    Build N×K×2 matrix from selected FFT results.
    
    Args:
        selected: Dictionary containing 'fft_x' and 'fft_y' arrays
    
    Returns:
        np.ndarray: Complex array of shape (N, K, 2) where:
                   - N is the number of trajectories
                   - K is the number of selected frequencies
                   - 2 represents [x, y] coordinates
    """
    fft_x = selected['fft_x']  # Shape: (N, K)
    fft_y = selected['fft_y']  # Shape: (N, K)
    
    N, K = fft_x.shape
    
    # Stack x and y into a single matrix
    fft_matrix = np.stack([fft_x, fft_y], axis=-1)  # Shape: (N, K, 2)
    
    print(f"\n{'='*60}")
    print(f"Built FFT Matrix")
    print(f"{'='*60}")
    print(f"  Shape: {fft_matrix.shape} (N={N}, K={K}, 2)")
    print(f"  Dtype: {fft_matrix.dtype}")
    print(f"  Memory: {fft_matrix.nbytes / 1024 / 1024:.2f} MB")
    
    # Show statistics
    print(f"\n📊 Matrix Statistics:")
    print(f"  Real part - mean: {fft_matrix.real.mean():.2e}, std: {fft_matrix.real.std():.2e}")
    print(f"  Imag part - mean: {fft_matrix.imag.mean():.2e}, std: {fft_matrix.imag.std():.2e}")
    print(f"  Magnitude - mean: {np.abs(fft_matrix).mean():.2e}, max: {np.abs(fft_matrix).max():.2e}")
    
    return fft_matrix


def create_transition_matrix(omega_i, dt, damping_factor):
    """
    Create a 2×2 transition matrix for a single frequency component.
    
    The state vector is [position, velocity].
    The transition matrix for a damped harmonic oscillator is:
    [[1, dt],
     [-omega_i^2 * dt, 1 - 2 * damping_factor * omega_i * dt]]
    
    Args:
        omega_i: Angular frequency (omega_i = 2 * pi * f_i)
        dt: Time step
        damping_factor: Damping coefficient (zeta)
    
    Returns:
        np.ndarray: 2×2 transition matrix
    """
    A = np.array([
        [1.0, dt],
        [-omega_i**2 * dt, 1.0 - 2.0 * damping_factor * omega_i * dt]
    ], dtype=np.float64)
    
    return A


def create_transition_matrices(frequencies, dt, damping_factor):
    """
    Create transition matrices for all K frequency components.
    
    Args:
        frequencies: Array of frequencies in Hz, shape (K,)
        dt: Time step in seconds
        damping_factor: Damping coefficient (zeta)
    
    Returns:
        np.ndarray: Transition matrices of shape (K, 2, 2)
    """
    K = len(frequencies)
    
    # Convert frequencies to angular frequencies
    omegas = 2.0 * np.pi * frequencies  # omega_i = 2*pi*f_i
    
    # Create transition matrices for each frequency
    transition_matrices = np.zeros((K, 2, 2), dtype=np.float64)
    
    for i in range(K):
        transition_matrices[i] = create_transition_matrix(omegas[i], dt, damping_factor)
    
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
    
    return transition_matrices


def initialize_state(K):
    """
    Initialize state vector for frequency components.
    
    Args:
        K: Number of frequency modes
    
    Returns:
        np.ndarray: State vector of shape (K, 2) initialized to zeros
    """
    state_y = np.zeros((K, 2), dtype=np.float64)
    
    print(f"\n{'='*60}")
    print(f"Initialized State Vector")
    print(f"{'='*60}")
    print(f"  Shape: {state_y.shape} (K={K}, 2)")
    print(f"  State components: [position, velocity]")
    
    return state_y


def extract_initial_positions(trajectories):
    """
    Extract initial positions from trajectory data.
    
    Args:
        trajectories: Trajectory array of shape (N, T, 2) or None
    
    Returns:
        np.ndarray or None: Initial positions of shape (N, 2)
    """
    if trajectories is None:
        print(f"\n⚠ No initial positions available")
        return None
    
    initial_positions = trajectories[:, 0, :]
    print(f"\n✓ Loaded initial positions from trajectories")
    print(f"  Shape: {initial_positions.shape} (N={initial_positions.shape[0]}, 2)")
    print(f"  X range: [{initial_positions[:, 0].min():.2f}, {initial_positions[:, 0].max():.2f}]")
    print(f"  Y range: [{initial_positions[:, 1].min():.2f}, {initial_positions[:, 1].max():.2f}]")
    
    return initial_positions


def filter_bright_trajectories(initial_positions, frame, pixel_threshold):
    """
    Filter trajectories based on pixel brightness.
    
    Args:
        initial_positions: Array of positions shape (N, 2)
        frame: Video frame (H, W, C)
        pixel_threshold: Minimum pixel value threshold
    
    Returns:
        np.ndarray: Indices of valid trajectories
    """
    pixel_values = []
    for pos in initial_positions:
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < frame.shape[0] and 0 <= y < frame.shape[1]:
            pixel_values.append(np.mean(frame[x, y, :]))
        else:
            pixel_values.append(0)
    
    pixel_values = np.array(pixel_values)
    valid_indices = np.where(pixel_values > pixel_threshold)[0]
    
    print(f"\nFiltered trajectories by brightness:")
    print(f"  Pixel threshold: {pixel_threshold:.2f}")
    print(f"  Valid trajectories: {len(valid_indices)} / {len(initial_positions)}")
    
    return valid_indices


def apply_manipulation(state_y, fft_matrix, frequencies, d, p, alpha):
    """
    Apply phase manipulation to the state vector at a specific position.
    
    Args:
        state_y: Current state vector (K, 2)
        fft_matrix: FFT matrix (N, K, 2)
        frequencies: Selected frequencies (K,)
        d: Direction vector (2,)
        p: Target position index
        alpha: Manipulation strength
    
    Returns:
        np.ndarray: Updated state vector
    """
    # Convert state to complex representation
    omegas = 2.0 * np.pi * frequencies
    q = state_y[:, 0] - 1j * state_y[:, 1] / (omegas + 1e-8)
    
    # Calculate manipulation strength and phase
    strength = np.abs(fft_matrix[p] @ d.reshape(2, 1)) * alpha
    angle = -np.angle(fft_matrix[p] @ d.reshape(2, 1)) + np.pi / 2
    
    # Apply phase shift
    phase_shift = np.exp(1j * angle)
    q = strength.flatten() * phase_shift.flatten()
    
    # Convert back to state representation
    state_y[:, 0] = np.real(q)
    state_y[:, 1] = -np.imag(q) * (omegas + 1e-8)
    
    return state_y


def simulate_dynamics(state_y, fft_matrix, transition_matrices, frequencies, 
                     timesteps, manipulation_params=None):
    """
    Simulate trajectory dynamics over time with optional manipulation.
    
    Args:
        state_y: Initial state vector (K, 2)
        fft_matrix: FFT matrix (N, K, 2)
        transition_matrices: Transition matrices (K, 2, 2)
        frequencies: Selected frequencies (K,)
        timesteps: Number of simulation steps
        manipulation_params: Dict with 'd', 'p', 'alpha' keys for manipulation
    
    Returns:
        np.ndarray: Displacement array of shape (timesteps, N, 2)
    """
    K = len(frequencies)
    N = fft_matrix.shape[0]
    omegas = 2.0 * np.pi * frequencies
    
    displacements = []
    
    for t in range(timesteps):
        # Convert state to complex representation
        q = state_y[:, 0] - 1j * state_y[:, 1] / (omegas + 1e-8)
        
        # Apply manipulation at first timestep
        if t == 0 and manipulation_params is not None:
            state_y = apply_manipulation(
                state_y, fft_matrix, frequencies,
                manipulation_params['d'],
                manipulation_params['p'],
                manipulation_params['alpha']
            )
            q = state_y[:, 0] - 1j * state_y[:, 1] / (omegas + 1e-8)
        
        # Compute displacement
        displacement = np.real(fft_matrix * q[None, :, None])
        displacement = displacement.sum(axis=1)
        displacements.append(displacement)
        
        # Update state using transition matrices
        for i in range(K):
            state_y[i] = transition_matrices[i] @ state_y[i]
    
    displacements = np.array(displacements)
    print(f"\n✓ Simulated displacements over {timesteps} timesteps")
    print(f"  Shape: {displacements.shape}")
    
    return displacements


def visualize_results(displacements, initial_positions, sample_indices, 
                     manipulation_params, frame, output_dir):
    """
    Generate visualization plots for simulation results.
    
    Args:
        displacements: Displacement array (T, N, 2)
        initial_positions: Initial positions (N, 2)
        sample_indices: Indices of sampled trajectories
        manipulation_params: Dict with manipulation info
        frame: Video first frame
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    
    # Add initial positions to displacements for absolute positions
    if initial_positions is not None:
        absolute_positions = displacements + initial_positions[None, :, :]
    else:
        absolute_positions = displacements
    
   
    # Create a video showing per-frame displacement overlays on the first frame
    video_path = output_dir / "displacements_overlay.mp4"
    T = displacements.shape[0]
    H, W = frame.shape[:2]
    fps = 20  # visualization fps

    # Prepare absolute positions
    if initial_positions is not None:
        abs_pos = displacements + initial_positions[None, :, :]
    else:
        abs_pos = displacements

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

        # Draw displacement arrows for sampled indices
        for i in sample_indices:
            dx = abs_pos[t, i, 1] - initial_positions[i, 1]
            dy = abs_pos[t, i, 0] - initial_positions[i, 0]
            plt.arrow(
                initial_positions[i, 1], initial_positions[i, 0],
                dx, dy,
                color="red", head_width=5, head_length=5, alpha=0.6, length_includes_head=True
            )

        # Draw manipulation direction
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

        # Resize to match video size
        img_resized = cv2.resize(img, (W, H))
        writer.write(cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"\n✓ Saved displacement overlay video to {video_path}")

    # Plot displacements on first frame
    plt.figure(figsize=(12, 8))
    plt.imshow(frame)

    for i in sample_indices:
        plt.arrow(initial_positions[i, 1], initial_positions[i, 0],
                 absolute_positions[-1, i, 1] - initial_positions[i, 1],
                 absolute_positions[-1, i, 0] - initial_positions[i, 0],
                 color='red', head_width=5, head_length=5, alpha=0.6)
    
    
    # Plot manipulation direction
    if manipulation_params is not None:
        p = manipulation_params['p']
        d = manipulation_params['d']
        plt.arrow(initial_positions[p, 1], initial_positions[p, 0],
                 d[1]*50, d[0]*50,
                 color='blue', head_width=20, head_length=20, linewidth=3)
    
    plt.title('Displacements on First Frame (Red: trajectories, Blue: manipulation direction)')
    plt.savefig(output_dir / 'displacements_on_frame.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot trajectory paths
    plt.figure(figsize=(12, 8))
    plt.xlim(0, frame.shape[1])
    plt.ylim(frame.shape[0], 0)
    
    for i in sample_indices[:20]:  # Limit to 20 for clarity
        plt.plot(absolute_positions[:, i, 1], absolute_positions[:, i, 0], 
                alpha=0.7, linewidth=1)
    
    plt.title('Trajectory Paths over Time (Sampled Points)')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'xy_displacements_sampled.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved visualizations to {output_dir}/")


def save_simulation_results(output_dir, selected, fft_matrix, state_y, 
                           transition_matrices, initial_positions, sim_params, top_k):
    """
    Save all simulation outputs to disk.
    
    Args:
        output_dir: Output directory path
        selected: Selected frequency data dict
        fft_matrix: FFT matrix
        state_y: Initial state vector
        transition_matrices: Transition matrices
        initial_positions: Initial positions array
        sim_params: Simulation parameters dict
        top_k: Number of top modes
    """
    output_dir = Path(output_dir)
    
    np.save(output_dir / f"selected_top{top_k}_frequencies.npy", selected['frequencies'])
    np.save(output_dir / f"selected_top{top_k}_fft_x.npy", selected['fft_x'])
    np.save(output_dir / f"selected_top{top_k}_fft_y.npy", selected['fft_y'])
    np.save(output_dir / f"selected_top{top_k}_fft_matrix.npy", fft_matrix)
    np.save(output_dir / f"state_y_init.npy", state_y)
    np.save(output_dir / f"transition_matrices.npy", transition_matrices)
    np.save(output_dir / "simulation_params.npy", sim_params)
    
    if initial_positions is not None:
        np.save(output_dir / "initial_positions.npy", initial_positions)
    
    print(f"\n✓ Saved simulation results to {output_dir}/")
    print(f"  - selected_top{top_k}_frequencies.npy")
    print(f"  - selected_top{top_k}_fft_matrix.npy  [N×K×2 matrix]")
    print(f"  - state_y_init.npy  [K×2 initial state]")
    print(f"  - transition_matrices.npy  [K×2×2 matrices]")
    print(f"  - simulation_params.npy")
    if initial_positions is not None:
        print(f"  - initial_positions.npy  [N×2 positions]")


def main():
    parser = argparse.ArgumentParser(
        description="Modal simulation for trajectory manipulation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trajectory-dir", required=True,
                       help="Path to trajectories directory containing FFT results")
    parser.add_argument("--video-path", default="data/00133.mp4",
                       help="Path to video file for visualization (default: data/00133.mp4)")
    parser.add_argument("--top-k", type=int, default=16,
                       help="Number of top frequencies to select (default: 16)")
    parser.add_argument("--dt", type=float, default=None,
                       help="Time step in seconds (default: 0.1/fps)")
    parser.add_argument("--damping-factor", type=float, default=0.03,
                       help="Damping coefficient zeta (default: 0.03)")
    parser.add_argument("--timesteps", type=int, default=300,
                       help="Number of simulation timesteps (default: 300)")
    parser.add_argument("--manipulation-strength", type=float, default=1e-4,
                       help="Manipulation strength alpha (default: 1e-4)")
    parser.add_argument("--n-samples", type=int, default=100,
                       help="Number of trajectories to visualize (default: 100)")
    
    args = parser.parse_args()
    
    # =========================================================================
    # 1. Load Data
    # =========================================================================
    fft_data = load_fft_results(args.trajectory_dir)
    print_fft_summary(fft_data)
    
    
    # =========================================================================
    # 2. Select Frequency Modes
    # =========================================================================
    selected_modes = load_selected_modes(args.trajectory_dir)
    
    print(f"\n{'='*60}")
    if selected_modes is not None:
        print("Using Pre-Selected Modes")
    else:
        print(f"Selecting Top {args.top_k} Frequencies")
    print(f"{'='*60}")
    
    selected = select_top_k_frequencies(
        fft_data['fft_x'], fft_data['fft_y'], 
        fft_data['power_x'], fft_data['power_y'], 
        fft_data['frequencies'],
        top_k=args.top_k,
        selected_modes=selected_modes
    )
    
    print(f"\n🎯 Selected Frequencies:")
    for i, (freq, power) in enumerate(zip(selected['frequencies'], selected['total_power']), 1):
        period = 1.0 / freq if freq > 0 else float('inf')
        print(f"  {i:2d}. {freq:8.4f} Hz  |  Period: {period:8.4f} sec  |  Power: {power:.2e}")
    
    # =========================================================================
    # 3. Build FFT Matrix and Extract Initial Positions
    # =========================================================================
    fft_matrix = build_fft_matrix(selected)
    initial_positions = extract_initial_positions(fft_data['trajectories'])
    
    # =========================================================================
    # 4. Initialize State and Transition Matrices
    # =========================================================================
    K = len(selected['frequencies'])
    state_y = initialize_state(K)
    
    # Determine time step
    if args.dt is None:
        fps = fft_data['summary']['fps']
        dt = 0.1 / fps
        print(f"\n  Time step: {dt:.6f} seconds (0.1/FPS)")
    else:
        dt = args.dt
        print(f"\n  Time step: {dt:.6f} seconds (user-specified)")
    
    transition_matrices = create_transition_matrices(
        selected['frequencies'], dt, args.damping_factor
    )
    
    # =========================================================================
    # 5. Load Video Frame and Filter Trajectories
    # =========================================================================
    cap = cv2.VideoCapture(args.video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise FileNotFoundError(f"Could not read video: {args.video_path}")
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pixel_average = np.mean(frame)
    
    print(f"\nLoaded video frame from: {args.video_path}")
    print(f"  Frame shape: {frame.shape}")
    print(f"  Average pixel value: {pixel_average:.2f}")
    
    # Filter bright trajectories
    if initial_positions is not None:
        valid_indices = filter_bright_trajectories(initial_positions, frame, pixel_average)
        n_samples = min(args.n_samples, len(valid_indices))
        sample_indices = np.random.choice(valid_indices, size=n_samples, replace=False)
    else:
        sample_indices = np.random.choice(fft_matrix.shape[0], 
                                         size=min(args.n_samples, fft_matrix.shape[0]), 
                                         replace=False)
        valid_indices = sample_indices
    
    # =========================================================================
    # 6. Set Up Manipulation Parameters
    # =========================================================================
    d = np.random.rand(2)
    d = d / np.linalg.norm(d)
    p = np.random.choice(valid_indices)
    
    manipulation_params = {
        'd': d,
        'p': p,
        'alpha': args.manipulation_strength
    }
    
    print(f"\nManipulation parameters:")
    print(f"  Direction: [{d[0]:.4f}, {d[1]:.4f}]")
    print(f"  Position index: {p}")
    print(f"  Strength: {args.manipulation_strength:.2e}")
    
    # =========================================================================
    # 7. Run Simulation
    # =========================================================================
    displacements = simulate_dynamics(
        state_y, fft_matrix, transition_matrices, 
        selected['frequencies'], args.timesteps,
        manipulation_params=manipulation_params
    )
    
    # =========================================================================
    # 8. Visualize Results
    # =========================================================================
    visualize_results(
        displacements, initial_positions, sample_indices,
        manipulation_params, frame, args.trajectory_dir
    )
    
    # =========================================================================
    # 9. Save Results
    # =========================================================================
    sim_params = {
        'dt': dt,
        'damping_factor': args.damping_factor,
        'top_k': K,
        'fps': fft_data['summary']['fps'],
        'timesteps': args.timesteps,
        'manipulation_strength': args.manipulation_strength
    }
    
    save_simulation_results(
        args.trajectory_dir, selected, fft_matrix, state_y,
        transition_matrices, initial_positions, sim_params, K
    )
    
    print("\n" + "="*60)
    print("✓ Simulation completed successfully")
    print("="*60)
    
    return {
        'selected': selected,
        'fft_matrix': fft_matrix,
        'displacements': displacements,
        'manipulation_params': manipulation_params
    }



if __name__ == "__main__":
    main()
