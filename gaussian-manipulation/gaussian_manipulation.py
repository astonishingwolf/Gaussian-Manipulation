"""
Modal Simulation for 3D Gaussian Manipulation

Usage:
    python gaussian-manipulation/gaussian_manipulation.py \
        --trajectory-dir data/fft_results/ \
        --damping-factor 0.03
    
    Note: Requires selected_modes.npy from 2_evaluate_candidate_modes.py
    Requires FFT results from 1_analyze_trajectories_fft.py
"""

import argparse
import numpy as np
from pathlib import Path


def load_fft_results(trajectories_dir):
    """Load all FFT analysis results from a directory for 3D trajectories."""
    trajectories_dir = Path(trajectories_dir)    
    required_files = ['fft_x.npy', 'fft_y.npy', 'fft_z.npy', 
                     'power_x.npy', 'power_y.npy', 'power_z.npy', 
                     'frequencies.npy', 'fft_summary.npy']
    
    for file in required_files:
        if not (trajectories_dir / file).exists():
            raise FileNotFoundError(f"Required file not found: {trajectories_dir / file}")
    
    fft_x = np.load(trajectories_dir / 'fft_x.npy')
    fft_y = np.load(trajectories_dir / 'fft_y.npy')
    fft_z = np.load(trajectories_dir / 'fft_z.npy')
    
    fft_data = {
        'fft_x': fft_x,
        'fft_y': fft_y,
        'fft_z': fft_z,
        'power_x': np.load(trajectories_dir / 'power_x.npy'),
        'power_y': np.load(trajectories_dir / 'power_y.npy'),
        'power_z': np.load(trajectories_dir / 'power_z.npy'),
        'frequencies': np.load(trajectories_dir / 'frequencies.npy'),
        'summary': np.load(trajectories_dir / 'fft_summary.npy', allow_pickle=True).item()
    }
    
    trajectories_file = trajectories_dir / 'trajectories.npy'
    if trajectories_file.exists():
        fft_data['trajectories'] = np.load(trajectories_file)
    else:
        fft_data['trajectories'] = None
    
    return fft_data


def load_selected_modes(trajectories_dir):
    """Load pre-selected modes from selected_modes.npy file."""
    trajectories_dir = Path(trajectories_dir)
    modes_file = trajectories_dir / 'modes' / 'selected_modes.npy'
    
    # Try in modes subdirectory first, then in main directory
    if not modes_file.exists():
        modes_file = trajectories_dir / 'selected_modes.npy'
    
    if modes_file.exists():
        return np.load(modes_file, allow_pickle=True).item()
    return None


def select_frequencies_from_modes(fft_x, fft_y, fft_z, power_x, power_y, power_z, 
                                  frequencies, selected_modes):
    """Select frequencies and their corresponding FFT coefficients using pre-selected modes for 3D."""
    if selected_modes is None or 'indices' not in selected_modes:
        raise ValueError("selected_modes is required and must contain 'indices' key. Run 2_evaluate_candidate_modes.py first.")
    mode_indices = np.array(selected_modes['indices'])
    if np.any(mode_indices < 0) or np.any(mode_indices >= len(frequencies)):
        raise ValueError(f"Invalid frequency indices in selected_modes: {mode_indices}")
    
    selected_frequencies = frequencies[mode_indices]
    selected_fft_x = fft_x[:, mode_indices]
    selected_fft_y = fft_y[:, mode_indices]
    selected_fft_z = fft_z[:, mode_indices]
    selected_power_x = power_x[:, mode_indices]
    selected_power_y = power_y[:, mode_indices]
    selected_power_z = power_z[:, mode_indices]
    
    # Calculate total power for selected frequencies
    total_power_all = np.mean(power_x + power_y + power_z, axis=0)
    total_power_selected = total_power_all[mode_indices]
    
    result = {
        'frequencies': selected_frequencies,
        'fft_x': selected_fft_x,
        'fft_y': selected_fft_y,
        'fft_z': selected_fft_z,
        'power_x': selected_power_x,
        'power_y': selected_power_y,
        'power_z': selected_power_z,
        'indices': mode_indices,
        'total_power': total_power_selected
    }
    
    return result


def build_fft_matrix(selected):
    """Build N×K×3 matrix from selected FFT results for 3D."""
    return np.stack([selected['fft_x'], selected['fft_y'], selected['fft_z']], axis=-1)


def create_transition_matrices(frequencies, dt, damping_factor):
    """Create transition matrices for all K frequency components."""
    K = len(frequencies)
    omegas = 2.0 * np.pi * frequencies
    transition_matrices = np.zeros((K, 2, 2), dtype=np.float64)
    
    for i in range(K):
        omega_i = omegas[i]
        transition_matrices[i] = np.array([
            [1.0, dt],
            [-omega_i**2 * dt, 1.0 - 2.0 * damping_factor * omega_i * dt]
        ], dtype=np.float64)
    
    return transition_matrices


def extract_initial_positions(trajectories):
    """Extract initial positions from trajectory data (3D)."""
    return trajectories[:, 0, :] if trajectories is not None else None


def apply_manipulation(state_y, fft_matrix, frequencies, d, p, alpha):
    """Apply phase manipulation to the state vector at a specific position for 3D."""
    omegas = 2.0 * np.pi * frequencies
    q = state_y[:, 0] - 1j * state_y[:, 1] / (omegas + 1e-8)
    
    # d is now 3D direction vector
    strength = np.abs(fft_matrix[p] @ d.reshape(3, 1)) * alpha
    angle = -np.angle(fft_matrix[p] @ d.reshape(3, 1)) + np.pi / 2
    q = strength.flatten() * np.exp(1j * angle).flatten()
    
    state_y[:, 0] = np.real(q)
    state_y[:, 1] = -np.imag(q) * (omegas + 1e-8)
    return state_y


def simulate_dynamics(state_y, fft_matrix, transition_matrices, frequencies, 
                     timesteps, manipulation_params=None):
    """Simulate trajectory dynamics over time with optional manipulation for 3D."""
    K = len(frequencies)
    omegas = 2.0 * np.pi * frequencies
    displacements = []
    
    for t in range(timesteps):
        q = state_y[:, 0] - 1j * state_y[:, 1] / (omegas + 1e-8)
        
        if t == 0 and manipulation_params is not None:
            state_y = apply_manipulation(
                state_y, fft_matrix, frequencies,
                manipulation_params['d'],
                manipulation_params['p'],
                manipulation_params['alpha']
            )
            q = state_y[:, 0] - 1j * state_y[:, 1] / (omegas + 1e-8)
        
        # fft_matrix is now N×K×3, q is K×1, result is N×3
        displacement = np.real(fft_matrix * q[None, :, None]).sum(axis=1)
        displacements.append(displacement)
        
        for i in range(K):
            state_y[i] = transition_matrices[i] @ state_y[i]
    
    return np.array(displacements)


def save_displacements(displacements, output_dir):
    """Save displacements to a file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'displacements.npy', displacements)
    print(f"✓ Saved displacements to {output_dir / 'displacements.npy'}")
    print(f"  Shape: {displacements.shape} (timesteps, N_gaussians, 3)")


def print_fft_summary(fft_data):
    """Print summary of 3D FFT results."""
    summary = fft_data['summary']
    frequencies = fft_data['frequencies']
    power_x = fft_data['power_x']
    power_y = fft_data['power_y']
    power_z = fft_data['power_z']
    
    print("\n" + "=" * 60)
    print("3D FFT Analysis Results")
    print("=" * 60)
    
    print("\n📊 Data Shapes:")
    print(f"  FFT X (complex):     {fft_data['fft_x'].shape}")
    print(f"  FFT Y (complex):     {fft_data['fft_y'].shape}")
    print(f"  FFT Z (complex):     {fft_data['fft_z'].shape}")
    print(f"  Power X:             {power_x.shape}")
    print(f"  Power Y:             {power_y.shape}")
    print(f"  Power Z:             {power_z.shape}")
    print(f"  Frequencies:         {frequencies.shape}")
    
    print("\n📈 Summary:")
    print(f"  FPS:                 {summary.get('fps', 'N/A')}")
    if 'n_trajectories' in summary:
        print(f"  Number of Gaussians: {summary['n_trajectories']}")
    if 'n_frames' in summary:
        print(f"  Number of frames:    {summary['n_frames']}")
    
    print("\n💪 Power Statistics:")
    total_power = power_x + power_y + power_z
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
        freq_unit = 'Hz' if selected_modes.get('fps') else 'cycles/frame'
        print(f"  Frequency range: {selected_modes['modes'].min():.4f} - {selected_modes['modes'].max():.4f} {freq_unit}")
        if 'sum_powers' in selected_modes:
            print(f"  Total power range: {selected_modes['sum_powers'].min():.2e} - {selected_modes['sum_powers'].max():.2e}")


def print_frequency_selection(selected, selected_modes):
    """Print selected frequencies information."""
    print("\n🎯 Using pre-selected frequency modes")
    mode_indices = selected['indices']
    print(f"  Selected {len(mode_indices)} modes with indices: {mode_indices}")
    if selected_modes is not None and 'modes' in selected_modes:
        print(f"  Corresponding frequencies: {selected_modes['modes']}")
    
    print(f"\n🎯 Selected Frequencies:")
    for i, (freq, power) in enumerate(zip(selected['frequencies'], selected['total_power']), 1):
        period = 1.0 / freq if freq > 0 else float('inf')
        print(f"  {i:2d}. {freq:8.4f} Hz  |  Period: {period:8.4f} sec  |  Power: {power:.2e}")


def print_fft_matrix_info(fft_matrix):
    """Print FFT matrix information for 3D."""
    N, K = fft_matrix.shape[:2]
    print(f"\n{'='*60}")
    print(f"Built FFT Matrix (3D)")
    print(f"{'='*60}")
    print(f"  Shape: {fft_matrix.shape} (N={N} Gaussians, K={K} modes, 3 dimensions)")
    print(f"  Dtype: {fft_matrix.dtype}")
    print(f"  Memory: {fft_matrix.nbytes / 1024 / 1024:.2f} MB")


def print_state_info(state_y):
    """Print state information."""
    print(f"\n✓ Initialized state vector")
    print(f"  Shape: {state_y.shape} (K modes, 2 state variables)")
    print(f"  Initial values: all zeros")


def print_transition_matrices_info(transition_matrices, frequencies, dt, damping_factor):
    """Print transition matrices information."""
    print(f"\n✓ Created transition matrices")
    print(f"  Shape: {transition_matrices.shape} (K modes, 2x2 matrices)")
    print(f"  Time step (dt): {dt:.6f}")
    print(f"  Damping factor: {damping_factor}")
    print(f"  Number of modes: {len(frequencies)}")


def print_initial_positions_info(initial_positions):
    """Print initial positions information."""
    if initial_positions is not None:
        print(f"\n✓ Loaded initial Gaussian positions")
        print(f"  Shape: {initial_positions.shape} (N Gaussians, 3 coordinates)")
        print(f"  Position range:")
        print(f"    X: [{initial_positions[:, 0].min():.2f}, {initial_positions[:, 0].max():.2f}]")
        print(f"    Y: [{initial_positions[:, 1].min():.2f}, {initial_positions[:, 1].max():.2f}]")
        print(f"    Z: [{initial_positions[:, 2].min():.2f}, {initial_positions[:, 2].max():.2f}]")
    else:
        print(f"\n⚠ No initial positions loaded (trajectories.npy not found)")


def main():
    parser = argparse.ArgumentParser(
        description="Modal simulation for 3D Gaussian trajectory manipulation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--trajectory-dir", 
        required=True, 
        help="Path to trajectories directory containing FFT results and selected_modes.npy"
    )
    parser.add_argument(
        "--dt", 
        type=float, 
        default=None, 
        help="Time step in seconds (default: 0.1/fps)"
    )
    parser.add_argument(
        "--damping-factor", 
        type=float, 
        default=0.03, 
        help="Damping coefficient zeta (default: 0.03)"
    )
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=300, 
        help="Number of simulation timesteps (default: 300)"
    )
    parser.add_argument(
        "--manipulation-strength", 
        type=float, 
        default=1e-4, 
        help="Manipulation strength alpha (default: 1e-4)"
    )
    parser.add_argument(
        "--manipulation-gaussian-idx",
        type=int,
        default=None,
        help="Index of Gaussian to manipulate (default: random)"
    )
    parser.add_argument(
        "--manipulation-direction",
        type=float,
        nargs=3,
        default=None,
        help="3D manipulation direction vector (default: random unit vector)"
    )
    
    args = parser.parse_args()
    
    # Load FFT results
    fft_data = load_fft_results(args.trajectory_dir)
    print_fft_summary(fft_data)
    
    # Load selected modes
    selected_modes = load_selected_modes(args.trajectory_dir)
    if selected_modes is None:
        raise FileNotFoundError(
            f"selected_modes.npy not found in {args.trajectory_dir}. "
            "Please run 2_evaluate_candidate_modes.py first to generate selected modes."
        )
    
    print_selected_modes_info(selected_modes)
    selected = select_frequencies_from_modes(
        fft_data['fft_x'], fft_data['fft_y'], fft_data['fft_z'],
        fft_data['power_x'], fft_data['power_y'], fft_data['power_z'],
        fft_data['frequencies'],
        selected_modes
    )
    print_frequency_selection(selected, selected_modes)

    # Build FFT matrix (N×K×3)
    fft_matrix = build_fft_matrix(selected)
    print_fft_matrix_info(fft_matrix)
    
    # Extract initial positions
    initial_positions = extract_initial_positions(fft_data['trajectories'])
    print_initial_positions_info(initial_positions)

    # Initialize state
    K = len(selected['frequencies'])
    state_y = np.zeros((K, 2), dtype=np.float64)
    print_state_info(state_y)
    
    # Create transition matrices
    fps = fft_data['summary'].get('fps', None)
    if fps is None:
        fps = 30  # Default FPS if not specified
        print(f"⚠ FPS not found in summary, using default: {fps}")
    dt = args.dt if args.dt else 0.1 / fps
    transition_matrices = create_transition_matrices(
        selected['frequencies'], dt, args.damping_factor
    )
    print_transition_matrices_info(transition_matrices, selected['frequencies'], dt, args.damping_factor)

    # Setup manipulation parameters
    N = fft_matrix.shape[0]
    
    # Random direction vector (3D)
    if args.manipulation_direction is not None:
        d = np.array(args.manipulation_direction)
        d = d / np.linalg.norm(d)  # Normalize
    else:
        d = np.random.rand(3)
        d = d / np.linalg.norm(d)
    
    # Select Gaussian to manipulate
    if args.manipulation_gaussian_idx is not None:
        if args.manipulation_gaussian_idx < 0 or args.manipulation_gaussian_idx >= N:
            raise ValueError(f"Invalid Gaussian index: {args.manipulation_gaussian_idx} (must be in [0, {N-1}])")
        p = args.manipulation_gaussian_idx
    else:
        p = np.random.randint(0, N)
    
    manipulation_params = {
        'd': d, 
        'p': p, 
        'alpha': args.manipulation_strength
    }
    
    print(f"\n🎯 Manipulation Parameters:")
    print(f"  Gaussian index: {p}")
    print(f"  Direction vector: [{d[0]:.4f}, {d[1]:.4f}, {d[2]:.4f}]")
    print(f"  Strength (alpha): {args.manipulation_strength}")
    
    # Simulate dynamics
    displacements = simulate_dynamics(
        state_y, fft_matrix, transition_matrices, 
        selected['frequencies'], args.timesteps,
        manipulation_params=manipulation_params
    )
    print(f"\n✓ Simulated displacements over {args.timesteps} timesteps")
    print(f"  Shape: {displacements.shape} (timesteps, N_gaussians, 3)")
    print(f"  Displacement range:")
    print(f"    X: [{displacements[:, :, 0].min():.4f}, {displacements[:, :, 0].max():.4f}]")
    print(f"    Y: [{displacements[:, :, 1].min():.4f}, {displacements[:, :, 1].max():.4f}]")
    print(f"    Z: [{displacements[:, :, 2].min():.4f}, {displacements[:, :, 2].max():.4f}]")
    
    # Save displacements
    save_displacements(displacements, args.trajectory_dir)
    
    # Save simulation parameters
    sim_params = {
        'dt': dt,
        'damping_factor': args.damping_factor,
        'num_modes': K,
        'fps': fps,
        'timesteps': args.timesteps,
        'manipulation_strength': args.manipulation_strength,
        'manipulation_gaussian_idx': p,
        'manipulation_direction': d.tolist(),
        'selected_frequencies': selected['frequencies'].tolist()
    }
    
    output_dir = Path(args.trajectory_dir)
    np.save(output_dir / "simulation_params.npy", sim_params)
    print(f"✓ Saved simulation parameters to {output_dir / 'simulation_params.npy'}")
    
    print("\n✓ 3D Gaussian manipulation simulation complete!")
    
    return {
        'selected': selected,
        'fft_matrix': fft_matrix,
        'displacements': displacements,
        'manipulation_params': manipulation_params,
        'initial_positions': initial_positions
    }


if __name__ == "__main__":
    main()

