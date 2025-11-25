"""
Load FFT analysis results and use them for simulation.

Usage:
    python video-manipulation/preprocess/4_simulation.py \
        --input data/optical_flows/ball/trajectories
"""

import argparse
import numpy as np
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


def select_top_k_frequencies(fft_x, fft_y, power_x, power_y, frequencies, top_k=16):
    """
    Select top K dominant frequencies and their corresponding FFT coefficients.
    
    Args:
        fft_x: Complex FFT array for x-coordinates (N, T)
        fft_y: Complex FFT array for y-coordinates (N, T)
        power_x: Power spectrum for x (N, T)
        power_y: Power spectrum for y (N, T)
        frequencies: Frequency bins (T,)
        top_k: Number of top frequencies to select
    
    Returns:
        dict: Dictionary containing selected frequencies and coefficients
    """
    # Calculate total power across all trajectories
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
    
    result = {
        'frequencies': selected_frequencies,
        'fft_x': selected_fft_x,
        'fft_y': selected_fft_y,
        'power_x': selected_power_x,
        'power_y': selected_power_y,
        'indices': top_k_indices,
        'total_power': pos_power[top_k_indices_pos]
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


def main():
    parser = argparse.ArgumentParser(description="Load FFT results for simulation")
    parser.add_argument("--input", required=True,
                       help="Path to trajectories directory containing FFT results")
    parser.add_argument("--top-k", type=int, default=16,
                       help="Number of top frequencies to select for simulation (default: 16)")
    parser.add_argument("--dt", type=float, default=None,
                       help="Time step in seconds (default: 1/fps from FFT analysis)")
    parser.add_argument("--damping-factor", type=float, default=0.03,
                       help="Damping coefficient zeta (default: 0.03)")
    
    args = parser.parse_args()
    
    # Load FFT results
    fft_data = load_fft_results(args.input)
    
    # Print summary
    print_fft_summary(fft_data)
    
    # Access the data as numpy arrays
    fft_x = fft_data['fft_x']  # Complex array (N, T)
    fft_y = fft_data['fft_y']  # Complex array (N, T)
    power_x = fft_data['power_x']  # Real array (N, T)
    power_y = fft_data['power_y']  # Real array (N, T)
    frequencies = fft_data['frequencies']  # Real array (T,)
    summary = fft_data['summary']  # Dictionary
    
    print("\n✓ FFT data loaded as numpy arrays and ready for simulation")
    
    # Select top K frequencies
    print(f"\n{'='*60}")
    print(f"Selecting Top {args.top_k} Frequencies")
    print(f"{'='*60}")
    
    selected = select_top_k_frequencies(
        fft_x, fft_y, power_x, power_y, frequencies, top_k=args.top_k
    )
    
    print(f"\n✓ Selected {args.top_k} dominant frequencies:")
    print(f"  Shape of selected FFT X: {selected['fft_x'].shape}")
    print(f"  Shape of selected FFT Y: {selected['fft_y'].shape}")
    
    print(f"\n🎯 Selected Frequencies (sorted by power):")
    for i, (freq, power) in enumerate(zip(selected['frequencies'], selected['total_power']), 1):
        period = 1.0 / freq if freq > 0 else float('inf')
        print(f"  {i:2d}. {freq:8.4f} Hz  |  Period: {period:8.4f} sec  |  Power: {power:.2e}")
    
    # Build N×K×2 FFT matrix
    fft_matrix = build_fft_matrix(selected)

    # Initialize state vector Y with shape (K, 2)
    # State for each frequency: [position, velocity]
    K = args.top_k
    state_y = np.zeros((K, 2), dtype=np.float64)
    
    print(f"\n{'='*60}")
    print(f"Initialized State Vector Y")
    print(f"{'='*60}")
    print(f"  Shape: {state_y.shape} (K={K}, 2)")
    print(f"  State components: [position, velocity]")
    
    # Determine time step
    if args.dt is None:
        fps = summary['fps']
        dt = 1.0 / fps
        print(f"  Time step: {dt:.6f} seconds (from FPS={fps})")
    else:
        dt = args.dt
        print(f"  Time step: {dt:.6f} seconds (user-specified)")
    
    # Create transition matrices for all frequencies
    transition_matrices = create_transition_matrices(
        selected['frequencies'], dt, args.damping_factor
    )

    # simulate for 100 timesteps
    displacements = []
    timestep = 100
    for t in range(timestep):

        q = state_y[:, 0] - 1j * state_y[:, 1] / (2.0 * np.pi * selected['frequencies'] + 1e-8)

        # mainpulate for direction d, position p and strength alpha

        if t == 0:
            d = np.random.randn(2)
            alpha = 0.001
            p = np.random.randint(0, len(fft_matrix))
            strength = np.abs(fft_matrix[p] @ d.reshape(2, 1)) * alpha
            angle = -1 * np.angle(fft_matrix[p] @ d.reshape(2, 1)) + np.pi / 2

            phase_shift = np.exp(1j * angle)
            q = strength.flatten() * phase_shift.flatten()

            print("strength", strength)

            state_y[:, 0] = np.real(q)
            state_y[:, 1] = -np.imag(q) * (2.0 * np.pi * selected['frequencies'] + 1e-8)

        # =====================================================================

        # displacement
        displacement = np.real(fft_matrix * q[None, :, None])
        displacement = displacement.sum(axis=1)  # Sum over frequencies
        displacements.append(displacement)

        for i in range(K):
            state_y[i] = transition_matrices[i] @ state_y[i]

    displacements = np.array(displacements)  # Shape: (timesteps, N, 2)
    print(f"\n✓ Simulated displacements over {timestep} timesteps:")
    print(f"  Shape: {displacements.shape} (timesteps={timestep}, N={fft_matrix.shape[0]}, 2)")

    print(displacements)
    # plot the displacements for first 5 points for x-t plot
    import matplotlib.pyplot as plt
    for i in range(5):
        plt.plot(displacements[:, i, 0], label=f'Point {i} X')
    plt.title('X Displacements over Time for First 5 Points')
    plt.xlabel('Timestep')
    plt.ylabel('Displacement X')
    plt.legend()
    plt.savefig('x_displacements.png')
    plt.clf()

    

    
    # Save selected frequencies and matrix
    output_dir = Path(args.input)
    np.save(output_dir / f"selected_top{args.top_k}_frequencies.npy", selected['frequencies'])
    np.save(output_dir / f"selected_top{args.top_k}_fft_x.npy", selected['fft_x'])
    np.save(output_dir / f"selected_top{args.top_k}_fft_y.npy", selected['fft_y'])
    np.save(output_dir / f"selected_top{args.top_k}_fft_matrix.npy", fft_matrix)
    np.save(output_dir / f"state_y_init.npy", state_y)
    np.save(output_dir / f"transition_matrices.npy", transition_matrices)
    
    # Save simulation parameters
    sim_params = {
        'dt': dt,
        'damping_factor': args.damping_factor,
        'top_k': args.top_k,
        'fps': summary['fps']
    }
    np.save(output_dir / f"simulation_params.npy", sim_params)
    
    print(f"\n✓ Saved selected frequencies to {output_dir}/")
    print(f"  - selected_top{args.top_k}_frequencies.npy")
    print(f"  - selected_top{args.top_k}_fft_x.npy")
    print(f"  - selected_top{args.top_k}_fft_y.npy")
    print(f"  - selected_top{args.top_k}_fft_matrix.npy  [N×K×2 matrix]")
    print(f"  - state_y_init.npy  [K×2 initial state]")
    print(f"  - transition_matrices.npy  [K×2×2 matrices]")
    print(f"  - simulation_params.npy  [simulation parameters]")
    
    return selected, fft_matrix, state_y, transition_matrices



if __name__ == "__main__":
    main()
