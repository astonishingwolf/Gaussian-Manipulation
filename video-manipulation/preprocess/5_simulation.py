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


def main():
    parser = argparse.ArgumentParser(description="Load FFT results for simulation")
    parser.add_argument("--trajectory-dir", required=True,
                       help="Path to trajectories directory containing FFT results")
    parser.add_argument("--top-k", type=int, default=16,
                       help="Number of top frequencies to select for simulation (default: 16)")
    parser.add_argument("--dt", type=float, default=None,
                       help="Time step in seconds (default: 1/fps from FFT analysis)")
    parser.add_argument("--damping-factor", type=float, default=0.03,
                       help="Damping coefficient zeta (default: 0.03)")
    
    args = parser.parse_args()
    
    # Load FFT results
    fft_data = load_fft_results(args.trajectory_dir)
    
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
    
    # Try to load pre-selected modes
    selected_modes = load_selected_modes(args.trajectory_dir)
    
    # Select top K frequencies
    print(f"\n{'='*60}")
    if selected_modes is not None:
        print(f"Using Pre-Selected Modes from 4_evaluate_candidate_modes.py")
    else:
        print(f"Selecting Top {args.top_k} Frequencies")
    print(f"{'='*60}")
    
    selected = select_top_k_frequencies(
        fft_x, fft_y, power_x, power_y, frequencies, 
        top_k=args.top_k, 
        selected_modes=selected_modes
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
    
    # Load initial positions from trajectories
    trajectories = fft_data['trajectories']
    if trajectories is not None:
        # Get initial positions (first frame: t=0)
        initial_positions = trajectories[:, 0, :]  # Shape: (N, 2)
        print(f"\n✓ Loaded initial positions from trajectories")
        print(f"  Shape: {initial_positions.shape} (N={initial_positions.shape[0]}, 2)")
        print(f"  X range: [{initial_positions[:, 0].min():.2f}, {initial_positions[:, 0].max():.2f}]")
        print(f"  Y range: [{initial_positions[:, 1].min():.2f}, {initial_positions[:, 1].max():.2f}]")
    else:
        initial_positions = None
        print(f"\n⚠ No initial positions available")

    # Initialize state vector Y with shape (K, 2)
    # State for each frequency: [position, velocity]
    # Set K based on loaded modes or top_k argument
    if selected_modes is not None:
        K = len(selected_modes['indices'])
    else:
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
            alpha = 1E-26
            p = np.random.randint(0, len(fft_matrix))
            strength = np.abs(fft_matrix[p] @ d.reshape(2, 1)) * alpha
            angle = -1 * np.angle(fft_matrix[p] @ d.reshape(2, 1)) + np.pi / 2

            phase_shift = np.exp(1j * angle)
            q = strength.flatten() * phase_shift.flatten()

            # print("@@@@@ strength @@@@@ ", strength)

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

    # add the initial positions if available
    if initial_positions is not None:
        displacements += initial_positions[None, :, :]
    
    import matplotlib.pyplot as plt

    # plot displacement on the videos first frame
    import cv2
    video_path = 'data/00133.mp4'
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame)
    # Randomly select 100 samples
    n_samples = min(100, len(initial_positions))
    
    # only select the points that has pixel value larger than average
    pixel_average = np.mean(frame)
    print(f"Average pixel value in the frame: {pixel_average:.2f}")
    if initial_positions is not None:
        pixel_values = []
        for pos in initial_positions:
            x, y = int(pos[0]), int(pos[1])
            if x >= 0 and x < frame.shape[1] and y >= 0 and y < frame.shape[0]:
                pixel_values.append(np.mean(frame[y, x, :]))
            else:
                pixel_values.append(0)
        pixel_values = np.array(pixel_values)
        avg_pixel_value = np.mean(pixel_values)
        valid_indices = np.where(pixel_values > pixel_average)[0]
        print(f"Number of valid initial positions above average pixel value: {len(valid_indices)}")
        if len(valid_indices) < n_samples:
            sample_indices = np.random.choice(valid_indices, size=len(valid_indices), replace=False)
        else:
            sample_indices = np.random.choice(valid_indices, size=n_samples, replace=False)
    else:
        sample_indices = np.random.choice(len(initial_positions), size=n_samples, replace=False)
    
    for i in sample_indices:
        # Swap x and y coordinates to match image display orientation
        plt.arrow(initial_positions[i, 1], initial_positions[i, 0],
                  displacements[-1, i, 1] - initial_positions[i, 1],
                  displacements[-1, i, 0] - initial_positions[i, 0],
                  color='red', head_width=5, head_length=5)
    plt.title('Displacements on First Frame')
    plt.savefig('displacements_on_frame.png')
    plt.clf()

    for i in sample_indices:
        plt.plot(displacements[:, i, 0], displacements[:, i, 1], label=f'Point {i}')
    plt.title('X-Y Displacements over Time for Sampled Points')
    plt.xlabel('Displacement X')
    plt.ylabel('Displacement Y')
    plt.savefig('xy_displacements_sampled.png')
    plt.clf()
    

    
    # Save selected frequencies and matrix
    output_dir = Path(args.trajectory_dir)
    np.save(output_dir / f"selected_top{args.top_k}_frequencies.npy", selected['frequencies'])
    np.save(output_dir / f"selected_top{args.top_k}_fft_x.npy", selected['fft_x'])
    np.save(output_dir / f"selected_top{args.top_k}_fft_y.npy", selected['fft_y'])
    np.save(output_dir / f"selected_top{args.top_k}_fft_matrix.npy", fft_matrix)
    np.save(output_dir / f"state_y_init.npy", state_y)
    np.save(output_dir / f"transition_matrices.npy", transition_matrices)
    if initial_positions is not None:
        np.save(output_dir / f"initial_positions.npy", initial_positions)
    
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
    if initial_positions is not None:
        print(f"  - initial_positions.npy  [N×2 initial positions]")
    
    return selected, fft_matrix, state_y, transition_matrices



if __name__ == "__main__":
    main()
