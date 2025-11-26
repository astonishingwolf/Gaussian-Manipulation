"""
Modal Simulation for Video Manipulation
Usage:
    python video-manipulation/direct_manipulation.py \
        --trajectory-dir data/optical_flows/ball/trajectories \
        --damping-factor 0.03
    
    Note: Requires selected_modes.npy from 4_evaluate_candidate_modes.py
    If --use-rotated is set, requires rotated_fft_x.npy and rotated_fft_y.npy from dominant_orientation.py
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
from utils import (
    print_fft_summary, print_selected_modes_info, print_frequency_selection,
    print_fft_matrix_info, print_transition_matrices_info, print_state_info,
    print_initial_positions_info, print_brightness_filter_info,
    visualize_results, print_save_results, save_displacements,
    render_displacements, interpolate_sparse_displacements, interpolate_sparse_displacements_legacy
)


def load_fft_results(trajectories_dir, use_rotated=False):
    """Load all FFT analysis results from a directory."""
    trajectories_dir = Path(trajectories_dir)    
    required_files = ['fft_x.npy', 'fft_y.npy', 'power_x.npy', 'power_y.npy', 
                     'frequencies.npy', 'fft_summary.npy']
    
    for file in required_files:
        if not (trajectories_dir / file).exists():
            raise FileNotFoundError(f"Required file not found: {trajectories_dir / file}")
    
    # Load rotated FFT if requested and available
    if use_rotated:
        rotated_x_file = trajectories_dir / 'rotated_fft_x.npy'
        rotated_y_file = trajectories_dir / 'rotated_fft_y.npy'
        if rotated_x_file.exists() and rotated_y_file.exists():
            fft_x = np.load(rotated_x_file)
            fft_y = np.load(rotated_y_file)
            print("✓ Using rotated FFT data (from dominant_orientation.py)")
        else:
            print("⚠ Rotated FFT files not found, using original FFT data")
            print(f"  Expected: {rotated_x_file} and {rotated_y_file}")
            fft_x = np.load(trajectories_dir / 'fft_x.npy')
            fft_y = np.load(trajectories_dir / 'fft_y.npy')
    else:
        fft_x = np.load(trajectories_dir / 'fft_x.npy')
        fft_y = np.load(trajectories_dir / 'fft_y.npy')
    
    fft_data = {
        'fft_x': fft_x,
        'fft_y': fft_y,
        'power_x': np.load(trajectories_dir / 'power_x.npy'),
        'power_y': np.load(trajectories_dir / 'power_y.npy'),
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
    modes_file = trajectories_dir / 'selected_modes.npy'
    
    if modes_file.exists():
        return np.load(modes_file, allow_pickle=True).item()
    return None


def select_frequencies_from_modes(fft_x, fft_y, power_x, power_y, frequencies, selected_modes):
    """Select frequencies and their corresponding FFT coefficients using pre-selected modes."""
    if selected_modes is None or 'indices' not in selected_modes:
        raise ValueError("selected_modes is required and must contain 'indices' key. Run 4_evaluate_candidate_modes.py first.")
    mode_indices = np.array(selected_modes['indices'])
    if np.any(mode_indices < 0) or np.any(mode_indices >= len(frequencies)):
        raise ValueError(f"Invalid frequency indices in selected_modes: {mode_indices}")
    
    selected_frequencies = frequencies[mode_indices]
    selected_fft_x = fft_x[:, mode_indices]
    selected_fft_y = fft_y[:, mode_indices]
    selected_power_x = power_x[:, mode_indices]
    selected_power_y = power_y[:, mode_indices]
    
    # Calculate total power for selected frequencies
    total_power_all = np.mean(power_x + power_y, axis=0)
    total_power_selected = total_power_all[mode_indices]
    
    result = {
        'frequencies': selected_frequencies,
        'fft_x': selected_fft_x,
        'fft_y': selected_fft_y,
        'power_x': selected_power_x,
        'power_y': selected_power_y,
        'indices': mode_indices,
        'total_power': total_power_selected
    }
    
    return result


def build_fft_matrix(selected):
    """Build N×K×2 matrix from selected FFT results."""
    return np.stack([selected['fft_x'], selected['fft_y']], axis=-1)


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
    """Extract initial positions from trajectory data."""
    return trajectories[:, 0, :] if trajectories is not None else None


def filter_bright_trajectories(initial_positions, frame, pixel_threshold):
    """Filter trajectories based on pixel brightness."""
    pixel_values = []
    for pos in initial_positions:
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < frame.shape[0] and 0 <= y < frame.shape[1]:
            pixel_values.append(np.mean(frame[x, y, :]))
        else:
            pixel_values.append(0)
    return np.where(np.array(pixel_values) > pixel_threshold)[0]


def apply_manipulation(state_y, fft_matrix, frequencies, d, p, alpha):
    """Apply phase manipulation to the state vector at a specific position."""
    omegas = 2.0 * np.pi * frequencies
    q = state_y[:, 0] - 1j * state_y[:, 1] / (omegas + 1e-8)
    
    strength = np.abs(fft_matrix[p] @ d.reshape(2, 1)) * alpha
    angle = -np.angle(fft_matrix[p] @ d.reshape(2, 1)) + np.pi / 2
    q = strength.flatten() * np.exp(1j * angle).flatten()
    
    state_y[:, 0] = np.real(q)
    state_y[:, 1] = -np.imag(q) * (omegas + 1e-8)
    return state_y


def simulate_dynamics(state_y, fft_matrix, transition_matrices, frequencies, 
                     timesteps, manipulation_params=None):
    """Simulate trajectory dynamics over time with optional manipulation."""
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
        
        displacement = np.real(fft_matrix * q[None, :, None]).sum(axis=1)
        displacements.append(displacement)
        
        for i in range(K):
            state_y[i] = transition_matrices[i] @ state_y[i]
    
    return np.array(displacements)


def save_simulation_results(output_dir, selected, fft_matrix, state_y, 
                           transition_matrices, initial_positions, sim_params):
    """Save all simulation outputs to disk."""
    output_dir = Path(output_dir)
    num_modes = len(selected['frequencies'])
    
    np.save(output_dir / "selected_frequencies.npy", selected['frequencies'])
    np.save(output_dir / "selected_fft_x.npy", selected['fft_x'])
    np.save(output_dir / "selected_fft_y.npy", selected['fft_y'])
    np.save(output_dir / "selected_fft_matrix.npy", fft_matrix)
    np.save(output_dir / "state_y_init.npy", state_y)
    np.save(output_dir / "transition_matrices.npy", transition_matrices)
    np.save(output_dir / "simulation_params.npy", sim_params)
    
    if initial_positions is not None:
        np.save(output_dir / "initial_positions.npy", initial_positions)
    
    print_save_results(output_dir, num_modes, initial_positions)


def main():
    parser = argparse.ArgumentParser(
        description="Modal simulation for trajectory manipulation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--trajectory-dir", required=True, help="Path to trajectories directory containing FFT results and selected_modes.npy")
    parser.add_argument("--video-path", default="/hdd_data/nakul/soham/video-data/videos/00133.mp4", help="Path to video file for visualization (default: data/00133.mp4)")
    parser.add_argument("--use-rotated", action="store_true", help="Use rotated FFT data from dominant_orientation.py (rotated_fft_x.npy, rotated_fft_y.npy)")
    parser.add_argument("--dt", type=float, default=None, help="Time step in seconds (default: 0.1/fps)")
    parser.add_argument("--damping-factor", type=float, default=0.03, help="Damping coefficient zeta (default: 0.03)")
    parser.add_argument("--timesteps", type=int, default=300, help="Number of simulation timesteps (default: 300)")
    parser.add_argument("--manipulation-strength", type=float, default=1e-4, help="Manipulation strength alpha (default: 1e-4)")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of trajectories to visualize (default: 100)")
    parser.add_argument("--use_legacy_interpolation", action="store_true", help="Use legacy interpolation method")
    
    args = parser.parse_args()
    
    fft_data = load_fft_results(args.trajectory_dir, use_rotated=args.use_rotated)
    print_fft_summary(fft_data)
    
    selected_modes = load_selected_modes(args.trajectory_dir)
    if selected_modes is None:
        raise FileNotFoundError(
            f"selected_modes.npy not found in {args.trajectory_dir}. "
            "Please run 4_evaluate_candidate_modes.py first to generate selected modes."
        )
    
    print_selected_modes_info(selected_modes)
    selected = select_frequencies_from_modes(
        fft_data['fft_x'], fft_data['fft_y'], 
        fft_data['power_x'], fft_data['power_y'],
        fft_data['frequencies'],
        selected_modes
    )
    print_frequency_selection(selected, selected_modes)

    fft_matrix = build_fft_matrix(selected)
    print_fft_matrix_info(fft_matrix)
    initial_positions = extract_initial_positions(fft_data['trajectories'])
    print_initial_positions_info(initial_positions)

    K = len(selected['frequencies'])
    state_y = np.zeros((K, 2), dtype=np.float64)
    print_state_info(state_y)
    
    dt = args.dt if args.dt else 0.1 / fft_data['summary']['fps']
    transition_matrices = create_transition_matrices(
        selected['frequencies'], dt, args.damping_factor
    )
    print_transition_matrices_info(transition_matrices, selected['frequencies'], dt, args.damping_factor)

    cap = cv2.VideoCapture(args.video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise FileNotFoundError(f"Could not read video: {args.video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pixel_average = np.mean(frame)

    if initial_positions is not None:
        valid_indices = filter_bright_trajectories(initial_positions, frame, pixel_average)
        print_brightness_filter_info(valid_indices, initial_positions, pixel_average)
        n_samples = min(args.n_samples, len(valid_indices))
        sample_indices = np.random.choice(valid_indices, size=n_samples, replace=False)
    else:
        sample_indices = np.random.choice(fft_matrix.shape[0], 
                                         size=min(args.n_samples, fft_matrix.shape[0]), 
                                         replace=False)
        valid_indices = sample_indices

    d = np.random.rand(2)
    d = d / np.linalg.norm(d)
    p = np.random.choice(valid_indices)
    manipulation_params = {'d': d, 'p': p, 'alpha': args.manipulation_strength}
    
    displacements = simulate_dynamics(
        state_y, fft_matrix, transition_matrices, 
        selected['frequencies'], args.timesteps,
        manipulation_params=manipulation_params
    )
    print(f"\n✓ Simulated displacements over {args.timesteps} timesteps: shape {displacements.shape}")
    
    visualize_results(
        displacements, initial_positions, sample_indices,
        manipulation_params, frame, args.trajectory_dir
    )
    render_output_dir = Path(args.trajectory_dir) / "rendered_displacements"
    save_displacements(displacements, args.trajectory_dir)
    
    dense_displacements = None
    if initial_positions is not None:
        if args.use_legacy_interpolation:
            dense_displacements = interpolate_sparse_displacements_legacy(displacements, initial_positions, frame.shape[:2])
        else:
            dense_displacements = interpolate_sparse_displacements(displacements, initial_positions, frame.shape[:2])
    
    if dense_displacements is not None:
        render_displacements(
            dense_displacements,
            frame,
            render_output_dir,
            fps=20,
            initial_positions=initial_positions
        )
    else:
        print(
            "⚠ Skipping dense rendering: Provide initial_positions and ensure sparse interpolation succeeds."
        )
    
    sim_params = {
        'dt': dt, 'damping_factor': args.damping_factor, 'num_modes': K,
        'fps': fft_data['summary']['fps'], 'timesteps': args.timesteps,
        'manipulation_strength': args.manipulation_strength,
        'use_rotated': args.use_rotated
    }
    save_simulation_results(
        args.trajectory_dir, selected, fft_matrix, state_y,
        transition_matrices, initial_positions, sim_params
    )
    
    return {
        'selected': selected, 'fft_matrix': fft_matrix,
        'displacements': displacements, 'manipulation_params': manipulation_params
    }



if __name__ == "__main__":
    main()
