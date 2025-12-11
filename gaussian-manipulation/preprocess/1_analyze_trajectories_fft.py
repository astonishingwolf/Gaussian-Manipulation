"""
Perform FFT analysis on saved 3D trajectory data (N, T, 3).

Usage:
    # Basic FFT analysis
    python gaussian-manipulation/preprocess/1_analyze_trajectories_fft.py \
        --input data/trajectories.npy \
        --fps 30
    
    # With visualization
    python gaussian-manipulation/preprocess/1_analyze_trajectories_fft.py \
        --input data/trajectories.npy \
        --fps 30 --visualize
    
    # Save FFT results
    python gaussian-manipulation/preprocess/1_analyze_trajectories_fft.py \
        --input data/trajectories.npy \
        --fps 30 --save-fft
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_trajectories(trajectory_file):
    """
    Load trajectory data from numpy file.
    
    Args:
        trajectory_file: Path to .npy file containing trajectories of shape (T, N, 3)
                        where T is time steps, N is number of points, 3 is (x, y, z)
        
    Returns:
        trajectories: numpy array of shape (N, T, 3) transposed from input
    """
    trajectory_file = Path(trajectory_file)
    
    if not trajectory_file.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")
    
    trajectories = np.load(trajectory_file)
    print(f"Loaded trajectories: shape {trajectories.shape}")
    
    # Validate shape
    if len(trajectories.shape) != 3 or trajectories.shape[2] != 3:
        raise ValueError(f"Expected trajectories shape (T, N, 3) or (N, T, 3), got {trajectories.shape}")
    
    # breakpoint()
    # trajectories = trajectories.transpose(1, 0, 2)
    print(f"  N={trajectories.shape[0]} points, T={trajectories.shape[1]} time steps")
    
    return trajectories


def compute_trajectory_fft(trajectories, axis=1, fps=None):
    """
    Compute FFT of 3D trajectories along temporal axis.
    
    Args:
        trajectories: numpy array of shape (N, T, 3) where N is number of points,
                     T is number of time steps, and 3 is (x, y, z) coordinates
        axis: axis along which to compute FFT (default: 1, the time axis)
        fps: frames per second for converting frequencies to Hz
        
    Returns:
        fft_x, fft_y, fft_z: FFT results for each coordinate
        frequencies: frequency bins
        power_x, power_y, power_z: power spectra for each coordinate
    """
    # Extract x, y, and z components
    x_traj = trajectories[:, :, 0]  # Shape: (N, T)
    y_traj = trajectories[:, :, 1]  # Shape: (N, T)
    z_traj = trajectories[:, :, 2]  # Shape: (N, T)
    
    # Compute FFT along time axis
    fft_x = np.fft.fft(x_traj, axis=axis)
    fft_y = np.fft.fft(y_traj, axis=axis)
    fft_z = np.fft.fft(z_traj, axis=axis)
    
    # Frequency bins
    n_frames = trajectories.shape[axis]
    frequencies = np.fft.fftfreq(n_frames)
    
    # Convert to Hz if fps is provided
    if fps is not None:
        frequencies = frequencies * fps
    
    # Power spectrum
    power_x = np.abs(fft_x) ** 2
    power_y = np.abs(fft_y) ** 2
    power_z = np.abs(fft_z) ** 2
    
    return fft_x, fft_y, fft_z, frequencies, power_x, power_y, power_z


def analyze_dominant_frequencies(frequencies, power_x, power_y, power_z, top_k=5):
    """
    Find dominant frequencies in the trajectory data.
    
    Args:
        frequencies: frequency bins
        power_x, power_y, power_z: power spectra for each coordinate
        top_k: number of top frequencies to return
        
    Returns:
        top_frequencies: top k dominant frequencies
        top_powers: corresponding power values
        pos_freqs: all positive frequencies
        pos_power: average power at positive frequencies
    """
    # Average power across all trajectories
    avg_power = np.mean(power_x + power_y + power_z, axis=0)
    
    # Only consider positive frequencies
    pos_mask = frequencies > 0
    pos_freqs = frequencies[pos_mask]
    pos_power = avg_power[pos_mask]
    
    # Get top k frequencies
    top_indices = np.argsort(pos_power)[-top_k:][::-1]
    top_frequencies = pos_freqs[top_indices]
    top_powers = pos_power[top_indices]
    
    return top_frequencies, top_powers, pos_freqs, pos_power


def visualize_trajectory_fft(trajectories, fft_x, fft_y, fft_z, frequencies, power_x, power_y, power_z, 
                            output_dir=None, fps=None):
    """Visualize FFT analysis of trajectories."""
    freq_unit = 'Hz' if fps else 'cycles/frame'
    time_unit = 'seconds' if fps else 'frames'
    
    # Average power across all trajectories
    avg_power_x = np.mean(power_x, axis=0)
    avg_power_y = np.mean(power_y, axis=0)
    avg_power_z = np.mean(power_z, axis=0)
    avg_power_total = avg_power_x + avg_power_y + avg_power_z
    
    # Positive frequencies only
    pos_mask = frequencies >= 0
    pos_freqs = frequencies[pos_mask]
    
    # Get a few example trajectories
    n_examples = min(5, len(trajectories))
    example_indices = np.linspace(0, len(trajectories) - 1, n_examples, dtype=int)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Example trajectories in time domain
    ax1 = fig.add_subplot(gs[0, :])
    for idx in example_indices:
        x_pos = trajectories[idx, :, 0]
        y_pos = trajectories[idx, :, 1]
        z_pos = trajectories[idx, :, 2]
        ax1.plot(x_pos, label=f'Point {idx} - X', alpha=0.7)
        ax1.plot(y_pos, label=f'Point {idx} - Y', alpha=0.7, linestyle='--')
        ax1.plot(z_pos, label=f'Point {idx} - Z', alpha=0.7, linestyle=':')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Position')
    ax1.set_title('Example Trajectories - Time Domain')
    ax1.legend(ncol=3, fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average power spectrum - X direction
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(pos_freqs[1:], avg_power_x[pos_mask][1:], 'b-', linewidth=2)
    ax2.set_xlabel(f'Frequency ({freq_unit})')
    ax2.set_ylabel('Power')
    ax2.set_title('X-Direction Power Spectrum (averaged)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average power spectrum - Y direction
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(pos_freqs[1:], avg_power_y[pos_mask][1:], 'r-', linewidth=2)
    ax3.set_xlabel(f'Frequency ({freq_unit})')
    ax3.set_ylabel('Power')
    ax3.set_title('Y-Direction Power Spectrum (averaged)')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Total power spectrum
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(pos_freqs[1:], avg_power_total[pos_mask][1:], 'g-', linewidth=2)
    ax4.set_xlabel(f'Frequency ({freq_unit})')
    ax4.set_ylabel('Power')
    ax4.set_title('Total Power Spectrum (X + Y + Z, averaged)')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Heatmap of power spectrum for all trajectories
    ax5 = fig.add_subplot(gs[2, 1])
    power_heatmap = power_x + power_y + power_z  # Total power
    im = ax5.imshow(np.log10(power_heatmap[:, pos_mask][:, 1:] + 1e-10), 
                    aspect='auto', cmap='hot', interpolation='nearest')
    ax5.set_xlabel(f'Frequency ({freq_unit})')
    ax5.set_ylabel('Trajectory Index')
    ax5.set_title('Power Spectrum Heatmap (log scale)')
    plt.colorbar(im, ax=ax5, label='Log10(Power)')
    
    plt.suptitle('Trajectory FFT Analysis', fontsize=14, fontweight='bold')
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'fft_analysis.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved FFT visualization to {output_dir / 'fft_analysis.png'}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Perform FFT analysis on 3D trajectory data (N, T, 3)")
    parser.add_argument("--input", required=True, 
                       help="Path to trajectories.npy file")
    parser.add_argument("--output", default=None, 
                       help="Output directory for results (default: same as input)")
    
    # FFT options
    parser.add_argument("--fps", type=float, default=None, 
                       help="Video FPS for converting frequencies from cycles/frame to Hz")
    parser.add_argument("--top-k", type=int, default=5, 
                       help="Number of top dominant frequencies to report (default: 5)")
    
    # Visualization options
    parser.add_argument("--visualize", action="store_true", 
                       help="Generate FFT visualization plots")
    
    # Save options
    parser.add_argument("--save-fft", action="store_true", 
                       help="Save FFT results as numpy arrays")
    
    args = parser.parse_args()
    
    # Load trajectories
    trajectory_file = Path(args.input)
    trajectories = load_trajectories(trajectory_file)
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = trajectory_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute FFT
    print("\nComputing FFT of trajectories...")
    if args.fps:
        print(f"Using FPS: {args.fps} (frequencies will be in Hz)")
    else:
        print("No FPS provided (frequencies will be in cycles/frame)")
    
    fft_x, fft_y, fft_z, frequencies, power_x, power_y, power_z = compute_trajectory_fft(
        trajectories, fps=args.fps
    )
    
    print(f"FFT computed: shape {fft_x.shape}")
    
    # Analyze dominant frequencies
    print("\nAnalyzing dominant frequencies...")
    top_frequencies, top_powers, pos_freqs, pos_power = analyze_dominant_frequencies(
        frequencies, power_x, power_y, power_z, top_k=args.top_k
    )
    
    freq_unit = 'Hz' if args.fps else 'cycles/frame'
    period_unit = 'seconds' if args.fps else 'frames'
    
    print(f"\nTop {args.top_k} Dominant Frequencies:")
    for i, (freq, power) in enumerate(zip(top_frequencies, top_powers), 1):
        period = 1.0 / freq if freq > 0 else np.inf
        print(f"  {i}. Frequency: {freq:.4f} {freq_unit}, "
              f"Period: {period:.2f} {period_unit}, "
              f"Power: {power:.2e}")
    
    # Visualize
    if args.visualize:
        print("\nGenerating FFT visualization...")
        visualize_trajectory_fft(
            trajectories, fft_x, fft_y, fft_z, frequencies, power_x, power_y, power_z,
            output_dir, fps=args.fps
        )
    
    # Save FFT results
    if args.save_fft:
        print("\nSaving FFT results...")
        
        # Save complex FFT (for phase analysis if needed)
        np.save(output_dir / "fft_x.npy", fft_x)
        np.save(output_dir / "fft_y.npy", fft_y)
        np.save(output_dir / "fft_z.npy", fft_z)
        
        # Save power spectra (real values)
        np.save(output_dir / "power_x.npy", power_x)
        np.save(output_dir / "power_y.npy", power_y)
        np.save(output_dir / "power_z.npy", power_z)
        
        # Save frequencies
        np.save(output_dir / "frequencies.npy", frequencies)
        
        # Save summary
        summary = {
            'fps': args.fps,
            'top_frequencies': top_frequencies,
            'top_powers': top_powers,
            'n_trajectories': trajectories.shape[0],
            'n_frames': trajectories.shape[1]
        }
        np.save(output_dir / "fft_summary.npy", summary)
        
        print(f"✓ Saved FFT results to {output_dir}/")
        print(f"  - fft_x.npy, fft_y.npy, fft_z.npy (complex FFT)")
        print(f"  - power_x.npy, power_y.npy, power_z.npy (power spectra)")
        print(f"  - frequencies.npy")
        print(f"  - fft_summary.npy")
    
    print("\n✓ FFT analysis complete!")


if __name__ == "__main__":
    main()
