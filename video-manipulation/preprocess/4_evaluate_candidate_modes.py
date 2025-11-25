"""
python video-manipulation/preprocess/4_evaluate_candidate_modes.py \
        --input "${TRAJECTORY_DIR}" \
        --visualize \
        --save-modes

"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from scipy.signal import find_peaks

def choose_modes(power_x, frequencies, power_y, peak_prominence=0.05, num_modes_to_select=5):
    """
    Identifies the candidate frequency modes (local peaks) based on the sum of 
    averaged power spectra from x and y directions, selecting peaks with the 
    highest combined power from the positive frequency side only.
    """
    
    avg_power_x = power_x.mean(axis=0)
    avg_power_y = power_y.mean(axis=0)
    
    # Sum the power from both directions
    avg_power_sum = avg_power_x + avg_power_y
    
    pos_freq_mask = frequencies >= 0
    
    pos_freqs = frequencies[pos_freq_mask]
    pos_power_x = avg_power_x[pos_freq_mask]
    pos_power_y = avg_power_y[pos_freq_mask]
    pos_power_sum = avg_power_sum[pos_freq_mask]
    
    # Find peaks in the combined power spectrum
    indices, _ = find_peaks(pos_power_sum, prominence=peak_prominence)

    peak_powers_sum = pos_power_sum[indices]
    
    # Sort by combined power (highest first)
    sorted_indices = indices[np.argsort(peak_powers_sum)[::-1]]
    
    # Select top modes
    top_indices = sorted_indices[:min(num_modes_to_select, len(sorted_indices))]
    
    selected_modes = pos_freqs[top_indices]
    selected_powers_x = pos_power_x[top_indices]
    selected_powers_y = pos_power_y[top_indices]
    selected_powers_sum = pos_power_sum[top_indices]
    
    # Get the original frequency indices (accounting for positive frequency mask)
    original_indices = np.where(pos_freq_mask)[0][top_indices]
    
    selected_modes_dict = {
        'modes': selected_modes,
        'indices': original_indices,
        'x_powers': selected_powers_x,
        'y_powers': selected_powers_y,
        'sum_powers': selected_powers_sum
    }
    
    # Return the selected modes (positive only) and the original averaged spectra (full size)
    return selected_modes_dict, avg_power_x, avg_power_y

def visualize_candidate_modes(freq_axis, avg_power_x, avg_power_y, 
                              selected_modes, selected_powers_x, selected_powers_y,
                              selected_powers_sum, output_path):
    """Plot dominant candidate frequencies over the global power spectrum."""
    
    pos_mask = freq_axis > 0
    pos_freqs = freq_axis[pos_mask]
    pos_power_x = avg_power_x[pos_mask]
    pos_power_y = avg_power_y[pos_mask]
    pos_power_sum = pos_power_x + pos_power_y
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    
    # Plot X-direction power spectrum
    ax1 = axes[0]
    ax1.plot(pos_freqs, pos_power_x, 'b-', linewidth=2, label='Power Spectrum (X-direction)')
    ax1.scatter(selected_modes, selected_powers_x, color='red', s=100, 
                marker='o', zorder=5, label='Selected Modes', edgecolors='darkred', linewidths=2)
                
    for i, (freq, power) in enumerate(zip(selected_modes, selected_powers_x)):
        ax1.annotate(f'{freq:.4f} Hz', xy=(freq, power), xytext=(10, 10),
                    textcoords='offset points', fontsize=9, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax1.set_xlabel('Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('Power (Log Scale)', fontsize=12)
    ax1.set_title('X-Direction Power Spectrum with Selected Dominant Modes', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Plot Y-direction power spectrum
    ax2 = axes[1]
    ax2.plot(pos_freqs, pos_power_y, 'r-', linewidth=2, label='Power Spectrum (Y-direction)')
    ax2.scatter(selected_modes, selected_powers_y, color='blue', s=100,
                marker='o', zorder=5, label='Selected Modes', edgecolors='darkblue', linewidths=2)
    for i, (freq, power) in enumerate(zip(selected_modes, selected_powers_y)):
        ax2.annotate(f'{freq:.4f} Hz', xy=(freq, power), xytext=(10, 10),
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Power (Log Scale)', fontsize=12)
    ax2.set_title('Y-Direction Power Spectrum with Selected Dominant Modes', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Plot Combined power spectrum (sum)
    ax3 = axes[2]
    ax3.plot(pos_freqs, pos_power_sum, 'g-', linewidth=2, label='Combined Power Spectrum (X+Y)')
    ax3.scatter(selected_modes, selected_powers_sum, color='purple', s=100,
                marker='o', zorder=5, label='Selected Modes', edgecolors='darkmagenta', linewidths=2)
    for i, (freq, power) in enumerate(zip(selected_modes, selected_powers_sum)):
        ax3.annotate(f'{freq:.4f} Hz', xy=(freq, power), xytext=(10, 10),
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax3.set_xlabel('Frequency (Hz)', fontsize=12)
    ax3.set_ylabel('Power (Log Scale)', fontsize=12)
    ax3.set_title('Combined Power Spectrum (X+Y) with Selected Dominant Modes', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved candidate modes visualization to {output_path}")
    plt.close()

def save_modes(selected_modes, mode_dir):
    os.makedirs(mode_dir, exist_ok=True)
    # Using allow_pickle=True is default for dicts, but it's good practice to be aware of.
    np.save(os.path.join(mode_dir, "selected_modes.npy"), selected_modes, allow_pickle=True)
    return 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Finds and visualizes dominant frequency modes from averaged power spectra.")
    parser.add_argument("--trajectory_dir", type=str, required=True, help="Directory containing power_x.npy, power_y.npy, and frequencies.npy")
    parser.add_argument("--mode_dir", type=str, help="Output directory for modes and visualization. Defaults to <trajectory_dir>/modes")
    parser.add_argument("--peak-prominence", type=float, default=0.05, help="Minimum prominence for peak detection")
    parser.add_argument("--num-modes", type=int, default=3, help="Number of modes to select")
    args = parser.parse_args()

    trajectory_dir = args.trajectory_dir
    mode_dir = args.mode_dir if args.mode_dir else trajectory_dir 
    os.makedirs(mode_dir, exist_ok=True)
    
    # --- Data Loading ---
    try:
        power_x = np.load(os.path.join(trajectory_dir, "power_x.npy"))
        power_y = np.load(os.path.join(trajectory_dir, "power_y.npy"))
        frequencies = np.load(os.path.join(trajectory_dir, "frequencies.npy")) 
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure 'power_x.npy', 'power_y.npy', and 'frequencies.npy' are in the trajectory directory.")
        exit()
        
    # --- Input Validation ---
    if power_x.shape[-1] != frequencies.shape[0] or power_y.shape[-1] != frequencies.shape[0]:
        print("Error: Power array last dimension must match frequency axis dimension.")
        print(f"power_x last dim: {power_x.shape[-1]}, frequency dim: {frequencies.shape[0]}")
        exit()
    
    selected_modes, avg_power_x, avg_power_y = choose_modes(
        power_x, frequencies, power_y, 
        peak_prominence=args.peak_prominence,
        num_modes_to_select=args.num_modes
    )

    print(selected_modes)
    
    save_modes(selected_modes, mode_dir)
    
    # Visualize candidate modes
    viz_path = os.path.join(mode_dir, "candidate_modes_visualization.png")
    visualize_candidate_modes(
        frequencies, avg_power_x, avg_power_y,
        selected_modes['modes'], selected_modes['x_powers'], selected_modes['y_powers'],
        selected_modes['sum_powers'], viz_path
    )