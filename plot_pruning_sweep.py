#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Base directories
results_base = Path("profile/sweep_pruning")
plots_base = Path("plots/sweep_pruning")

# Find all subdirectories (one for each dataset/training_mode combination)
subdirs = [d for d in results_base.iterdir() if d.is_dir()]

if not subdirs:
    print("No subdirectories found. Looking for JSON files in root directory...")
    subdirs = [results_base]

for subdir in sorted(subdirs):
    print(f"\nProcessing {subdir.name}...")

    # Read all JSON files from this subdirectory
    data = []
    for json_file in sorted(subdir.glob("*.json")):
        with open(json_file) as f:
            entry = json.load(f)
            data.append({
                'pruning_start': entry['pruning_start'],
                'pruning_end': entry['pruning_end'],
                'time': entry['time'],
                'success_rate': entry['success_rate'],
                'dataset': entry.get('dataset', 'unknown'),
                'training_mode': entry.get('training_mode', 'unknown'),
                'horizon': entry.get('horizon_length', 10),
                'resampling': entry.get('num_resampling_steps', 10)
            })

    if not data:
        print(f"  No data found in {subdir.name}, skipping...")
        continue

    # Get dataset and training mode from first entry
    dataset = data[0]['dataset']
    training_mode = data[0]['training_mode']
    horizon = data[0]['horizon']
    resampling = data[0]['resampling']

    # Get unique values for both dimensions
    pruning_starts = sorted(list(set(d['pruning_start'] for d in data)))
    pruning_ends = sorted(list(set(d['pruning_end'] for d in data)))

    # Create matrices for time and success rate
    time_matrix = np.full((len(pruning_ends), len(pruning_starts)), np.nan)
    success_matrix = np.full((len(pruning_ends), len(pruning_starts)), np.nan)

    # Fill the matrices (only for valid pairs where end > start)
    for entry in data:
        s_idx = pruning_starts.index(entry['pruning_start'])
        e_idx = pruning_ends.index(entry['pruning_end'])
        time_matrix[e_idx, s_idx] = entry['time']
        success_matrix[e_idx, s_idx] = entry['success_rate']

    # Create output directory
    if subdir == results_base:
        output_dir = plots_base
    else:
        output_dir = plots_base / subdir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create title suffix
    title_suffix = f"\n({dataset} dataset, {training_mode} training, horizon={horizon}, resampling={resampling})"

    # Create first plot: Time heatmap
    fig1, ax1 = plt.subplots(figsize=(10, 9))
    im1 = ax1.imshow(time_matrix, aspect='auto', cmap='viridis', origin='lower')
    ax1.set_xlabel('Pruning Start', fontsize=12)
    ax1.set_ylabel('Pruning End', fontsize=12)
    ax1.set_title(f'Time (s) vs Pruning Start and End{title_suffix}', fontsize=14)

    # Set ticks
    ax1.set_xticks(range(len(pruning_starts)))
    ax1.set_xticklabels([f'{x:.1f}' for x in pruning_starts])
    ax1.set_yticks(range(len(pruning_ends)))
    ax1.set_yticklabels([f'{x:.1f}' for x in pruning_ends])

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Time (s)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'time_heatmap.png', dpi=150)
    print(f"  Saved {output_dir / 'time_heatmap.png'}")
    plt.close(fig1)

    # Create second plot: Success rate heatmap
    fig2, ax2 = plt.subplots(figsize=(10, 9))
    im2 = ax2.imshow(success_matrix, aspect='auto', cmap='RdYlGn', origin='lower', vmin=0, vmax=1)
    ax2.set_xlabel('Pruning Start', fontsize=12)
    ax2.set_ylabel('Pruning End', fontsize=12)
    ax2.set_title(f'Success Rate vs Pruning Start and End{title_suffix}', fontsize=14)

    # Set ticks
    ax2.set_xticks(range(len(pruning_starts)))
    ax2.set_xticklabels([f'{x:.1f}' for x in pruning_starts])
    ax2.set_yticks(range(len(pruning_ends)))
    ax2.set_yticklabels([f'{x:.1f}' for x in pruning_ends])

    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Success Rate', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'success_heatmap.png', dpi=150)
    print(f"  Saved {output_dir / 'success_heatmap.png'}")
    plt.close(fig2)

    # Create third plot: Line plots showing effect of pruning window size
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 6))

    # For each pruning_start value, plot time and success vs pruning_end
    for start in pruning_starts:
        # Get data for this start value
        times_for_start = []
        success_for_start = []
        ends_for_start = []

        for entry in data:
            if entry['pruning_start'] == start:
                ends_for_start.append(entry['pruning_end'])
                times_for_start.append(entry['time'])
                success_for_start.append(entry['success_rate'])

        if times_for_start:
            # Sort by end value
            sorted_data = sorted(zip(ends_for_start, times_for_start, success_for_start))
            ends_for_start, times_for_start, success_for_start = zip(*sorted_data)

            ax3a.plot(ends_for_start, times_for_start, 'o-', linewidth=2, markersize=4,
                      label=f'Start={start:.1f}')
            ax3b.plot(ends_for_start, success_for_start, 'o-', linewidth=2, markersize=4,
                      label=f'Start={start:.1f}')

    ax3a.set_xlabel('Pruning End', fontsize=12)
    ax3a.set_ylabel('Time (s)', fontsize=12)
    ax3a.set_title(f'Time vs Pruning End{title_suffix}', fontsize=14)
    ax3a.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3a.grid(True, alpha=0.3)

    ax3b.set_xlabel('Pruning End', fontsize=12)
    ax3b.set_ylabel('Success Rate', fontsize=12)
    ax3b.set_title(f'Success Rate vs Pruning End{title_suffix}', fontsize=14)
    ax3b.set_ylim([0, 1.1])
    ax3b.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3b.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'line_plots.png', dpi=150, bbox_inches='tight')
    print(f"  Saved {output_dir / 'line_plots.png'}")
    plt.close(fig3)

    # Create fourth plot: Pruning window size (end - start) vs metrics
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 6))

    window_sizes = []
    times_by_window = []
    success_by_window = []

    for entry in data:
        window_size = entry['pruning_end'] - entry['pruning_start']
        window_sizes.append(window_size)
        times_by_window.append(entry['time'])
        success_by_window.append(entry['success_rate'])

    ax4a.scatter(window_sizes, times_by_window, alpha=0.6, s=50)
    ax4a.set_xlabel('Pruning Window Size (End - Start)', fontsize=12)
    ax4a.set_ylabel('Time (s)', fontsize=12)
    ax4a.set_title(f'Time vs Pruning Window Size{title_suffix}', fontsize=14)
    ax4a.grid(True, alpha=0.3)

    ax4b.scatter(window_sizes, success_by_window, alpha=0.6, s=50, color='green')
    ax4b.set_xlabel('Pruning Window Size (End - Start)', fontsize=12)
    ax4b.set_ylabel('Success Rate', fontsize=12)
    ax4b.set_title(f'Success Rate vs Pruning Window Size{title_suffix}', fontsize=14)
    ax4b.set_ylim([0, 1.1])
    ax4b.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'window_size_plots.png', dpi=150)
    print(f"  Saved {output_dir / 'window_size_plots.png'}")
    plt.close(fig4)

print("\nAll plots generated successfully!")
