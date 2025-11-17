#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Base directories
results_base = Path("profile/sweep_horizon_resampling_matrix")
plots_base = Path("plots/sweep_horizon_resampling_matrix")

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
                'horizon': entry['horizon_length'],
                'resampling': entry['num_resampling_steps'],
                'time': entry['time'],
                'success_rate': entry['success_rate'],
                'dataset': entry.get('dataset', 'unknown'),
                'training_mode': entry.get('training_mode', 'unknown')
            })

    if not data:
        print(f"  No data found in {subdir.name}, skipping...")
        continue

    # Get dataset and training mode from first entry
    dataset = data[0]['dataset']
    training_mode = data[0]['training_mode']

    # Get unique values for both dimensions
    horizons = sorted(list(set(d['horizon'] for d in data)))
    horizons = [x for x in horizons if x <= 40]
    resampling_steps = sorted(list(set(d['resampling'] for d in data)))

    # Create matrices for time and success rate
    time_matrix = np.full((len(resampling_steps), len(horizons)), np.nan)
    success_matrix = np.full((len(resampling_steps), len(horizons)), np.nan)

    # Fill the matrices
    for entry in data:
        if entry['horizon'] > 40:
            continue
        h_idx = horizons.index(entry['horizon'])
        r_idx = resampling_steps.index(entry['resampling'])
        time_matrix[r_idx, h_idx] = entry['time']
        success_matrix[r_idx, h_idx] = entry['success_rate']

    # Create output directory
    if subdir == results_base:
        output_dir = plots_base
    else:
        output_dir = plots_base / subdir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create title suffix
    title_suffix = f"\n({dataset} dataset, {training_mode} training, pruning disabled)"

    # Create first plot: Time heatmap
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    im1 = ax1.imshow(time_matrix, aspect='auto', cmap='viridis', origin='lower')
    ax1.set_xlabel('Horizon Length', fontsize=12)
    ax1.set_ylabel('Number of Resampling Steps', fontsize=12)
    ax1.set_title(f'Time (s) vs Horizon Length and Resampling Steps{title_suffix}', fontsize=14)

    # Set ticks
    x_ticks = np.arange(0, len(horizons), max(1, len(horizons)//10))
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([horizons[i] for i in x_ticks])
    ax1.set_yticks(range(len(resampling_steps)))
    ax1.set_yticklabels(resampling_steps)

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Time (s)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'time_heatmap.png', dpi=150)
    print(f"  Saved {output_dir / 'time_heatmap.png'}")
    plt.close(fig1)

    # Create second plot: Success rate heatmap
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    im2 = ax2.imshow(success_matrix, aspect='auto', cmap='RdYlGn', origin='lower', vmin=0, vmax=1)
    ax2.set_xlabel('Horizon Length', fontsize=12)
    ax2.set_ylabel('Number of Resampling Steps', fontsize=12)
    ax2.set_title(f'Success Rate vs Horizon Length and Resampling Steps{title_suffix}', fontsize=14)

    # Set ticks
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([horizons[i] for i in x_ticks])
    ax2.set_yticks(range(len(resampling_steps)))
    ax2.set_yticklabels(resampling_steps)

    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Success Rate', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'success_heatmap.png', dpi=150)
    print(f"  Saved {output_dir / 'success_heatmap.png'}")
    plt.close(fig2)

    # Create third plot: Line plots for each resampling step value
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 6))

    for resampling in resampling_steps:
        # Get data for this resampling step
        times_for_resampling = []
        success_for_resampling = []
        horizons_for_resampling = []

        for h in horizons:
            matching = [d for d in data if d['horizon'] == h and d['resampling'] == resampling]
            if matching:
                horizons_for_resampling.append(h)
                times_for_resampling.append(matching[0]['time'])
                success_for_resampling.append(matching[0]['success_rate'])

        ax3a.plot(horizons_for_resampling, times_for_resampling, 'o-', linewidth=2, markersize=4,
                  label=f'Resampling={resampling}')
        ax3b.plot(horizons_for_resampling, success_for_resampling, 'o-', linewidth=2, markersize=4,
                  label=f'Resampling={resampling}')

    ax3a.set_xlabel('Horizon Length', fontsize=12)
    ax3a.set_ylabel('Time (s)', fontsize=12)
    ax3a.set_title(f'Time vs Horizon Length{title_suffix}', fontsize=14)
    ax3a.legend()
    ax3a.grid(True, alpha=0.3)

    ax3b.set_xlabel('Horizon Length', fontsize=12)
    ax3b.set_ylabel('Success Rate', fontsize=12)
    ax3b.set_title(f'Success Rate vs Horizon Length{title_suffix}', fontsize=14)
    ax3b.set_ylim([0, 1.1])
    ax3b.legend()
    ax3b.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'line_plots.png', dpi=150)
    print(f"  Saved {output_dir / 'line_plots.png'}")
    plt.close(fig3)

print("\nAll plots generated successfully!")
