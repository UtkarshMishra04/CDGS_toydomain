#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read all JSON files from the sweep_horizon_resampling_matrix directory
sweep_dir = Path("profile/sweep_horizon_resampling_matrix")
data = []

for json_file in sorted(sweep_dir.glob("*.json")):
    with open(json_file) as f:
        entry = json.load(f)
        data.append({
            'horizon': entry['horizon_length'],
            'resampling': entry['num_resampling_steps'],
            'time': entry['time'],
            'success_rate': entry['success_rate']
        })

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

# Create first plot: Time heatmap
fig1, ax1 = plt.subplots(figsize=(14, 6))
im1 = ax1.imshow(time_matrix, aspect='auto', cmap='viridis', origin='lower')
ax1.set_xlabel('Horizon Length', fontsize=12)
ax1.set_ylabel('Number of Resampling Steps', fontsize=12)
ax1.set_title('Time (s) vs Horizon Length and Resampling Steps', fontsize=14)

# Set ticks
x_ticks = np.arange(0, len(horizons), max(1, len(horizons)//10))
ax1.set_xticks(x_ticks)
ax1.set_xticklabels([horizons[i] for i in x_ticks])
ax1.set_yticks(range(len(resampling_steps)))
ax1.set_yticklabels(resampling_steps)

cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Time (s)', fontsize=12)
plt.tight_layout()
plt.savefig('profile/sweep_horizon_resampling_matrix/time_heatmap.png', dpi=150)
print("Saved time_heatmap.png")

# Create second plot: Success rate heatmap
fig2, ax2 = plt.subplots(figsize=(14, 6))
im2 = ax2.imshow(success_matrix, aspect='auto', cmap='RdYlGn', origin='lower', vmin=0, vmax=1)
ax2.set_xlabel('Horizon Length', fontsize=12)
ax2.set_ylabel('Number of Resampling Steps', fontsize=12)
ax2.set_title('Success Rate vs Horizon Length and Resampling Steps', fontsize=14)

# Set ticks
ax2.set_xticks(x_ticks)
ax2.set_xticklabels([horizons[i] for i in x_ticks])
ax2.set_yticks(range(len(resampling_steps)))
ax2.set_yticklabels(resampling_steps)

cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Success Rate', fontsize=12)
plt.tight_layout()
plt.savefig('profile/sweep_horizon_resampling_matrix/success_heatmap.png', dpi=150)
print("Saved success_heatmap.png")

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
ax3a.set_title('Time vs Horizon Length', fontsize=14)
ax3a.legend()
ax3a.grid(True, alpha=0.3)

ax3b.set_xlabel('Horizon Length', fontsize=12)
ax3b.set_ylabel('Success Rate', fontsize=12)
ax3b.set_title('Success Rate vs Horizon Length', fontsize=14)
ax3b.set_ylim([0, 1.1])
ax3b.legend()
ax3b.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('profile/sweep_horizon_resampling_matrix/line_plots.png', dpi=150)
print("Saved line_plots.png")

plt.show()
