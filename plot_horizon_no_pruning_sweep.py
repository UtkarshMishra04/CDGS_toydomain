#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Read all JSON files from the sweep_horizon_no_pruning directory
sweep_dir = Path("profile/sweep_horizon_no_pruning")
data = []

for json_file in sorted(sweep_dir.glob("*.json")):
    with open(json_file) as f:
        entry = json.load(f)
        data.append({
            'horizon': entry['horizon_length'],
            'time': entry['time'],
            'success_rate': entry['success_rate']
        })

# Sort by horizon length
data.sort(key=lambda x: x['horizon'])

# Extract data for plotting
horizons = [d['horizon'] for d in data]
times = [d['time'] for d in data]
success_rates = [d['success_rate'] for d in data]

# Create first plot: Time vs Horizon
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(horizons, times, 'o-', linewidth=2, markersize=6)
ax1.set_xlabel('Horizon Length', fontsize=12)
ax1.set_ylabel('Time (s)', fontsize=12)
ax1.set_title('Time vs Horizon Length (No Pruning/Resampling)', fontsize=14)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('profile/sweep_horizon_no_pruning/time_vs_horizon.png', dpi=150)
print("Saved time_vs_horizon.png")

# Create second plot: Success Rate vs Horizon
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(horizons, success_rates, 'o-', linewidth=2, markersize=6, color='green')
ax2.set_xlabel('Horizon Length', fontsize=12)
ax2.set_ylabel('Success Rate', fontsize=12)
ax2.set_title('Success Rate vs Horizon Length (No Pruning/Resampling)', fontsize=14)
ax2.set_ylim([0, 1.1])
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('profile/sweep_horizon_no_pruning/success_vs_horizon.png', dpi=150)
print("Saved success_vs_horizon.png")

plt.show()
