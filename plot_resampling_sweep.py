#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Read all JSON files from the sweep_resampling directory
sweep_dir = Path("profile/sweep_resampling")
data = []

for json_file in sorted(sweep_dir.glob("*.json")):
    with open(json_file) as f:
        entry = json.load(f)
        data.append({
            'num_resampling_steps': entry['num_resampling_steps'],
            'time': entry['time'],
            'success_rate': entry['success_rate']
        })

# Sort by num_resampling_steps
data.sort(key=lambda x: x['num_resampling_steps'])

# Extract data for plotting
resampling_steps = [d['num_resampling_steps'] for d in data]
times = [d['time'] for d in data]
success_rates = [d['success_rate'] for d in data]

# Create first plot: Time vs Resampling Steps
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(resampling_steps, times, 'o-', linewidth=2, markersize=6)
ax1.set_xlabel('Number of Resampling Steps', fontsize=12)
ax1.set_ylabel('Time (s)', fontsize=12)
ax1.set_title('Time vs Number of Resampling Steps', fontsize=14)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('profile/sweep_resampling/time_vs_resampling.png', dpi=150)
print("Saved time_vs_resampling.png")

# Create second plot: Success Rate vs Resampling Steps
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(resampling_steps, success_rates, 'o-', linewidth=2, markersize=6, color='green')
ax2.set_xlabel('Number of Resampling Steps', fontsize=12)
ax2.set_ylabel('Success Rate', fontsize=12)
ax2.set_title('Success Rate vs Number of Resampling Steps', fontsize=14)
ax2.set_ylim([0, 1.1])
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('profile/sweep_resampling/success_vs_resampling.png', dpi=150)
print("Saved success_vs_resampling.png")

plt.show()
