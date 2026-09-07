import matplotlib.pyplot as plt
import numpy as np

# Set seed for reproducible synthetic data curve
np.random.seed(42)
episodes = np.arange(1, 201)

# Synthetic convergence trajectories (replace with actual evaluation logs)
mean_time_raw = 35.0 * np.exp(-episodes / 35.0) + 6.2
noise = 12.0 * np.exp(-episodes / 50.0) + 1.2
runs_time = np.clip([mean_time_raw + np.random.normal(0, noise, len(episodes)) for _ in range(5)], 5.0, 60.0)

mean_time = np.mean(runs_time, axis=0)
std_time = np.std(runs_time, axis=0)
mean_reward = -mean_time
std_reward = std_time

# Exponential Moving Average for trend smoothing
def ema(values, alpha=0.8):
    smoothed = []
    curr = values[0]
    for v in values:
        curr = curr * alpha + (1 - alpha) * v
        smoothed.append(curr)
    return np.array(smoothed)

smoothed_reward = ema(mean_reward, 0.8)
smoothed_time = ema(mean_time, 0.8)

# Matplotlib Publication Styling
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

# 1. Reward Curve Plot
ax1.plot(episodes, mean_reward, alpha=0.25, color='#1f77b4', linewidth=1, label='Raw Episode Reward')
ax1.plot(episodes, smoothed_reward, color='#1f77b4', linewidth=2.5, label='Smoothed Reward (EMA)')
ax1.fill_between(episodes, smoothed_reward - std_reward, smoothed_reward + std_reward, color='#1f77b4', alpha=0.15)
ax1.set_title(r'Mean Episode Reward ($R_{\mathrm{striker}}$)', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Training Episodes', fontsize=11)
ax1.set_ylabel('Reward', fontsize=11)
ax1.legend(loc='lower right', frameon=True)
ax1.grid(True, linestyle=':', alpha=0.6)

# 2. Time-to-Ball Metric Plot
ax2.plot(episodes, mean_time, alpha=0.25, color='#ff7f0e', linewidth=1, label='Raw Time')
ax2.plot(episodes, smoothed_time, color='#ff7f0e', linewidth=2.5, label='Smoothed Time (EMA)')
ax2.fill_between(episodes, smoothed_time - std_time, smoothed_time + std_time, color='#ff7f0e', alpha=0.15)
ax2.set_title(r'Time to Reach Ball ($\Delta t_{\mathrm{reach}}$)', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Training Episodes', fontsize=11)
ax2.set_ylabel('Time to Ball (seconds)', fontsize=11)
ax2.legend(loc='upper right', frameon=True)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('striker_learning_curve.png', bbox_inches='tight') # Vector PDF export for LaTeX
#plt.show()