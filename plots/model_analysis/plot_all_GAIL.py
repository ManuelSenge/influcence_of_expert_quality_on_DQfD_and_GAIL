import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def smooth_data(data, window_size=11, polyorder=2):
    return savgol_filter(data, window_size, polyorder)

file_path = "/Users/manuelsenge/Documents/TUM/Semester 4/Robotic_Seminar/workspace/plots/model_analysis/results/GAIL/GAIL.csv"
df = pd.read_csv(file_path)

# Extract unique run types dynamically
run_types = set(col.split(f"GAIL_training_")[1].split("_HalfCheetah")[0]
                for col in df.columns if "train/reward" in col)

# Prepare data structures for plotting
data = {run_type: [] for run_type in run_types}

# Group by run type and collect data for each seed
for run_type in run_types:
    reward_columns = [col for col in df.columns if run_type in col and "train/reward" in col]
    for col in reward_columns:
        rewards = df[col].dropna().values  # Exclude NaN values
        data[run_type].append(rewards)

means = {}
stds = {}

# handle varying lengths of runs
for run_type, runs in data.items():
    min_length = min(len(run) for run in runs)
    
    truncated_runs = [run[:min_length] for run in runs]
    
    truncated_runs = np.array(truncated_runs)
    means[run_type] = np.mean(truncated_runs, axis=0)
    stds[run_type] = np.std(truncated_runs, axis=0)

fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=False)
colors = ['green', 'red', 'blue', 'purple', 'orange', 'yellow', 'black']  # Adjust as needed

for i, run_type in enumerate(run_types):
    steps = df['train/timestep'][:len(means[run_type])]  # Match steps to truncated data length
    smoothed_mean = smooth_data(means[run_type], window_size=11, polyorder=2)
    smoothed_std = smooth_data(stds[run_type], window_size=11, polyorder=2)

    axs[0].plot(steps, smoothed_mean, label=run_type, color=colors[i % len(colors)])
    axs[0].fill_between(steps, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std,
                        color=colors[i % len(colors)], alpha=0.1)

axs[0].set_title("GAIL Training Results")
axs[0].set_ylabel("Reward")
axs[0].legend(loc="upper left", fontsize=10)
axs[0].set_xlim(0, None)
axs[0].grid(alpha=0.5)


for i, run_type in enumerate(run_types):
    steps = df['train/timestep'][:len(means[run_type])]  # Match steps to truncated data length
    smoothed_mean = smooth_data(means[run_type], window_size=11, polyorder=2)
    smoothed_std = smooth_data(stds[run_type], window_size=11, polyorder=2)

    # Filter for the last 3000 steps
    last_3000_idx = steps >= 300000#(steps.max() - 300000)
    axs[1].plot(steps[last_3000_idx], smoothed_mean[last_3000_idx], label=run_type, color=colors[i % len(colors)])
    axs[1].fill_between(steps[last_3000_idx],
                        smoothed_mean[last_3000_idx] - smoothed_std[last_3000_idx],
                        smoothed_mean[last_3000_idx] + smoothed_std[last_3000_idx],
                        color=colors[i % len(colors)], alpha=0.1)

#axs[1].set_title("GAIL Training Results (Last 3000 Steps)")
axs[1].set_xlabel("Timesteps")
axs[1].set_ylabel("Reward")
axs[1].set_xlim(300000, None)
axs[1].legend(loc="upper left", fontsize=10)
axs[1].grid(alpha=0.5)

plt.tight_layout()
plt.show()
