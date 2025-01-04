import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.signal import savgol_filter


# Mapping of original column names to WandB labels
column_mapping = {
    "DQfD_random_no_train - DQfD_0/score": "Random Policy",
    "DQfD_bad_data_under_100 - DQfD_0/score": "under 100",
    "DQfD_bad_data_100_to_300 - DQfD_0/score": "100 to 300",
    "DQfD_middle_300_400 - DQfD_0/score": "300 to 400",
    "DQfD_mixed_good_400_500 - DQfD_0/score": "400 to 500",
    "DQfD_perfect_data_500 - DQfD_0/score": "only perfect data (500)",
}

ddqn_column_mapping = {
    "full_policy_DDQN - DDPG_0/score": "standard DDQN",
}


def smooth_data(data, window_size=11, polyorder=2):
    return savgol_filter(data, window_size, polyorder)


# Step 1: Load all CSV files
dqfd_file_paths = glob.glob("/Users/manuelsenge/Documents/TUM/Semester 4/Robotic_Seminar/workspace/DQfD_new/DQfD/model_analysis/results/DQfD/*.csv")
ddqn_file_paths = glob.glob("/Users/manuelsenge/Documents/TUM/Semester 4/Robotic_Seminar/workspace/DQfD_new/DQfD/model_analysis/results/DDQN/*.csv")

dqfd_dfs = [pd.read_csv(file) for file in dqfd_file_paths]
ddqn_dfs = [pd.read_csv(file) for file in ddqn_file_paths]

# Step 2: Rename columns for all DQfD DataFrames
for df in dqfd_dfs:
    for original, new_name in column_mapping.items():
        for i in range(5):  # Assume indices range from 0 to 4
            index_name = original.replace("DQfD_0", f"DQfD_{i}")
            if index_name in df.columns:
                df.rename(columns={index_name: f"{new_name} {i}"}, inplace=True)

# Step 2.1: Rename columns for all DDQN DataFrames
for df in ddqn_dfs:
    for original, new_name in ddqn_column_mapping.items():
        for i in range(5):  # Assume indices range from 0 to 4
            index_name = original.replace("DDPG_0", f"DDPG_{i}")
            if index_name in df.columns:
                df.rename(columns={index_name: f"{new_name} {i}"}, inplace=True)

# Step 3: Aggregate data for each category across all DQfD files
categories = list(column_mapping.values())
ddqn_categories = list(ddqn_column_mapping.values())

data = {cat: [] for cat in categories}
ddqn_data = {cat: [] for cat in ddqn_categories}

# Collect data from each DQfD file
for df in dqfd_dfs:
    for cat in categories:
        relevant_columns = [col for col in df.columns if col.startswith(cat)]
        data[cat].append(df[relevant_columns].mean(axis=1).values)

# Collect data from each DDQN file
for df in ddqn_dfs:
    for cat in ddqn_categories:
        relevant_columns = [col for col in df.columns if col.startswith(cat)]
        ddqn_data[cat].append(df[relevant_columns].mean(axis=1).values)

# Step 4: Compute mean and std across files
means = {cat: np.mean(data[cat], axis=0) for cat in categories}
stds = {cat: np.std(data[cat], axis=0) for cat in categories}

ddqn_means = {cat: np.mean(ddqn_data[cat], axis=0) for cat in ddqn_categories}
ddqn_stds = {cat: np.std(ddqn_data[cat], axis=0) for cat in ddqn_categories}

# Step 5: Plot the mean and standard deviation
plt.figure(figsize=(12, 8))
colors = ['green', 'red', 'blue', 'purple', 'orange', 'yellow', 'black']  # Added 'black' for DDQN

# Plot DQfD categories
for i, cat in enumerate(categories):
    steps = range(len(means[cat]))
    smoothed_mean = smooth_data(means[cat], window_size=11, polyorder=2)
    smoothed_std = smooth_data(stds[cat], window_size=11, polyorder=2)

    plt.plot(steps, smoothed_mean, label=cat, color=colors[i])
    plt.fill_between(steps, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, color=colors[i], alpha=0.1)

# Plot DDQN category
for i, cat in enumerate(ddqn_categories):
    steps = range(len(ddqn_means[cat]))
    smoothed_mean = smooth_data(ddqn_means[cat], window_size=11, polyorder=2)
    smoothed_std = smooth_data(ddqn_stds[cat], window_size=11, polyorder=2)

    plt.plot(steps, smoothed_mean, label=cat, color=colors[len(categories) + i])  # Use 'black' for DDQN
    plt.fill_between(steps, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, color=colors[len(categories) + i], alpha=0.3)

# Step 6: Format the plot
plt.xlabel("Iteration")
plt.ylabel("Return")
plt.title("Accumulated Return for 500 Steps")
plt.legend(loc="upper left", fontsize=10)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()
