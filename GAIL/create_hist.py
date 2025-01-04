import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_combined_reward_histogram(data_dir, output_path=None):
    # Find all pkl files in the directory
    dataset_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pkl")]

    all_rewards = {}
    for dataset_file in dataset_files:
        with open(dataset_file, "rb") as f:
            rollouts = pickle.load(f)
        rewards = [np.sum(r.rews) for r in rollouts]
        dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
        all_rewards[dataset_name] = rewards

    plt.figure(figsize=(10, 6))
    for dataset_name, rewards in all_rewards.items():
        plt.hist(rewards, bins=20, alpha=0.7, label=dataset_name)

    plt.xlabel("Episode Rewards")
    plt.ylabel("Occurance")
    plt.title("Different Dataset Distributions for GAIL training")
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Histogram saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    data_dir = "/Users/manuelsenge/Documents/TUM/Semester 4/Robotic_Seminar/workspace/GAIL/models/GAIL/final_data"
    output_path = "/Users/manuelsenge/Documents/TUM/Semester 4/Robotic_Seminar/workspace/GAIL/models/GAIL/reward_histogram.png"
    plot_combined_reward_histogram(data_dir, output_path)
