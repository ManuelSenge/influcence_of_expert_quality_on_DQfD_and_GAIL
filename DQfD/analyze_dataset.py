import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    task2return = {
        'perfect_data_500': [],
        'mixed_good_400_500': [],
        'middle_300_400': [],
        'bad_data_under_100': [],
        'bad_data_100_to_300': [],
        'random_no_train': []
    }

    exp_names = ['perfect_data_500', 'mixed_good_400_500', 'middle_300_400', 'bad_data_100_to_300', 'bad_data_under_100', 'random_no_train']
    for e in exp_names:
        path_data = f"/Users/manuelsenge/Documents/TUM/Semester 4/Robotic_Seminar/workspace/DQfD/data/{e}/scores.txt"
        with open(f'{path_data}', 'r') as f:
            for line in f.readlines():
                task2return[e].append(float(line.replace('\n', '')))

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))  # 2 rows, 1 column
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

    for i, (task, rewards) in enumerate(task2return.items()):
        if task != 'random_no_train':
            bins = 20
            heights, bins_edges, _ = axes[0].hist(rewards, bins=bins, alpha=0.5, label=task, color=colors[i])

    axes[0].set_xlabel("Reward")
    axes[0].set_ylabel("Frequency")
    axes[0].legend(loc='upper center', ncol=3)

    rewards = task2return['random_no_train']
    bins = 20
    heights, bins_edges, _ = axes[1].hist(rewards, bins=bins, alpha=0.5, label='random_no_train', color='brown')

    axes[1].set_xlabel("Reward")
    axes[1].set_ylabel("Frequency")
    axes[1].legend(loc='upper right')

    fig.suptitle("Different Dataset Distributions for DQfD training.", fontsize=16)
    plt.tight_layout()
    plt.show()
