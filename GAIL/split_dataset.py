import pickle
import numpy as np
import matplotlib.pyplot as plt
from imitation.data.types import TrajectoryWithRew
from imitation.data import rollout
from tqdm import tqdm
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
from imitation.util.util import make_vec_env
import os
from process_data import save_rollouts, load_and_sort_rollouts


def split_rollouts_by_reward(sorted_rollouts, output_dir):
    reward_ranges = [(-np.inf, 0), (0, 1000), (1000, 3000), (3000, 5000), (5000, 6000), (6000, 7000)]
    range_labels = ["neg_inf_to_0", "0_to_1000", "1000_to_3000", "3000_to_5000", "5000_to_6000", "6000_to_7000"]

    os.makedirs(output_dir, exist_ok=True)
    rollouts_count = {}

    for (low, high), label in zip(reward_ranges, range_labels):
        filtered_rollouts = [r for r in sorted_rollouts if low < np.sum(r.rews) <= high]
        output_path = os.path.join(output_dir, f"final_rollouts_{label}.pkl")
        save_rollouts(filtered_rollouts, output_path)
        rollouts_count[label] = len(filtered_rollouts)

    return rollouts_count


if __name__ == '__main__':
    main_path = '/Users/manuelsenge/Documents/TUM/Semester 4/Robotic_Seminar/workspace/GAIL/models/GAIL/'
    env_name = "HalfCheetah-v4"
    exp_names = ['perfect_policy','random_policy', 'bad_2000_rew', 'middle_6000_rew', 'ok_8000_reward'] 
    data_paths = [f'{main_path}{env_name}_{data_sample_name}_rollouts.pkl' for data_sample_name in exp_names]
    sorted_rollouts = load_and_sort_rollouts(data_paths)

    nums = split_rollouts_by_reward(sorted_rollouts, main_path)
    print(nums)