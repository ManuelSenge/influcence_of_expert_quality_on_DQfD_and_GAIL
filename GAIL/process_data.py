import pickle
import numpy as np
import matplotlib.pyplot as plt
from imitation.data.types import TrajectoryWithRew
from imitation.data import rollout
from tqdm import tqdm
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
from imitation.util.util import make_vec_env


def generate_data(save_folder, model_folder, model_name, seed=1234, min_episodes=200):    
    rng = np.random.default_rng(seed)
    env = make_vec_env(env_name, rng=rng)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    expert = SAC.load(model_folder+model_name, env=env)

    print("Generating expert rollouts...")
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=min_episodes),
        rng=rng,
    )
    f = open(f'{save_folder}{exp_name}_rollout_{seed}.txt', 'w')
    f.write(str([r.rews.sum() for r in rollouts]))
    f.close()
    print('The Data has the following returns:', [r.rews.sum() for r in rollouts])

    return rollouts

def save_rollouts(rollouts, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(rollouts, f)

def load_and_sort_rollouts(file_paths, sort=True):
    all_rollouts = []
    for file_path in file_paths:
        with open(file_path, "rb") as f:
            rollouts = pickle.load(f)
            all_rollouts.extend(rollouts)
    if sort:
        sorted_rollouts = sorted(all_rollouts, key=lambda r: np.sum(r.rews), reverse=True)        
        return sorted_rollouts
    
    return all_rollouts

def create_transitions(rollouts):
    return rollout.flatten_trajectories(rollouts)

def plot_reward_histogram(rollouts, output_path=None):
    rewards = [np.sum(r.rews) for r in rollouts]
    plt.figure()
    plt.hist(rewards, bins=20, edgecolor="k", alpha=0.7)
    plt.xlabel("Episode Rewards")
    plt.ylabel("Frequency")
    plt.title("Histogram of Episode Rewards")
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

if __name__ == "__main__":
    env_name = "HalfCheetah-v4"
    folder_path = "/Users/manuelsenge/Documents/TUM/Semester 4/Robotic_Seminar/workspace/GAIL/models/GAIL/"
    model_folder_path = "/Users/manuelsenge/Documents/TUM/Semester 4/Robotic_Seminar/workspace/GAIL/models/SAC/"
    exp_names = ['perfect_policy','random_policy', 'bad_2000_rew', 'middle_6000_rew', 'ok_8000_reward'] 
    paths = []
    min_episodes = 200
    for exp_name in tqdm(exp_names):
        rollouts = generate_data(save_folder=folder_path, model_folder=model_folder_path, model_name=f'{env_name}_{exp_name}', min_episodes=min_episodes)
        file_path = f"{folder_path}{env_name}_{exp_name}_rollouts.pkl"
        save_rollouts(rollouts, file_path)
        paths.append(file_path)

    sorted_rollouts = load_and_sort_rollouts(paths)

    histogram_path = f"{folder_path}{env_name}_{exp_name}_reward_histogram.png"
    plot_reward_histogram(sorted_rollouts, output_path=histogram_path)
