import wandb
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.util.util import make_vec_env
from stable_baselines3 import SAC
from sb3_contrib import TRPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
from eval_callback import EvalCallback
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
import random
import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed
from process_data import load_and_sort_rollouts

if __name__ == "__main__":
    train_steps =  3_000_000 #for SAC: 575_000 ajdust depending on algorithm (SAC need longer but less steps) TRPO is faster but needs more steps
    folder_path = "/Users/manuelsenge/Documents/TUM/Semester 4/Robotic_Seminar/workspace/GAIL/"
    range_labels = ["neg_inf_to_0", "0_to_1000", "1000_to_3000", "3000_to_5000", "5000_to_6000", "6000_to_7000"]
    use_algo = 'TRPO'
    for seed in [1234, 2345, 3456, 4567, 5678]:
        for exp_name in range_labels:
            set_random_seed(seed)

            env_name = "HalfCheetah-v4"
            run = wandb.init(
                name=f"GAIL_{use_algo}_training_{exp_name}_{env_name}_{seed}",
                project="robotic_seminar",
                group="train_gail",
                entity="manuelsenge",
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
            )

            
            rng = np.random.default_rng(seed)
            env = make_vec_env(env_name, rng=rng)
            env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
            env.seed(seed)
            eval_env = make_vec_env(env_name, rng=rng)
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
            eval_env.seed(seed)
            
            rollouts = load_and_sort_rollouts([f'{folder_path}/models/GAIL/final_data/final_rollouts_{exp_name}.pkl'], sort=False)
            transitions = rollout.flatten_trajectories(rollouts)

            reward_net = BasicRewardNet(
                observation_space=env.observation_space,
                action_space=env.action_space,
                normalize_input_layer=RunningNorm,
            )

            if use_algo == 'SAC':
                gen_algo = SAC(
                    policy="MlpPolicy",
                    env=env,
                    seed=seed,
                    batch_size=256,
                    learning_rate=3e-4,
                    buffer_size=1_000_000,
                    learning_starts=10_000,
                    ent_coef="auto",
                    train_freq=1,
                    gradient_steps=1,
                    tau=0.005,
                    gamma=0.99,
                )
            else:
                gen_algo = TRPO(
                    policy="MlpPolicy",
                    env=env,
                    seed=seed
                )

            gail_trainer = GAIL(
                demonstrations=rollouts,
                demo_batch_size=1024,
                gen_replay_buffer_capacity=512,
                n_disc_updates_per_round=8,
                venv=env,
                gen_algo=gen_algo,
                reward_net=reward_net,
                allow_variable_horizon=True,
            )

            print("Evaluating the untrained GAIL policy.")
            reward, _ = evaluate_policy(
                gail_trainer.gen_algo.policy,
                env,
                n_eval_episodes=5,
                render=False,
            )
            print(f"Reward before training: {reward}")

            print("Training GAIL policy...")
            eval_callback = EvalCallback(eval_env, eval_freq=20_000, n_eval_episodes=5, wandb_run=run)
            for timestep in range(0, train_steps, 20_000):
                gail_trainer.train(total_timesteps=20_000)
                reward, _ = evaluate_policy(
                    gail_trainer.gen_algo.policy,
                    env,
                    n_eval_episodes=5,
                    render=False,
                )
                print(f"Reward after {timestep} steps of training: {reward}")
                wandb.log({'train/timestep':timestep, 'train/reward':reward})

            gail_model_path = f"{folder_path}models/GAIL/models/gail_{use_algo}_{env_name}_{exp_name}_{seed}"
            gail_trainer.gen_algo.save(gail_model_path)

            #gail_trainer.save(gail_model_path) # uncomment if model needs to be saved

            print("Evaluating the trained GAIL policy.")
            reward, _ = evaluate_policy(
                gail_trainer.gen_algo.policy,
                env,
                n_eval_episodes=5,
                render=False,
            )
            print(f"Final Reward after training: {reward}")
            wandb.log({"final_reward": reward})

            run.finish()
