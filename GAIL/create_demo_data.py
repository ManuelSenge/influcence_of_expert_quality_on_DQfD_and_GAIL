import os
import wandb
from stable_baselines3 import SAC
from imitation.util.util import make_vec_env
import numpy as np
from eval_callback import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.logger import configure
from sb3_contrib import TRPO
from stable_baselines3.common.utils import set_random_seed


env_name = "HalfCheetah-v4"
algo = 'TRPO'
different_steps = [0, 200000, 500000, 1_500_000, 3_000_000]
exp_name = ['random_policy', 'bad_2000_rew', 'middle_6000_rew', 'ok_8000_reward', 'perfect_policy']
for train_steps, exp_name in zip(different_steps, exp_name):
    for seed in [1234, 2345, 3456, 4567, 5678]:
        set_random_seed(seed)
        run = wandb.init(
            name=f"GAIL_demo_sac_{env_name}_{train_steps}_{exp_name}_steps",
            project="robotic_seminar",
            group="train_sac_GAIL_demo_data",
            entity="manuelsenge",
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )

        rng = np.random.default_rng(1234)
        env = make_vec_env(env_name, rng=rng)
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        eval_env = make_vec_env(env_name, rng=rng)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        if algo =='SAC':
            # Train SAC with Zoo Hyperparameters
            model = SAC(
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
            model = TRPO(
                policy="MlpPolicy",
                env=env,
                seed=seed
            )

        model.learn(
            total_timesteps=train_steps,
            callback=EvalCallback(eval_env, eval_freq=10_000, n_eval_episodes=5, wandb_run=run),
            log_interval=10
        )

        # Save the trained model if necessary
        '''os.makedirs("models", exist_ok=True)
        model_path = f"models/{env_name}_{exp_name}_TRPO"
        model.save(model_path)
        print(f"Model saved to {model_path}")'''

        run.finish()
