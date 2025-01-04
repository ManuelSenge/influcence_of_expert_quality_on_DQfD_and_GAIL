
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import wandb

class EvalCallback(BaseCallback):
    """
    Custom callback for evaluating an agent's performance during training.

    Attributes:
        eval_env: The environment to evaluate the agent on.
        eval_freq: How often (in timesteps) to evaluate the agent.
        n_eval_episodes: Number of episodes to run during each evaluation.
        wandb_run: WandB run object for logging.
    """

    def __init__(self, eval_env, eval_freq=10_000, n_eval_episodes=5, wandb_run=None, verbose=0):
        super(EvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.wandb_run = wandb_run

    def _on_step(self) -> bool:
        # Check if it's time to evaluate
        if self.n_calls % self.eval_freq == 0:
            # Evaluate the policy
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes
            )

            # Log to WandB
            if self.wandb_run is not None:
                wandb.log({
                    "eval/mean_reward": mean_reward,
                    "eval/std_reward": std_reward,
                    "eval/timesteps": self.num_timesteps,
                })
            print('--------------')
            print('Evaluation:')
            print("timesteps:", self.num_timesteps)
            print("mean_reward:", mean_reward)
            print("std_reward:", std_reward)
            print('--------------')

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}: mean_reward={mean_reward:.2f} ± {std_reward:.2f}")

        return True