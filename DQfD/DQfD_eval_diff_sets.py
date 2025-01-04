# -*- coding: utf-8 -*
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from gym import wrappers
import gym
import numpy as np
import pickle
from Config import Config, DDQNConfig, DQfDConfig
from DQfD_V3 import DQfD
from DQfDDDQN import DQfDDDQN
from collections import deque
import itertools
import wandb


def map_scores(dqfd_scores=None, ddqn_scores=None, xlabel=None, ylabel=None):
    if dqfd_scores is not None:
        plt.plot(dqfd_scores, 'r')
    if ddqn_scores is not None:
        plt.plot(ddqn_scores, 'b')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()


def run_DDQN(index, env, use_wandb):
    with tf.name_scope('DDQN_' + str(index)):
        agent = DQfDDDQN(env, DDQNConfig())
    scores = []
    for e in range(Config.episode):
        done = False
        score = 0  # sum of reward in one episode
        state = env.reset()
        while done is False:
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            score += reward
            reward = reward if not done or score == 499 else -100
            agent.perceive([state, action, reward, next_state, done, 0.0])  # 0. means it is not a demo data
            agent.train_Q_network(update=False)
            state = next_state
        if done:
            scores.append(score)
            agent.sess.run(agent.update_target_net)
            if use_wandb:
                wandb.log({f"DDPG_{i}/episode": e, f"DDPG_{i}/score": score, f"DDPG_{i}/demo_buffer_len": len(agent.demo_buffer),
                    f"DDPG_{i}/memory length": len(agent.replay_buffer), f"DDPG_{i}/epsilon": agent.epsilon})
            
            print(f"DDPG_{i}/episode:", e, f"  DDPG_{i}/score:", score, f"  DDPG_{i}/demo_buffer:", len(agent.demo_buffer),
                  f"  DDPG_{i}/memory length:", len(agent.replay_buffer), f"  DDPG_{i}/epsilon:", agent.epsilon)
            # if np.mean(scores[-min(10, len(scores)):]) > 490:
            #     break
    agent.sess.close()
    return scores


def run_DQfD(index, env, use_wandb, path_data):
    with open(path_data+'/demo.p', 'rb') as f:
        demo_transitions = pickle.load(f)
        demo_transitions = deque(itertools.islice(demo_transitions, 0, Config.demo_buffer_size))
        assert len(demo_transitions) == Config.demo_buffer_size
    with tf.name_scope('DQfD_' + str(index)):
        agent = DQfD(env, DQfDConfig(), demo_transitions=demo_transitions)

    agent.pre_train()  # use the demo data to pre-train network
    scores, e, replay_full_episode = [], 0, None
    while True:
        done, score, n_step_reward, state = False, 0, None, env.reset()
        t_q = deque(maxlen=Config.trajectory_n)
        while done is False:
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            score += reward
            reward = reward if not done or score == 499 else -100
            reward_to_sub = 0. if len(t_q) < t_q.maxlen else t_q[0][2]  # record the earliest reward for the sub
            t_q.append([state, action, reward, next_state, done, 0.0])
            if len(t_q) == t_q.maxlen:
                if n_step_reward is None:  # only compute once when t_q first filled
                    n_step_reward = sum([t[2]*Config.GAMMA**i for i, t in enumerate(t_q)])
                else:
                    n_step_reward = (n_step_reward - reward_to_sub) / Config.GAMMA
                    n_step_reward += reward*Config.GAMMA**(Config.trajectory_n-1)
                t_q[0].extend([n_step_reward, next_state, done, t_q.maxlen])  # actual_n is max_len here
                agent.perceive(t_q[0])  # perceive when a transition is completed
                if agent.replay_memory.full():
                    agent.train_Q_network(update=False)  # train along with generation
                    replay_full_episode = replay_full_episode or e
            state = next_state
        if done:
            # handle transitions left in t_q
            t_q.popleft()  # first transition's n-step is already set
            transitions = set_n_step(t_q, Config.trajectory_n)
            for t in transitions:
                agent.perceive(t)
                if agent.replay_memory.full():
                    agent.train_Q_network(update=False)
                    replay_full_episode = replay_full_episode or e
            if agent.replay_memory.full():
                scores.append(score)
                agent.sess.run(agent.update_target_net)
            if replay_full_episode is not None:
                if use_wandb:
                    wandb.log({f"DQfD_{index}/episode": e, f"DQfD_{index}/trained-episode": e-replay_full_episode,
                               f"DQfD_{index}/score": score, f"DQfD_{index}/memory length": len(agent.replay_memory),
                               f"DQfD_{index}/epsilon": agent.epsilon})
                
                print(f"DQfD_{index}/episode: {e}  DQfD_{index}/trained-episode: {e-replay_full_episode}  DQfD_{index}/score: {score}  DQfD_{index}/memory length: { len(agent.replay_memory)}  DQfD_{index}/epsilon: {agent.epsilon}")            # if np.mean(scores[-min(10, len(scores)):]) > 495:
            #     break
            # agent.save_model()
        if len(scores) >= Config.episode:
            break
        e += 1
    agent.sess.close()
    return scores


# extend [n_step_reward, n_step_away_state] for transitions in demo
def set_n_step(container, n):
    t_list = list(container)
    # accumulated reward of first (trajectory_n-1) transitions
    n_step_reward = sum([t[2] * Config.GAMMA**i for i, t in enumerate(t_list[0:min(len(t_list), n) - 1])])
    for begin in range(len(t_list)):
        end = min(len(t_list) - 1, begin + Config.trajectory_n - 1)
        n_step_reward += t_list[end][2]*Config.GAMMA**(end-begin)
        # extend[n_reward, n_next_s, n_done, actual_n]
        t_list[begin].extend([n_step_reward, t_list[end][3], t_list[end][4], end-begin+1])
        n_step_reward = (n_step_reward - t_list[begin][2])/Config.GAMMA
    return t_list


def get_demo_data(env, use_wandb, lower_goal_score=500, higher_goal_score=500, demo_data_path=Config.DEMO_DATA_PATH, run_sub_optimal=False):
    """
    env: The env used
    use_wandb:whether to log on wandb
    lower_goal_score: scores higher than that reward should be considered in buffer
    higher_goal_score: scores higher than that reward should be considered in buffer
    demo_data_path: path to store demo data
    run_sub_optimal: if we want a suboptimal demo data, we want to train the policy until it reached 4 times the desired episode reward, then we want to stop training it
    """
    # env = wrappers.Monitor(env, '/tmp/CartPole-v0', force=True)
    # agent.restore_model()
    with tf.name_scope('get_demo_data'):#tf.variable_scope('get_demo_data'):
        agent = DQfDDDQN(env, DDQNConfig())

    e = 0
    while True:
        done = False
        score = 0  # sum of reward in one episode
        state = env.reset()
        demo = []
        stop_training=0
        while done is False:
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            score += reward
            reward = reward if not done or score == 499 else -100
            agent.perceive([state, action, reward, next_state, done, 0.0])  # 0. means it is not a demo data
            demo.append([state, action, reward, next_state, done, 1.0])  # record the data that could be expert-data
            if stop_training < 4:
                agent.train_Q_network(update=False)
            state = next_state
        if done:
            if score >= lower_goal_score and score <= higher_goal_score:  # 500 expert demo data, we want data that is in between two ranges
                if run_sub_optimal and stop_training < 4:
                    stop_training += 1
                demo = set_n_step(demo, Config.trajectory_n)
                agent.demo_buffer.extend(demo)
            agent.sess.run(agent.update_target_net)
            if use_wandb:
                wandb.log({"demo/episode": e, "demo/score":score, "demo/demo_buffer_len":len(agent.demo_buffer),
                  "demo/memory length":len(agent.replay_buffer), "demo/epsilon":agent.epsilon})

            print("demo/episode:", e, "  demo/score:", score, "  demo/demo_buffer:", len(agent.demo_buffer),
                  "  demo/memory length:", len(agent.replay_buffer), "  demo/epsilon:", agent.epsilon)
            if len(agent.demo_buffer) >= Config.demo_buffer_size:
                agent.demo_buffer = deque(itertools.islice(agent.demo_buffer, 0, Config.demo_buffer_size))
                break
        e += 1

    with open(demo_data_path, 'wb') as f:
        pickle.dump(agent.demo_buffer, f, protocol=2)
    agent.sess.close()


if __name__ == '__main__':
    only_gather_data = True
    use_wandb = True
    env = gym.make(Config.ENV_NAME)
    exp_names = ['perfect_data_500', 'mixed_good_400_500', 'middle_300_400', 'bad_data_100_to_300', 'bad_data_under_100', 'random_no_train']

    # Load data into task2return
    for e in exp_names:
        if use_wandb:
            run = wandb.init(
                            name=f"DQfD_{e}",
                            project="robotic_seminar",
                            group="DQfD",
                            entity="manuelsenge",
                            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                            monitor_gym=True,  # auto-upload the videos of agents playing the game
                            save_code=True,  # optional
                        )
        path_data = f"/Users/manuelsenge/Documents/TUM/Semester 4/Robotic_Seminar/workspace/DQfD_new/DQfD/data/{e}"
        # ----------------------------- get DQfD scores --------------------------------
        dqfd_sum_scores = np.zeros(Config.episode)
        for i in range(Config.iteration):
            scores = run_DQfD(i, env, use_wandb, path_data)
            dqfd_sum_scores = np.array([a + b for a, b in zip(scores, dqfd_sum_scores)])
        dqfd_mean_scores = dqfd_sum_scores / Config.iteration
        with open(f'{path_data}/dqfd_mean_scores.p', 'wb') as f:
            pickle.dump(dqfd_mean_scores, f, protocol=2)
        wandb.finish()
        env.close()


