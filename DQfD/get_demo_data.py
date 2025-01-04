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


def get_demo_data(env, use_wandb, lower_goal_score=500, higher_goal_score=500, demo_data_path=Config.DEMO_DATA_PATH, run_sub_optimal=False, stop_after=-1):
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
    scores = []
    e = 0
    while True:
        done = False
        score = 0  # sum of reward in one episode
        state = env.reset()
        demo = []
        stop_training=0
        stop_after = stop_after
        while done is False:
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            score += reward
            reward = reward if not done or score == 499 else -100
            agent.perceive([state, action, reward, next_state, done, 0.0])  # 0. means it is not a demo data
            demo.append([state, action, reward, next_state, done, 1.0])  # record the data that could be expert-data
            if stop_training < stop_after+1:
                agent.train_Q_network(update=False)
                if run_sub_optimal:
                    stop_training += 1
            state = next_state
        if done:
            if score >= lower_goal_score and score <= higher_goal_score:  # 500 expert demo data, we want data that is in between two ranges
                scores.append(score)
                demo = set_n_step(demo, Config.trajectory_n)
                agent.demo_buffer.extend(demo)
            agent.sess.run(agent.update_target_net)
            if use_wandb:
                wandb.log({"demo/episode": e, "demo/score":score, "demo/demo_buffer_len":len(agent.demo_buffer),
                  "demo/memory length":len(agent.replay_buffer), "demo/epsilon":agent.epsilon, 'stop_training':stop_training})

            print("demo/episode:", e, "  demo/score:", score, "  demo/demo_buffer:", len(agent.demo_buffer),
                  "  demo/memory length:", len(agent.replay_buffer), "  demo/epsilon:", agent.epsilon)
            if len(agent.demo_buffer) >= Config.demo_buffer_size:
                agent.demo_buffer = deque(itertools.islice(agent.demo_buffer, 0, Config.demo_buffer_size))
                break
        e += 1

    with open(demo_data_path+f'/demo.p', 'wb') as f:
        pickle.dump(agent.demo_buffer, f, protocol=2)
    f = open(f'{demo_data_path}/scores.txt', 'w')
    for elem in scores:
        f.write(f'{elem}\n')
    f.close()
    agent.sess.close()


if __name__ == '__main__':
    use_wandb = True
    task2range = {'perfect_data_500':[500,500],
        'mixed_good_400_500':[400, 500],
        'middle_300_400':[300, 400],
        'bad_data_under_100':[0,100],
        'bad_data_100_to_300':[100, 300],
        'random_no_train':[0,500]}
    
    task2train_steps = {
        'perfect_data_500':2, # is disabled anyways
        'mixed_good_400_500':4*500,
        'middle_300_400':200,
        'bad_data_under_100':10,
        'bad_data_100_to_300':100,
        'random_no_train':-1}
    
    exp_names = ['random_no_train'] # 'perfect_data_500', 'mixed_good_400_500', 'middle_300_400, 'bad_data_100_to_300', 'bad_data_under_100', 

    env = gym.make(Config.ENV_NAME)
    for e in exp_names:
        if use_wandb:
            run = wandb.init(
                    name=f"2_demo_data_{e}",
                    project="robotic_seminar",
                    group="get_demo_data",
                    entity="manuelsenge",
                    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                    monitor_gym=True,  # auto-upload the videos of agents playing the game
                    save_code=True,  # optional
                )
    
        path_data = f"/Users/manuelsenge/Documents/TUM/Semester 4/Robotic_Seminar/workspace/DQfD_new/DQfD/data/{e}"
        if e == 'perfect_data_500':
            run_sub_optimal = False
        else:
            run_sub_optimal = True
        get_demo_data(env, use_wandb, lower_goal_score=task2range[e][0], higher_goal_score=task2range[e][1], demo_data_path=path_data, run_sub_optimal=run_sub_optimal, stop_after=task2train_steps[e])
        wandb.finish()
