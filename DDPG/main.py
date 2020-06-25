import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from collections import namedtuple, deque
import random
import gym
import matplotlib.pyplot as plt
from ddpg_full import Agent

#env = gym.make("Pendulum-v0")
#state_dim = env.observation_space.shape[0]
#action_dim = env.action_space.shape[0]
#print(state_dim, action_dim)


def Main(n_episodes=5000, max_t=500, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    avg_rewards = []
    episode_durations = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done or t >= max_t - 1:
                #agent.writer.add_scalar('live/finish_step', t+1, global_step=i_episode)
                break
        scores_deque.append(score)
        scores.append(score)
        avg_rewards.append(np.mean(scores[-10:]))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))


    return scores, avg_rewards

env = gym.make('Pendulum-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

env.seed(2)
agent = Agent(state_size=state_dim, action_size=action_dim, random_seed=2)
scores, avg_rewards = Main()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(scores)
plt.plot(avg_rewards)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()



# Helper function to plot the results while training
######################################
'''
def plot_duration(scores, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(scores)

    moving_avg = get_moving_average(moving_avg_period, scores)
    plt.plot(moving_avg)
    plt.pause(0.001)
    print("Episode", len(scores), "\n", \
        moving_avg_period, "episode moving avg:", moving_avg[-1])
    if is_ipython: display.clear_output(wait=True)

def get_moving_average(period, scores):
    scores = torch.tensor(scores, dtype=torch.float)
    if len(scores) >= period:
        moving_avg = scores.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(scores))
        return moving_avg.numpy()
'''
############################################
