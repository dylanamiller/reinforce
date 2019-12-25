import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.distributions import Categorical

import numpy as np
import matplotlib.pyplot as plt

from statistics import stdev, mean

import gym


dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

torch.manual_seed(0)


class VPG(nn.Module):
    def __init__(self, obs_dim, n_action, hidden_dim=128):
        super(VPG, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.6)
        self.out = nn.Linear(hidden_dim, n_action)

    def forward(self, x):
        x = self.l1(x)
        x = self.dropout(x)
        x = F.relu(x)

        action_scores = self.out(x)
        return F.softmax(action_scores, dim=1)


def plot(rewards):
    plt.plot(rewards)
    plt.show()


def update(model, log_probs, rewards, gamma=0.9, 
            eps=np.finfo(np.float32).eps.item()):
    returns = []
    R = 0

    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)

    # torch.cat retains the gradient_fcn from log probs
    # if recasted as new torch.tensor, gradient_fcn goes away
    loss = torch.cat([-lp * ((r-mean(returns))/(stdev(returns)+eps)) 
                        for lp, r in zip(log_probs, returns)]
                        ).sum()

    model['opt'].zero_grad()
    loss.backward()
    model['opt'].step()


def select_action(action_probs):
    m = Categorical(action_probs)
    action = m.sample()

    return action.item(), m.log_prob(action)


def train(env, model, n_episodes=200, max_timesteps=10000):
    print('Start Training...')

    avg_returns = []

    for episode in range(1, n_episodes+1):
        obs = env.reset()

        log_probs, rewards = [], []

        for ts in range(max_timesteps):
            obs = torch.from_numpy(obs).float().unsqueeze(0)
            action_probs = model['net'](obs)
            action, log_prob = select_action(action_probs)
            obs, reward, done, _ = env.step(action)
                
            log_probs.append(log_prob)
            rewards.append(reward)
            if done:
                break

        avg_returns.append(sum(rewards))
        update(model, log_probs, rewards)

        if episode % 10 == 0:
            print('Episode: {} - Episode Return: {:.2f} - Average Return: {:.2f}'.format(
                  episode, sum(rewards), mean(avg_returns)))

    plot(avg_returns)
    

def main(environment='CartPole-v1', episodes=200, max_timesteps=10000):
    env = gym.make(environment)
    env.seed(0)

    obs_dim = env.observation_space.shape[0]
    n_action = env.action_space.n

    model = {}

    model['net'] = VPG(obs_dim, n_action)
    model['opt'] = optim.Adam(model['net'].parameters(), lr=1e-2)

    train(env, model, episodes, max_timesteps)


if __name__ == '__main__':
    main()