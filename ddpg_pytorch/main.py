"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
from env import ArmEnv
from rl import DDPG
import time as t
import numpy as np
import matplotlib.pyplot as plt



MAX_EPISODES = 500
MAX_EP_STEPS = 200
ON_TRAIN = True


# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(s_dim,a_dim, a_bound)

steps = []
ep_rewards = []
avg_rewards = []

def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            env.render()

            #a=env.sample_action()
            a = rl.choose_action(s)

            if np.isnan(a[0]):
                a[0]=0
            if np.isnan(a[1]):
                a[1]=0

            s_, r, done = env.step(a)
            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                ep_rewards.append(ep_r)
                avg_reward = np.average(ep_rewards[-100:])
                avg_rewards.append(avg_reward)
                break
    rl.save()
    Ep_ = range(MAX_EPISODES)
    plt.plot(Ep_, ep_rewards )
    plt.show()



    fig, ax = plt.subplots()
    t = np.arange(MAX_EPISODES)
    ax.plot(t, ep_rewards, label="Total Reward")
    ax.plot(t, avg_rewards, label="Average Reward")
    ax.set_title("Reward vs Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    plt.show()


def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    while True:
        s = env.reset()
        for _ in range(200):
            env.render()
            a = rl.choose_action(s)
            s, r, done = env.step(a)
            if done:
                break


if ON_TRAIN:
    train()
else:
    eval()



