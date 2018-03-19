##########################################
# Stat232A&CS266A Project 3:
# Solving CartPole with Deep Q-Network
# Author: Feng Gao
##########################################

import gym
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


import argparse
import random
import math
import numpy as np
from collections import deque, namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T


FloatTensor = torch.FloatTensor
ByteTensor = torch.ByteTensor
LongTensor = torch.LongTensor
Tensor = FloatTensor

parser = argparse.ArgumentParser(description='DQN_AGENT')
parser.add_argument('--epochs', type=int, default=200, metavar='E',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='batch size for training (default: 32)')
parser.add_argument('--memory-size', type=int, default=10000, metavar='M',
                    help='memory length (default: 10000)')
parser.add_argument('--max-step', type=int, default=250,
                    help='max steps allowed in gym (default: 250)')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
###################################################################
# Image input network architecture and forward propagation. Dimension
# of output layer should match the number of actions.
###################################################################
        # Define your network structure here:
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)
        # Define your forward propagation function here:
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
###################################################################
# State vector input network architecture and forward propagation.
# Dimension of output layer should match the number of actions.
##################################################################
        # Define your network structure here (no need to have conv
        # block for state input):

class DQNagent():
    def __init__(self):
        self.model = DQN()
        self.memory = deque(maxlen=args.memory_size)
        self.gamma = 0.8
        self.epsilon_start = 1
        self.epsilon_min = 0.05
        self.epsilon_decay = 200

        self.steps_done = 0
        self.optimizer = optim.RMSprop(self.model.parameters())
###################################################################
# remember() function
# remember function is for the agent to get "experience". Such experience
# should be storaged in agent's memory. The memory will be used to train
# the network. The training example is the transition: (state, action,
# next_state, reward). There is no return in this function, instead,
# you need to keep pushing transition into agent's memory. For your
# convenience, agent's memory buffer is defined as deque.
###################################################################
    def remember(self, state, action, next_state, reward):
        # one_hot_action = np.zeros(2)
        # one_hot_action[action] = 1
        # one_hot_action = LongTensor(one_hot_action.reshape((1, 2)))
        self.memory.append((state, action, reward, next_state))

        if len(self.memory) > args.memory_size:
          self.memory.popleft()

        if len(self.memory) > args.batch_size:
          self.replay(args.batch_size)

###################################################################
# act() fucntion
# This function is for the agent to act on environment while training.
# You need to integrate epsilon-greedy in it. Please note that as training
# goes on, epsilon should decay but not equal to zero. We recommend to
# use the following decay function:
# epsilon = epsilon_min+(epsilon_start-epsilon_min)*exp(-1*global_step/epsilon_decay)
# act() function should return an action according to epsilon greedy.
# Action is index of largest Q-value with probability (1-epsilon) and
# random number in [0,1] with probability epsilon.
###################################################################
    def act(self, state):
        sample = random.random()
        eps_threshold = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            return self.model(
                Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
        else:
            return torch.LongTensor([[random.randrange(2)]])
###################################################################
# replay() function
# This function performs an one step replay optimization. It first
# samples a batch from agent's memory. Then it feeds the batch into
# the network. After that, you will need to implement Q-Learning.
# The target Q-value of Q-Learning is Q(s,a) = r + gamma*max_{a'}Q(s',a').
# The loss function is distance between target Q-value and current
# Q-value. We recommend to use F.smooth_l1_loss to define the distance.
# There is no return of act() function.
# Please be noted that parameters in Q(s', a') should not be updated.
# You may use Variable().detach() to detach Q-values of next state
# from the current graph.
###################################################################
    def replay(self, batch_size):
            # Q-learning
            # loss smooth l1 loss
            # Q(s', a') Variable().detach()
            # one step replay optimization

        # sample batch
        transitions = random.sample(self.memory, batch_size)
        Transition = namedtuple('Transition',
                ('state', 'action', 'next_state', 'reward'))
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)))

        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                    if s is not None]),
                                         volatile=True)
        state_batch = Variable(torch.cat(batch.state))
        reward_batch = Variable(torch.cat(batch.reward))
        action_batch = Variable(torch.cat(batch.action))

        # feed network
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(batch_size).type(Tensor))
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0]
        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        next_state_values.volatile = False
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
#################################################################
# Functions 'getCartLocation' and 'getGymScreen' are designed for
# capturing current renderred image in gym. You can directly take
# the return of 'getGymScreen' function, which is a resized image
# with size of 3*40*80.
#################################################################

def getCartLocation2(env):
    world_width = env.x_threshold*2
    scale = 600/world_width
    return int(env.state[0]*scale+600/2.0)

def getGymScreen2(env):
    screen = env.render(mode='rgb_array').transpose((2,0,1))
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = getCartLocation(env)
    if cart_location < view_width//2:
        slice_range = slice(view_width)
    elif cart_location > (600-view_width//2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width//2, cart_location+view_width//2)
    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32)/255
    screen = torch.FloatTensor(screen)
    return resize(screen).unsqueeze(0)

env = gym.make('CartPole-v0').unwrapped
screen_width = 600
# def get_cart_location():
def getCartLocation():
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

# def get_screen():
def getGymScreen():
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
    # Strip off the top and bottom of the screen
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = getCartLocation()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).type(Tensor)

def ppp():
    x1 = np.linspace(0.0, 5.0)
    x2 = np.linspace(0.0, 2.0)

    y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
    y2 = np.cos(2 * np.pi * x2)

    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('A tale of 2 subplots')
    plt.ylabel('Damped oscillation')

    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('time (s)')
    plt.ylabel('Undamped')

    plt.show()

def plot_durations(durations):
    durations_t = FloatTensor(durations)
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def main():
    # env = gym.make('CartPole-v0').unwrapped
    env._max_episode_steps = args.max_step
    print('env max steps:{}'.format(env._max_episode_steps))
    steps_done = 0
    agent = DQNagent()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, agent.model.parameters()), lr=1e-3)
    durations = []

    plt.ion()
    ################################################################
    # training loop
    # You need to implement the training loop here. In each epoch,
    # play the game until trial ends. At each step in one epoch, agent
    # need to remember the transitions in self.memory and perform
    # one step replay optimization. Use the following function to
    # interact with the environment:
    #   env.step(action)
        # It gives you infomation about next step after taking the action.
        # The return of env.step() is (next_state, reward, done, info). You
    # do not need to use 'info'. 'done=1' means current trial ends.
    # if done equals to 1, please use -1 to substitute the value of reward.
    ################################################################
    for epoch in range(1, args.epochs+1):
        steps = 0

    ################################################################
    # Image input. We recommend to use the difference between two
    # images of current_screen and last_screen as input image.
    ################################################################
        env.reset()

        # ppp()
        plot_durations(durations)

        last_screen = getGymScreen()
        current_screen = getGymScreen()
        state = current_screen - last_screen

        for t in count():
            # Select and perform an action
            action = agent.act(state)
            _, reward, done, _ = env.step(action[0, 0])
            reward = -1 if done else reward
            reward = Tensor([reward])

            # Observe new state
            last_screen = current_screen
            current_screen = getGymScreen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            # Perform one step of the optimization (on the target network)
            agent.remember(state, action, reward, next_state)

            # Move to the next state
            state = next_state

            if done:
                durations.append(t + 1)
                # plot_durations(durations)
                break

    print('Complete')
    env.render(close=True)
    env.close()
    plt.ioff()
    plt.show()
    ################################################################
    # State vector input. You can direct take observation from gym
    # as input of agent's DQN
    ################################################################
        # state = env.reset()

    ################################################################

if __name__ == "__main__":
    main()
