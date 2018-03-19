##########################################
# Stat232A&CS266A Project 3:
# Solving CartPole with Deep Q-Network
# Author: Feng Gao
##########################################

import argparse
import gym
import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

env = gym.make('CartPole-v0').unwrapped


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        ###################################################################
        # Image input network architecture and forward propagation. Dimension
        # of output layer should match the number of actions.
        ###################################################################
        # Define your network structure here:
        # self.conv_block = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=5, stride=2),
        #     nn.BatchNorm2d(16),
        #     nn.Conv2d(16, 32, kernel_size=5, stride=2),
        #     nn.BatchNorm2d(32),
        #     nn.Conv2d(32, 32, kernel_size=5, stride=2),
        #     nn.BatchNorm2d(32)
        # )
        # self.fc_block = nn.Sequential(
        #     nn.Linear(448, 2)
        # )
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc = nn.Linear(384, 2)

    # Define your forward propagation function here:
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)

        ###################################################################
        # State vector input network architecture and forward propagation.
        # Dimension of output layer should match the number of actions.
        ##################################################################
        # Define your network structure here (no need to have conv
        # block for state input):

        # Define your forward propagation function here:
        # def forward(self, x):
        # pass


class DQNagent():
    def __init__(self):
        self.model = DQN()
        self.memory = deque(maxlen=args.memory_size)
        self.gamma = 0.999
        self.epsilon_start = 0.9
        self.epsilon_min = 0.05
        self.epsilon_decay = 200

        self.steps_done = 0
        # self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=5e-3)
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
            return self.model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
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
        # sample batch
        transitions = random.sample(self.memory, batch_size)
        Transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward'))
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))  # Return True or False

        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)
        state_batch = Variable(torch.cat(batch.state))
        reward_batch = Variable(torch.cat(batch.reward))
        action_batch = Variable(torch.cat(batch.action))

        # feed network
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.model(state_batch).gather(1, action_batch)  # [Q(s,a)] with size = batch_size x 1

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(batch_size).type(Tensor))
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0]  # V(s_t+1) = max_a Q(s_t+1, a)
        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        next_state_values.volatile = False
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch  # Target

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

def getCartLocation():
    world_width = env.x_threshold * 2
    scale = 600 / world_width
    return int(env.state[0] * scale + 600 / 2.0)


def getGymScreen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = getCartLocation()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (600 - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)
    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.FloatTensor(screen)
    return resize(screen).unsqueeze(0)


def plot_durations(durations):
    durations_t = FloatTensor(durations)
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 30:
        means = durations_t.unfold(0, 30, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(29), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())


def main():
    env._max_episode_steps = args.max_step
    print('env max steps:{}'.format(env._max_episode_steps))
    agent = DQNagent()
    durations = []
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
    for epoch in range(1, args.epochs + 1):
        ################################################################
        # Image input. We recommend to use the difference between two
        # images of current_screen and last_screen as input image.
        ################################################################
        env.reset()
        last_screen = getGymScreen()
        current_screen = getGymScreen()
        state = current_screen - last_screen

        for t in range(args.max_step):
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

            if done or t+1 == args.max_step:
                durations.append(t + 1)
                plot_durations(durations)
                break

    # plot_durations(durations)
    print('Complete')
    env.render()
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
