""" Trains an agent to play either CartPole or Pong using Policy Gradients.

The CartPole case will run fast, but Pong may take about 20 minutes to observe
improvements, and that's assuming vectorized gradients.

This requires OpenAI gym and (for Pong) the accompanying atari library, which is
*not* included in the basic gym installation.
"""

import numpy as np
import cPickle as pickle
import gym
import sys
import time


class PolicyGradient(object):
    """ An agent which learns how to play CartPole or Pong using Policy Gradients.
    
    The default architecture uses a two-layer fully connected net (no biases)
    with ReLU non-linearity, and the sigmoid at the end. Specifically:

    (Pong pixels) - (FC layer) - (ReLU) - (FC layer) - (Sigmoid)

    The output is a single scalar in (0,1), indicating the probability of
    playing LEFT (for CartPole) or UP (for Pong). The only other actions are,
    respectively, RIGHT and DOWN. We assume that these are only actions we take,
    i.e., we ignore the NO-OP action. In addition, RMSProp is the gradient
    updating scheme.
    """

    def __init__(self, D=4, H=20, learning_rate=1e-2, batch_size=10, gamma=0.99, 
                 decay_rate=0.99, render=False):
        """ Initialize the Policy Gradient agent.

        Inputs:
        - D: The dimension of the input (for Pong, this is after downsampling).
        - H: The number of hidden layer neurons.
        - learning_rate: The learning rate for training.
        - batch_size: Run this many episodes before doing a parameter update.
        - gamma: The discount factor for rewards.
        - decay_rate: The decay factor for RMPSProp leaky sum of grad^2.
        - render: Whether you want to see the game in action or not; setting
          this as False means the code runs much faster.
        """
        self.D = D 
        self.H = H 
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.render = render

        # Set up weights, gradients, and stuff needed for RMSProp updates.
        self.model = {}
        self.model['W1'] = np.random.randn(H,D) / np.sqrt(D)
        self.model['W2'] = np.random.randn(H) / np.sqrt(H)
        self.grad_buffer = { k : np.zeros_like(v) for k,v in self.model.iteritems() }
        self.rmsprop_cache = { k : np.zeros_like(v) for k,v in self.model.iteritems() }

        # For saving/evaluating output and plotting.
        self.final_ep_rewards = []
        self.running_rewards = []


    def sigmoid(self, x): 
        """ Standard sigmoid, to make the output in (0,1). """
        return 1.0 / (1.0 + np.exp(-x))


    def preprocess(self, I):
        """ Preprocess Pong game frames into vectors.
        
        Input:
        - (210,160,3) uint8 frame representing Pong game screen.
        
        Returns:
        - Downsampled (DxD) matrix of 0s and 1s, "raveled" into a 1-D vector.
        """
        I = I[35:195]
        I = I[::2,::2,0]
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1
        return I.astype(np.float).ravel()


    def policy_forward(self, x):
        """ Given an input image, determine probability of going LEFT or UP.

        Input:
        - x: One input image, downsampled and in the form of a 1-D vector.
        
        Returns:
        - (p, h) where p is the probability of going LEFT (action 0) if
          CartPole, or probability of UP (action 2) if Pong. Also, h is the
          hidden state output.
        """
        h = np.dot(self.model['W1'], x)
        h[h<0] = 0
        output = np.dot(self.model['W2'], h)
        p = self.sigmoid(output)
        return p, h 


    def discount_rewards(self, r, do_reset=False):
        """ Compute discounted rewards from per-turn rewards from one episode.

        Note: in Pong, after any player earns a point, the game resets to when
        the ball/pong reappears.  Thus, it is recommended to reset the rewards
        to zero after any non-zero point.  In addition, the discount factor
        should be scaled to start at \gamma^0.
       
        Input:
        - r: A list representing the rewards obtained after *each timestep*,
          within *one* episode. The length of r thus depends on how long the
          episode lasted.
        - do_reset: A boolean to indicate whether we need to reset the rewards to
          zero. This should be False for CartPole, True for Pong.

        Returns:
        - A list of the same length as r, with each component representing
          a discounted sum of rewards, to be used later for Policy Gradients.
        """
        discounted_r = np.zeros_like(r)

        ########################################################################
        # TODO: Fill in the sum of discounted returns in the components of     #
        # discounted_r. These serve as coefficients for policy gradients.      #
        ########################################################################

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return discounted_r

    
    def policy_backward(self, ep_x, ep_h, ep_dprobs, discounted_epr):
        """ The backward pass, i.e. where policy gradients happen!

        Uses the stacked data, all from *one* episode of CartPole or Pong, to
        compute the gradient updates for W1 and W2. For shaping purposes, let T
        be the number of timesteps of the current CartPole or Pong episode.

        Inputs:
        - ep_x: Array of shape (T,D) representing observed states.
        - ep_h: Array of shape (T,H) representing hidden states after ReLUs.
        - ep_dprobs: Array of shape (T,1) representing the final gradients in
          the computational graph.
        - discounted_epr: Array of shape (T,1) with discounted sum of rewards.
        
        Returns:
        - Dictionary of dW1 and dW2, representing estimated gradients.
        """

        ########################################################################
        # TODO: Determine the gradients dW1 and dW2 based on the policy        #
        # gradients formula. You will need to backpropgate the gradients and   #
        # also make sure the sum-of-rewards coefficients from discounted_epr   #
        # are appropriately multiplied to the gradients.                       #
        #                                                                      #
        # NOTE: You may find it useful to review homeworks 1 and 2, especially #
        # the simple two-layer FC cases, which apply here. You will also need  #
        # to vectorize the code to get (Pong) results quickly.                 #
        ########################################################################
        dW1 = None
        dW2 = None
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return {'W1':dW1, 'W2':dW2}


    def train(self, environment="CartPole-v0", max_episodes=100, print_every=1):
        """ Runs policy gradients and records output.

        Inputs:
        - environment: Currently, only CartPole or Pong are supported.
        - max_episodes: An integer representing the maximum number of episodes
          to train. For Pong, you will likely need to run this for at least 500
          episodes observe any improvement in running_mean performance.
        - print_every: Interval for printing debug messages.
        """

        if environment != "CartPole-v0" and environment != "Pong-v0":
            raise ValueError("input environment={} not supported".format(environment))

        # OpenAI initialization. Each observation for Pong is (210,160,3)
        # representing the game screen; for CartPole, it is a (4,) array.
        env = gym.make(environment)
        observation = env.reset()
        xs,hs,dprobs,drs = [],[],[],[]
        prev_x = None

        # Various statistics for rewards, timing, etc.
        reward_sum = 0
        episode_number = 0
        start = time.time()

        # Each iteration is one time step; loop exits according to max_episodes.
        while True:
            if self.render: env.render()
        
            if environment == "CartPole-v0":
                # In CartPole, 0=LEFT, 1=RIGHT.
                x = observation
                aprob, h = self.policy_forward(x)
                action = 0 if np.random.uniform() < aprob else 1

                # Assume y=1 means LEFT and y=0 means RIGHT. These are like fake
                # supervised learning labels. We write these here and the policy
                # gradient reward sum term will scale it (+ or -) appropriately.
                y = 1 if action == 0 else 0 

            elif environment == "Pong-v0":
                # Preprocess observation, setting network input to be the frame difference.
                cur_x = self.preprocess(observation)
                x = cur_x - prev_x if prev_x is not None else np.zeros(self.D)
                prev_x = cur_x

                # Forward input to network and sample action from the probability.
                # In Pong, 2=UP and 3=DOWN. For now, ignore the NO-OP action (i.e. 0).
                aprob, h = self.policy_forward(x)
                action = 2 if np.random.uniform() < aprob else 3

                # Assume y=1 means UP and y=0 means DOWN. Fake labels.
                y = 1 if action == 2 else 0 

            # Record various values that we will use for backpropagation later.
            xs.append(x)
            hs.append(h)

            ####################################################################
            # TODO: write down a gradient that encourages the action that was  #
            # chosen above (i.e. "y") to be chosen again later and record it   #
            # in dprobs. Do this by calling dprobs.append(your_gradient_here). #
            # This should be a one-liner. For more info, see:                  #
            # http://cs231n.github.io/neural-networks-2/#losses                #
            ####################################################################

            ####################################################################
            #                        END OF YOUR CODE                          #
            ####################################################################
       
            # One time-step into the environment and get new observation, etc.
            # Also record reward in 'drs', which has to be done after step().
            observation, reward, done, info = env.step(action)
            reward_sum += reward
            drs.append(reward) 
        
            if done:
                episode_number += 1

                # Stack all inputs, hidden states, action gradients, and rewards for this episode
                ep_x = np.vstack(xs)
                ep_h = np.vstack(hs)
                ep_dprobs = np.vstack(dprobs)
                ep_r = np.vstack(drs)
                xs,hs,dprobs,drs = [],[],[],[]
        
                # Compute the discounted reward backwards through time, using
                # the method you implemented earlier.
                reset = False
                if environment == "Pong-v0":
                    reset = True
                discounted_epr = self.discount_rewards(ep_r, do_reset=reset)

                ################################################################
                # TODO: Using discounted_epr, ep_x, ep_h, and ep_dprobs,       #
                # call self.policy_backward to get the policy gradients. Then  #
                # use those gradients to update self.grad_buffer, since        #
                # official updates only happen every self.batch_size episodes. #
                #                                                              #
                # NOTE: You will want to standardize discounted_epr so that it #
                # has mean 0 and standard deviation 1, to control the gradient #
                # estimator variance.                                          #
                ################################################################

                ################################################################
                #                        END YOUR CODE                         #
                ################################################################
       
                # Do RMSProp parameter update every batch_size episodes.
                if episode_number % self.batch_size == 0:
                    for k,v in self.model.iteritems():
                        g = self.grad_buffer[k] # gradient
                        self.rmsprop_cache[k] = self.decay_rate * self.rmsprop_cache[k] + (1 - self.decay_rate) * g**2
                        self.model[k] += self.learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
                        self.grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
        
                # running_reward is 99% previous value, 1% this reward (i.e. a
                # moving average). This gets appended to self.running_rewards.
                running_reward = reward_sum if len(self.running_rewards)==0 else self.running_rewards[-1]*0.99 + reward_sum*0.01
                elapsed = time.time() - start

                if episode_number % print_every == 0:
                    print("Ep. {} done, reward: {}, running_reward: {:.4f}, time (sec): {:.4f}".
                          format(episode_number, reward_sum, running_reward, elapsed))
                self.final_ep_rewards.append(reward_sum)
                self.running_rewards.append(running_reward)

                if episode_number == max_episodes:
                    print("Whew! All done with {} episodes!".format(episode_number))
                    break
        
                # Reset stuff to let the next episode run.
                reward_sum = 0
                observation = env.reset()
                prev_x = None
