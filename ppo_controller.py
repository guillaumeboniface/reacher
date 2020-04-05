import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from memory import AgentMemory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPOController:
    """
    Deep learning agent based on Proximal Policy Optimization, based on https://arxiv.org/pdf/1506.02438.pdf

    """

    def __init__(self, env, brain_name, config, policy=None, critic=None):
        """
        Constructor methods to create the controller

        Parameters
        ----------
        env - Unity environment for the agent to solve
        brain_name, string, brain name used in conjunction with the environment
        config - Dictionary containing the following keys:
        - 'num_episodes', int, number of episodes to run the agent for
        - 'epsilon_start', float, initial value for epsilon used in the PPO algorithm to clip the surrogate
        - 'epsilon_decay', float, rate of decay for epsilon, applied after every episode
        - 'gamma', float, discount rate for future rewards
        - 'tau', float, rate for the soft update of the target network
        - 'max_memory', int, size of the replay buffer in number of samples
        - 'update_every', int, update frequency, in number of steps
        - 'train_iterations', int, number of training passes over a data batch
        - 'mlp_layers', int tuple, shape of the multilayer perceptron model
        - 'learning_rate', float, learning rate for the training of the model
        - 'std', float, standard deviation used for the Normal distribution of the policy
        - 'state_size', int
        - 'action_size', int
        - 'num_agents', int, number of agents running in parallel in the environment

        - 'policy', optional, used to pass a mock policy for testing purposes
        - 'critic', optional, used to pass a mock critic for testing purposes

        """
        self.env = env
        self.brain_name = brain_name
        self.__dict__.update(config.as_dict())
        self.policy = Policy(config, self.state_size,
                             self.action_size) if policy is None else policy
        self.trained_critic = Critic(
            config, self.state_size) if critic is None else critic
        self.target_critic = Critic(
            config, self.state_size) if critic is None else critic
        self.target_critic.eval()
        self.memory = AgentMemory(
            ((self.num_agents, self.state_size), (self.num_agents, self.action_size), (self.num_agents,), (self.num_agents, self.state_size), (self.num_agents,), (self.num_agents,)), int(self.max_memory))
        self.epsilon = config.epsilon_start
        self.scores = []
        self.surrogates = []

        self.optimizer = optim.Adam([{'params': self.policy.parameters()},
                                     {'params': self.trained_critic.parameters()}], lr=config.learning_rate)

    def solve(self):
        """
        Main method to launch the environment loop

        """
        step = 1

        for i_episode in range(1, self.num_episodes + 1):
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            state = env_info.vector_observations
            rewards = []
            surrogates = []
            while True:
                action, log_probability = self.act(state)
                env_info = self.env.step(action)[self.brain_name]
                next_state = env_info.vector_observations
                reward = env_info.rewards
                done = env_info.local_done
                self.memory.add(
                    (state, action, log_probability, next_state, reward, done))
                state = next_state
                rewards.append(reward)
                if not step % self.update_every:
                    surrogate_buffer = self.train_loop()
                    surrogates.append(surrogate_buffer)
                step += 1
                if np.any(done):
                    break

            self.scores.append(np.mean(np.sum(rewards, axis=0)))
            self.surrogates.append(np.mean(surrogates))

            self.epsilon *= self.epsilon_decay
            self.print_status(i_episode)

        return self.scores, self.surrogates

    def act(self, states):
        """
        Based on states, returns the on-policy actions
        
        Parameter
        ---------
        states - float array shape=(num_agents, state_size)
        
        Return
        ---------
        Float array shape=(num_agents, action_size), chosen action

        """
        states = torch.from_numpy(states).float().to(device)
        self.policy.eval()
        actions, log_probabilities = self.policy.next_actions(states)
        return actions.cpu().data.numpy(), log_probabilities.cpu().data.numpy()

    def train_loop(self):
        """
        Training routine to update the policy and critic

        """
        surrogate_buffer = []
        states, actions, old_log_probabilities, next_states, rewards, dones = self.memory.get_latest(
            self.update_every)

        future_rewards = self.compute_discounted_future_rewards(rewards)

        old_log_probabilities = torch.from_numpy(
            old_log_probabilities).float().to(device)
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        future_rewards = torch.from_numpy(future_rewards).float().to(device)
        dones = torch.from_numpy(dones).bool().to(device)
        self.policy.train()
        self.trained_critic.train()
        for _ in range(self.train_iterations):
            surrogate = self.compute_surrogate(
                old_log_probabilities, states, actions, next_states, future_rewards, dones)
            surrogate_buffer.append(surrogate.cpu().data.numpy())
            self.optimizer.zero_grad()
            surrogate.backward()
            self.optimizer.step()
            self.target_network_update()
        return surrogate_buffer

    def compute_surrogate(self, old_log_probabilities, states, actions, next_states, future_rewards, dones):
        """
        Compute the surrogate, i.e. the function optimized at training time

        Parameters
        ----------
        - old_log_probabilities, float Tensor shape=(batch_size, num_agents), original probabilities for the performed action
        - states, float Tensor shape=(batch_size, num_agents, state_size)
        - actions, float Tensor shape=(batch_size, num_agents, action_size)
        - next_states, float Tensor shape=(batch_size, num_agents, state_size)
        - future_rewards, float Tensor shape=(batch_size, num_agents), discounted sum of future rewards over the length of the trajectory
        - dones, float Tensor shape=(batch_size, num_agents)

        Return
        ---------
        Surrogate, float Tensor

        """
        new_log_probabilities, entropy = self.policy.get_log_probabilities_and_entropy(
            states, actions)
        ratio = torch.exp(new_log_probabilities - old_log_probabilities)

        with torch.no_grad():
            states_values = self.target_critic(states)
            next_states_values = self.target_critic(next_states[-1, :])
        if torch.any(dones):
            final_states_values = 0
        else:
            final_states_values = next_states_values.expand(
                states_values.shape)

        future_rewards = self.normalize(future_rewards)

        discount = self.gamma ** torch.arange(
            len(states_values), 0, -1, dtype=torch.float).unsqueeze(1)
        target_states_values = future_rewards + final_states_values * discount
        advantages = target_states_values - states_values

        clip = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        clipped_surrogate = torch.min(ratio * advantages, clip * advantages)

        return -1 * torch.mean(clipped_surrogate) + 0.5 * self.trained_critic.mse(states_values, target_states_values) - 0.01 * entropy.mean()

    def normalize(self, a):
        """
        Normalize a torch Tensor

        Parameters
        ----------
        - a, float Tensor to normalize

        """
        mean = torch.mean(a, -1)
        std = torch.std(a, -1)
        b = a
        mask = std != 0
        b[mask] = (a[mask] - mean[mask].unsqueeze(1)) / std[mask].unsqueeze(1)
        # if the deviation is null set the normalized reward to 0
        mask = std == 0
        b[mask] = 0
        return b

    def compute_discounted_future_rewards(self, rewards):
        """
        Compute the discounted sum of future reward over the trajectory

        Parameters
        ----------
        - rewards, float array shape=(batch_size, num_agents)

        Return
        ----------
        Discounted future rewards, float array shape=(batch_size, num_agents)

        """
        # This is complex so giving an example with gamma = 0.5 and
        # rewards = [[1, 0], 
        #            [1, 1]]
        main_dim = len(rewards)
        # discounts = [1, 0.5]
        discounts = (self.gamma ** np.arange(main_dim))
        # discounts = [[1, 0.5],
        #              [1, 0.5]]
        discounts = np.tile(discounts, main_dim).reshape(main_dim, main_dim)
        # indexes = [[0, 1],
        #            [1, 2]]
        indexes = np.tile(np.arange(main_dim), main_dim).reshape(
            main_dim, main_dim) + np.arange(main_dim)[:, np.newaxis]
        # indexes = [[0, 1],
        #            [1, 0]]
        indexes = np.mod(indexes, main_dim)
        # discounts = [[1, 0.5],
        #              [0, 1]]
        discounts = np.triu(discounts[range(main_dim), indexes])
        # rewards = [[1.5, 0.5],
        #              [1, 1]]
        return np.dot(discounts, rewards)

    def target_network_update(self):
        """
        Performs a soft update with rate tau from the trained_model to the target_model.

        """
        target_model_weights = self.target_critic.get_weights()
        train_model_weights = self.trained_critic.get_weights()
        new_weights = []
        for w1, w2 in zip(target_model_weights, train_model_weights):
            new_weights.append(w1 * (1 - self.tau) + w2 * self.tau)
        self.target_critic.set_weights(new_weights)

    def print_status(self, i_episode):
        """
        Print the latest status of the agent

        Parameter
        ---------
        i_episode, int

        """
        print("\rEpisode %d/%d | Average Score: %.2f | Model surrogate: %.5f   " % (
            i_episode,
            self.num_episodes,
            self.scores[-1],
            self.surrogates[-1]), end="")
        sys.stdout.flush()


class Policy(nn.Module):

    def __init__(self, config, state_size, action_size):
        """
        Constructor for the policy
        
        Parameters
        ----------
        - config, dictionary with the same keys as the controller 
        - state_size, int, size of the input to the model
        - action_size, int, size of the policy output

        """
        super(Policy, self).__init__()
        self.__dict__.update(config.as_dict())
        self.action_size = action_size
        self.fc = []
        in_node = state_size
        for spec in self.mlp_specs:
            self.fc.append(nn.Linear(in_node, spec))
            in_node = spec
        # the layers need to be properties of the class instance for the train operation to work
        for i, fc in enumerate(self.fc):
            setattr(self, 'fc_' + str(i), fc)
        self.normal_mean_fc = nn.Linear(self.mlp_specs[-1], self.mlp_specs[-1])
        self.normal_mean = nn.Linear(self.mlp_specs[-1], action_size)

    def forward(self, states):
        """
        Implements the forward pass of the policy.
        
        Parameters
        ----------
        - states, float Tensor shape=(batch_size, num_agents, state_size)
        
        Return
        ----------
        Predictions, float Tensor shape=(batch_size, num_agents, action_space_size)

        """
        x = states
        for fc in self.fc:
            x = F.relu(fc(x))
        normal_mean = torch.relu(self.normal_mean_fc(x))
        normal_mean = torch.tanh(self.normal_mean(normal_mean))
        return normal_mean

    def next_actions(self, states):
        """
        Return the actions sampled from the policy
        
        Parameters
        ----------
        - states, float Tensor shape=(batch_size, num_agents, state_size)
        
        Return
        ----------
        - Actions, float Tensor shape=(batch_size, num_agents, action_space_size)
        - Log probabilities, float Tensor shape=(batch_size, num_agents, action_space_size)

        """
        with torch.no_grad():
            normal_mean = self.forward(states)
            cov_mat = torch.diag_embed(
                torch.full((self.action_size,), self.std ** 2))
            normal = torch.distributions.multivariate_normal.MultivariateNormal(
                normal_mean, cov_mat)
            actions = normal.sample()
            log_probabilities = normal.log_prob(actions)
        return actions, log_probabilities

    def get_log_probabilities_and_entropy(self, states, actions):
        """
        Return the log probabilities for the given actions over the input states
        as well as the entropy of the policy distribution
        
        Parameters
        ----------
        - states, float Tensor shape=(batch_size, num_agents, state_size)
        - actions, float Tensor shape=(batch_size, num_agents, action_size)
        
        Return
        ----------
        - Log probabilities, float Tensor shape=(batch_size, num_agents, action_space_size)
        - Entropy, float

        """
        normal_mean = self.forward(states)
        cov_mat = torch.diag_embed(torch.full(
            (self.action_size,), self.std ** 2))
        normal = torch.distributions.multivariate_normal.MultivariateNormal(
            normal_mean, cov_mat)
        log_probabilities = normal.log_prob(actions)
        return log_probabilities, normal.entropy()


class Critic(nn.Module):

    def __init__(self, config, state_size):
        """
        Constructor for the critic.
        
        Parameters
        ----------
        - config, dictionary with the same keys as the controller 
        - state_size, int, size of the input to the model

        """
        super(Critic, self).__init__()
        self.__dict__.update(config.as_dict())
        self.mse = torch.nn.MSELoss()
        self.fc = []
        in_node = state_size
        for spec in self.mlp_specs:
            self.fc.append(nn.Linear(in_node, spec))
            in_node = spec
        # the layers need to be properties of the class instance for the train operation to work
        for i, fc in enumerate(self.fc):
            setattr(self, 'fc_' + str(i), fc)
        self.final_layer = nn.Linear(self.mlp_specs[-1], 1)

    def forward(self, states):
        """
        Implements the forward pass of the critic.
        
        Parameters
        ----------
        - states, float Tensor shape=(batch_size, num_agents, state_size)
        
        Return
        ----------
        State values, float Tensor shape=(batch_size, num_agents)

        """
        x = states
        for fc in self.fc:
            x = F.relu(fc(x))
        return self.final_layer(x).squeeze()

    def get_weights(self):
        """
        Returns the model weights, necessary to update the target model of the DQN 

        Return
        ---------
        Model weights, float array, 2D arrays (one sub-array per layer in the model)

        """
        return [w.data for w in self.parameters()]

    def set_weights(self, weights):
        """
        Replace the model weights, necessary to update the target model of the DQN 

        Parameters
        ---------
        - weights, float array, 2D arrays (one sub-array per layer in the model)

        """
        for w1, w2 in zip(self.parameters(), weights):
            w1.data.copy_(w2)
