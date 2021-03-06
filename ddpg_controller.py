from memory import AgentMemory
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGController:
    """
    Deep learning agent based on Deep Deterministic Policy Gradient described in https://arxiv.org/pdf/1509.02971.pdf
    
    """

    def __init__(self, env, brain_name, config):
        """
        Constructor methods to create the controller

        Parameters
        ----------
        env - Unity environment for the agent to solve
        brain_name, string, brain name used in conjunction with the environment
        config - Dictionary containing the following keys:
        - 'num_episodes', int, number of episodes to run the agent for
        - 'gamma', float, discount rate for future rewards
        - 'tau', float, rate for the soft update of the target network
        - 'max_memory', int, size of the replay buffer in number of samples
        - 'batch_size', int, size of the batches sampled to train the model on each update
        - 'update_every', int, update frequency, in number of steps
        - 'mlp_layers', int tuple, shape of the multilayer perceptron model
        - 'learning_rate', float, learning rate for the training of the model
        - 'state_size', int
        - 'action_size', int
        - 'num_agents', int, number of agents running in parallel in the environment

        """
        self.env = env
        self.brain_name = brain_name
        self.__dict__.update(config.as_dict())
        self.trained_policy = Policy(config, self.state_size, self.action_size)
        self.target_policy = Policy(config, self.state_size, self.action_size)
        self.trained_critic = Critic(config, self.state_size, self.action_size)
        self.target_critic = Critic(config, self.state_size, self.action_size)
        # those networks will never be trained
        self.target_policy.eval()
        self.target_critic.eval()
        self.memory = AgentMemory(
            ((self.num_agents, self.state_size), (self.num_agents, self.action_size), (self.num_agents, self.state_size), (self.num_agents,), (self.num_agents,)), int(self.max_memory))
        self.scores = []
        self.critic_losses = []
        self.surrogates = []

        self.critic_optimizer = optim.Adam(
            self.trained_critic.parameters(), lr=config.learning_rate)
        self.policy_optimizer = optim.Adam(
            self.trained_policy.parameters(), lr=config.learning_rate)

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
            critic_losses = []
            while True:
                action = self.act(state)
                env_info = self.env.step(action)[self.brain_name]
                next_state = env_info.vector_observations
                reward = env_info.rewards
                done = env_info.local_done
                self.memory.add(
                    (state, action, next_state, reward, done))
                state = next_state
                rewards.append(reward)
                if self.memory.size >= self.batch_size and not step % self.update_every:
                    surrogate_buffer, critic_loss = self.train()
                    surrogates.append(surrogate_buffer)
                    critic_losses.append(critic_loss)
                step += 1
                if np.any(done):
                    break

            self.scores.append(np.mean(np.sum(rewards, axis=0)))
            self.surrogates.append(np.mean(surrogates))
            self.critic_losses.append(np.mean(critic_losses))

            self.print_status(i_episode)

        return self.scores, self.surrogates, self.critic_losses

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
        self.trained_policy.eval()
        with torch.no_grad():
            actions = self.trained_policy(states)
        # TODO: add exploration noise
        return actions.cpu().data.numpy()

    def train(self):
        """
        Training routine to update the policy and critic

        """
        states, actions, next_states, rewards, dones = self.memory.sample(
            self.batch_size)

        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        # critic update
        next_actions = self.target_policy(next_states)
        self.trained_critic.train()
        self.critic_optimizer.zero_grad()
        done_mask = 1 - dones
        target_states_values = rewards + self.gamma * \
            self.target_critic(next_states, next_actions) * done_mask
        predicted_states_values = self.trained_critic(states, actions)
        critic_loss = torch.mean(
            (target_states_values - predicted_states_values) ** 2)
        critic_loss.backward()
        self.critic_optimizer.step()

        # policy update
        self.trained_policy.train()
        self.policy_optimizer.zero_grad()
        action_values = self.trained_critic(
            states, self.trained_policy(states))
        surrogate = -torch.mean(action_values)
        surrogate.backward()
        self.policy_optimizer.step()

        self.target_network_update(self.trained_critic, self.target_critic)
        self.target_network_update(self.trained_policy, self.target_policy)

        return surrogate.cpu().data.numpy(), critic_loss.cpu().data.numpy()

    def target_network_update(self, trained_model, target_model):
        """
        Performs a soft update with rate tau from the trained_model to the target_model.

        """
        target_model_weights = target_model.get_weights()
        train_model_weights = trained_model.get_weights()
        new_weights = []
        for w1, w2 in zip(target_model_weights, train_model_weights):
            new_weights.append(w1 * (1 - self.tau) + w2 * self.tau)
        target_model.set_weights(new_weights)

    def print_status(self, i_episode):
        """
        Print the latest status of the agent

        Parameter
        ---------
        i_episode, int

        """
        print("\rEpisode %d/%d | Average Score: %.2f | Surrogate: %.5f | Critic loss: %.5f  " % (
            i_episode,
            self.num_episodes,
            self.scores[-1],
            self.surrogates[-1],
            self.critic_losses[-1]), end="")
        sys.stdout.flush()


class Policy(nn.Module):

    def __init__(self, config, state_size, action_size):
        """
        Constructor for the policy.
        
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
        self.action_output = nn.Linear(self.mlp_specs[-1], action_size)

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
        action_output = torch.tanh(self.action_output(x))
        return action_output

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


class Critic(nn.Module):

    def __init__(self, config, state_size, action_size):
        """
        Constructor for the critic.
        
        Parameters
        ----------
        - config, dictionary with the same keys as the controller 
        - state_size, int, size of the input to the model
        - action_size, int, size of the policy output

        """
        super(Critic, self).__init__()
        self.__dict__.update(config.as_dict())

        # state branch
        self.state_fc = []
        in_node = state_size
        for spec in self.mlp_specs:
            self.state_fc.append(nn.Linear(in_node, spec))
            in_node = spec
        # the layers need to be properties of the class instance for the train operation to work
        for i, fc in enumerate(self.state_fc):
            setattr(self, 'fc_state_' + str(i), fc)

        # merge layer
        self.merge_fc = []
        in_node = self.mlp_specs[-1] + action_size
        for spec in self.mlp_specs:
            self.merge_fc.append(nn.Linear(in_node, spec))
            in_node = spec
        # the layers need to be properties of the class instance for the train operation to work
        for i, fc in enumerate(self.merge_fc):
            setattr(self, 'fc_merge_' + str(i), fc)
        self.final_layer = nn.Linear(self.mlp_specs[-1], 1)

    def forward(self, states, actions):
        """
        Implements the forward pass of the critic.
        
        Parameters
        ----------
        - states, float Tensor shape=(batch_size, num_agents, state_size)
        - actions, float Tensor shape=(batch_size, num_agents, action_size)
        
        Return
        ----------
        Action values, float Tensor shape=(batch_size, num_agents)

        """
        x = states
        for fc in self.state_fc:
            x = F.relu(fc(x))

        merge = torch.cat((x, actions), -1)
        z = merge
        for fc in self.merge_fc:
            z = F.relu(fc(z))

        return self.final_layer(z).squeeze()

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
