import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def rolling_avg_scores(scores, window):
    if len(scores) <= window:
        return [np.mean(scores)]
    else:
        return [np.mean(scores[x:x+window]) for x in range(len(scores) - window + 1)]

class PPOController:
    
    def __init__(self, env, brain_name, config, policy=None):
        self.env = env
        self.brain_name = brain_name
        self.__dict__.update(config.as_dict())
        self.policy = Policy(config, 33, 4) if policy is None else policy
        self.epsilon = config.epsilon_start
        self.beta = config.beta_start
        self.scores = []
        self.surrogates = []
        self.last_divergence = 0
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
    def solve(self):
        for i_episode in range(1, self.num_episodes + 1):    

            old_log_probabilities, states, actions, rewards = self.collect_trajectories(self.env, self.brain_name, self.policy)
            
            self.scores.append(np.mean(np.sum(rewards, axis=0)))
            
            surrogate_buffer = self.train_loop(old_log_probabilities, states, actions, rewards)

            self.epsilon *= self.epsilon_decay

            self.surrogates.append(np.mean(surrogate_buffer))

            self.print_status(i_episode)
        
        return self.scores, self.surrogates
            
    def act(self, states):
        states = torch.from_numpy(states).float().to(device)
        self.policy.eval()
        actions, log_probabilities = self.policy.next_actions(states, self.std)
        return actions.cpu().data.numpy(), log_probabilities.cpu().data.numpy()
            
    def collect_trajectories(self, env, brain_name, policy):
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        state = env_info.vector_observations
        states = []
        actions = []
        log_probabilities = []
        rewards = []
        while True:
            states.append(state)
            action, log_probability = self.act(state)
            actions.append(action)
            log_probabilities.append(log_probability)
            env_info = self.env.step(action)[self.brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            rewards.append(reward)
            done = env_info.local_done
            state = next_state
            if np.any(done):
                break
        return np.array(log_probabilities), np.array(states), np.array(actions), np.array(rewards)
    
    def train_loop(self, old_log_probabilities, states, actions, rewards):
        surrogate_buffer = []
        
        rewards_future = self.compute_discounted_future_rewards(rewards)
        
        mean = np.mean(rewards_future, axis=1)
        std = np.std(rewards_future, axis=1)
        
        normalized_rewards = rewards_future
        mask = std != 0
        normalized_rewards[mask] = (rewards_future[mask] - mean[mask][:,np.newaxis]) / std[mask][:,np.newaxis]
        # if the deviation is null set the normalized reward to 0
        mask = std == 0
        normalized_rewards[mask] = 0
        
        old_log_probabilities = torch.from_numpy(old_log_probabilities).float().to(device)
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        normalized_rewards = torch.from_numpy(normalized_rewards).float().to(device)
        self.policy.train()
        for _ in range(self.train_iterations):
            surrogate, divergence = self.compute_surrogate(old_log_probabilities, states, actions, normalized_rewards)
            surrogate_buffer.append(surrogate.cpu().data.numpy())
            self.optimizer.zero_grad()
            surrogate.backward()
            #torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
            self.optimizer.step()
            if divergence > self.divergence_target * 1.5:
                self.beta *= 2
            elif divergence < self.divergence_target / 1.5:
                self.beta /= 2
        self.last_divergence = 0
        return surrogate_buffer
    
    def compute_surrogate(self, old_log_probabilities, states, actions, rewards):
        
        new_log_probabilities, entropy = self.policy.get_log_probabilities_and_entropy(states, actions, self.std)
        
        ratio = torch.exp(new_log_probabilities - old_log_probabilities)

        clip = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        clipped_surrogate = torch.min(ratio * rewards, clip * rewards)

        # include a regularization term to promote a step wise evolution of the policy
        # divergence = F.kl_div(new_probabilities, old_probabilities, reduction='batchmean')

        # this returns an average of all the entries of the tensor
        # effective computing L_sur^clip / T
        # averaged over time-step and number of trajectories
        # this is desirable because we have normalized our rewards
        # return torch.mean(clipped_surrogate + self.beta * divergence), divergence
        return -1 * torch.mean(clipped_surrogate) - 0.01 * entropy.mean(), 0
            
    def compute_discounted_future_rewards(self, rewards):
        # This is complex so giving an example with gamma = 0.5 and
        # rewards = [[1, 0], [1, 1]]
        main_dim = len(rewards)
        # discounts = [1, 0.5]
        discounts = (self.gamma ** np.arange(main_dim))
        # discounts = [[1, 0.5],
        #              [1, 0.5]]
        discounts = np.tile(discounts, main_dim).reshape(main_dim, main_dim)
        # indexes = [[0, 1],
        #            [1, 2]]
        indexes = np.tile(np.arange(main_dim), main_dim).reshape(main_dim, main_dim) + np.arange(main_dim)[:,np.newaxis]
        # indexes = [[0, 1],
        #            [1, 0]]
        indexes = np.mod(indexes, main_dim)
        # discounts = [[1, 0.5],
        #              [0, 1]]
        discounts = np.triu(discounts[range(main_dim), indexes])
        # discounts = [[1.5, 0.5],
        #              [1, 1]]
        return np.dot(rewards.T, discounts.T).T
    
    def print_status(self, i_episode):
        print("\rEpisode %d/%d | Average Score: %.2f | Model surrogate: %.5f | Divergence: %.2f   " % (
                i_episode,
                self.num_episodes,
                self.scores[-1],
                self.surrogates[-1],
                self.last_divergence), end="")
        sys.stdout.flush()
        
class Policy(nn.Module):
    
    def __init__(self, config, state_size, action_size):
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
        x = states
        for fc in self.fc:
            x = F.relu(fc(x))
        normal_mean = F.relu(self.normal_mean_fc(x))
        normal_mean = torch.tanh(self.normal_mean(normal_mean))
        return normal_mean
        
    def next_actions(self, states, std):
        with torch.no_grad():
            normal_mean = self.forward(states)
            cov_mat = torch.diag_embed(torch.full((self.action_size,), std ** 2))
            normal = torch.distributions.multivariate_normal.MultivariateNormal(normal_mean, cov_mat)
            actions = torch.clamp(normal.sample(), -1, 1)
            log_probabilities = normal.log_prob(actions)
        return actions, log_probabilities
        
    def get_log_probabilities_and_entropy(self, states, actions, std):
        normal_mean = self.forward(states)
        cov_mat = torch.diag_embed(torch.full((self.action_size,), std ** 2))
        normal = torch.distributions.multivariate_normal.MultivariateNormal(normal_mean, cov_mat)
        log_probabilities = normal.log_prob(actions)
        return log_probabilities, normal.entropy()
        
        
        