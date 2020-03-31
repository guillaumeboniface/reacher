import unittest
import numpy as np
from collections import *
from ppo_controller import *
import torch
    
EnvInfo = namedtuple('EnvInfo', ['vector_observations', 'rewards', 'local_done'])
    
class MockEnvironment:
    
    def __init__(self, state_shape, action_shape, n_agents):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.n_agents = n_agents
        self.done_generator = self.gen_done()
        
    def reset(self, train_mode=True):
        return defaultdict(lambda: EnvInfo(
            np.zeros((self.n_agents,) + self.state_shape),
            np.zeros(self.n_agents),
            np.zeros(self.n_agents) + self.done))
    
    def step(self, state):
        result = self.reset()
        return result
    
    @property
    def done(self):
        return next(self.done_generator)
    
    def gen_done(self):
        for i in range(2):
            yield 0
        while True:
            yield 1
    

class MockContinuousPolicy:

    def __init__(self, action_shape):
        self.action_shape = action_shape
    
    def next_actions(self, state):
        # return actions and probabilities
        return torch.ones((state.shape[0],) + self.action_shape), torch.ones((state.shape[0],) + self.action_shape)
    
    def parameters(self):
        return torch.nn.Linear(5, 5).parameters()
    
    def eval(self):
        return
    
class MockCritic:
        
    def __call__(self, states):
        return torch.arange(np.prod(states.shape)).view(states.shape)

class Config:
    def __init__(self):
        self.num_episodes = 1
        self.epsilon_start = 0.05
        self.beta_start = 1
        self.epsilon_decay = 0.995
        self.learning_rate = 5e-4
        self.train_iterations = 4
        self.gamma = 0.5
        self.mlp_specs = (200, 150)
        self.gae_lambda = 0.5
        
    def as_dict(self):
        return self.__dict__
    
class TestCollectTrajectories(unittest.TestCase):
    
    def setUp(self):
        self.env = MockEnvironment((33,), [[-1, 1] for i in range(4)], 20)
        self.policy = MockContinuousPolicy((4,))
        self.critic = MockCritic()
        self.controller = PPOController(self.env, 'bla', Config(), policy=self.policy, critic=self.critic)

    def test_shape(self):
        probabilities, states, actions, rewards = self.controller.collect_trajectories(self.env, 'bla', self.policy)
        self.assertEqual(probabilities.shape, (2, 20, 4))
        self.assertEqual(states.shape, (2, 20, 33))
        self.assertEqual(actions.shape, (2, 20, 4))
        self.assertEqual(rewards.shape, (2, 20))
        
    def test_compute_discounted_future_rewards(self):
        rewards = np.array([[1, 0], [1, 1]])
        rewards = self.controller.compute_discounted_future_rewards(rewards)
        self.assertTrue(np.all(rewards == np.array([[1.5, 0.5], [1, 1]])))
        
    def test_compute_gae_matrix(self):
        rewards = np.array([[1, 0], [1, 1]])
        matrix = self.controller.compute_gae_matrix(rewards)
        self.assertTrue(np.all(matrix == np.array([[[1, 0], [1, 1]], [[1.5, 0.5], [1, 1]]])))
        
    def test_compute_gae(self):
        rewards = np.array([[1, 0], [1, 1]])
        matrix = self.controller.compute_gae_matrix(rewards)
        print(matrix)
        states = torch.ones((2, 2))
        gae = self.controller.compute_gae(states, torch.from_numpy(matrix))
        print(gae)
        self.assertTrue(torch.all(gae == torch.Tensor([[1.875, 0.875], [-0.75, -1.5]])))
        
        

if __name__ == '__main__':
    unittest.main()