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

    def parameters(self):
        return torch.nn.Linear(5, 5).parameters()

    def eval(self):
        return

class Config:
    def __init__(self):
        self.num_agents = 2
        self.state_size = 5
        self.action_size = 4
        
        self.num_episodes = 1
        self.epsilon_start = 0.05
        self.max_memory = 2
        self.epsilon_decay = 0.995
        self.learning_rate = 5e-4
        self.train_iterations = 4
        self.gamma = 0.5
        self.mlp_specs = (200, 150)
        
    def as_dict(self):
        return self.__dict__
    
class TestController(unittest.TestCase):
    
    def setUp(self):
        self.env = MockEnvironment((33,), [[-1, 1] for i in range(4)], 20)
        self.policy = MockContinuousPolicy((4,))
        self.critic = MockCritic()
        self.controller = PPOController(self.env, 'bla', Config(), policy=self.policy, critic=self.critic)
        
    def test_compute_discounted_future_rewards(self):
        rewards = np.array([[1, 0], [1, 1]])
        rewards = self.controller.compute_discounted_future_rewards(rewards)
        self.assertTrue(np.all(rewards == np.array([[1.5, 0.5], [1, 1]])))
        
        

if __name__ == '__main__':
    unittest.main()