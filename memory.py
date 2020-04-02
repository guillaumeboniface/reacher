import numpy as np

class AgentMemory:
    """
    Replay buffer storing samples made of states, actions, next_states, rewards and dones.
    Implemented through numpy arrays. Can support prioritized sampling.
    
    """
    def __init__(self, storage_shapes, maxlen, is_prioritization=False):
        """
        Constructor for AgentMemory
        
        Parameters
        ----------
        - storage_shapes, array, each element describes the shape of the item that will be stored in a bucket
        'None' is used to indicate a scalar bucket.
        - maxlen, int, maximum number of samples to store in the buffer
        - is_prioritization, boolean, whether to use prioritization

        """
        # creating the buckets of each type of items
        self.buckets = []
        for shape in storage_shapes:
            if shape is None:
                self.buckets.append(np.zeros(maxlen))
            else:
                self.buckets.append(np.zeros((maxlen,) + shape))
        self.index = 0
        self.maxlen = maxlen
        self.max_prioritization = 1.0
        self.is_prioritization = is_prioritization
        if is_prioritization:
            # in the prioritization case, maintain a memory of the last samples index to allow for weights updates
            self.last_index = []
            self.prioritizations = np.zeros(maxlen)
            self.sample = self.sample_with_prioritization
        else:
            self.sample = self.simple_sample
        
    def add(self, sample):
        """
        Add a sample to the replay buffer
        
        Parameter
        ---------
        - sample, tuple of items with orders and shapes matching the initial storage_shapes passed to the constructor
        
        """
        for item, bucket in zip(sample, self.buckets):
            bucket[self.index % self.maxlen] = item
        if self.is_prioritization:
            # new samples should always be stored with max_prioritization
            self.prioritizations[self.index % self.maxlen] = self.max_prioritization
        # point to the next empty location in the buffer
        self.index += 1
        
    def get_latest(self, n):
        """
        Returns the last n samples added to the replay buffer
        
        Parameter
        ---------
        - n, int, number of samples
        
        Return
        ---------
        - samples, array of buckets of length n
        
        """
        samples_index = np.array(range(self.index - n, self.index)) % self.maxlen
        return [bucket[samples_index] for bucket in self.buckets]
    
    def simple_sample(self, n):
        """
        Returns n samples from the replay buffer, chosen with uniform probability
        
        Parameter
        ---------
        - n, int, number of samples
        
        Return
        ---------
        - samples, array of buckets of length n
        
        """
        samples_index = np.random.choice(min(self.index, self.maxlen), replace=False, size=n)
        return [bucket[samples_index] for bucket in self.buckets]
    
    def sample_with_prioritization(self, n):
        """
        Returns n samples from the replay buffer, sampled according to respective priorities
        
        Parameter
        ---------
        - n, int, number of samples
        
        Return
        ---------
        - samples, array of buckets of length n, the last bucket in the array are the samples' priorities
        
        """
        if self.index < self.maxlen:
            probabilities = self.prioritizations[:self.index]/sum(self.prioritizations[:self.index])
        else:
            probabilities = self.prioritizations/sum(self.prioritizations)
        samples_index = np.random.choice(min(self.index, self.maxlen), replace=False, size=n, p=probabilities)
        # maintain a memory of the last samples index to allow for weights updates
        self.last_index = samples_index
        samples = [bucket[samples_index] for bucket in self.buckets]
        samples.append(probabilities[samples_index])
        return samples
    
    def update_prioritization(self, prioritizations):
        """
        Updates the priorities of the last sampled elements
        
        Parameter
        ---------
        - prioritizations, float array, length must match the last_index length
        
        """
        self.prioritizations[self.last_index] = prioritizations
        self.max_prioritization = np.max(self.prioritizations)
    
    @property
    def size(self):
        return min(self.index, self.maxlen)