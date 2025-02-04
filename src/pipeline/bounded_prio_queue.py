from itertools import takewhile
from more_itertools import before_and_after
import torch
import utils

class Bounded_Priority_Queue():
    def __init__(self, max_size, start_elements):
        self.max_size = max_size
        self.queue = []
        self.push(start_elements)

    def push(self, nodes): # NOTE: nodes should have shape (eval_result, value)
        nodes = sorted(nodes, key=lambda x: x[0])
        if len(self.queue) >= self.max_size:
            nodes = list(takewhile(lambda x: x[0] < self.queue[-1][0], nodes)) # if queue already full, pre-filter new nodes

        self.temp_queue = []
        _ = map(self.insert, nodes)
        self.queue = self.temp_queue

        if len(self.queue) > self.max_size:
            self.queue = self.queue[:self.max_size]

    def insert(self, node):
        smaller_values, greater_values = before_and_after(lambda x: x[0] < node[0], self.queue)
        self.queue = greater_values
        smaller_values.append(node)
        self.temp_queue.extend(smaller_values)
    
    def pop(self, n=1):
        values = self.queue[:n]
        self.queue = self.queue[n:]
        return values
    
    def pop_max(self, n=1):
        values = self.queue[-n:]
        self.queue = self.queue[:-n]
        return values
    
    def is_not_in(self, samples):
        return utils.is_not_in(samples, [node[1] for node in self.queue])
    
    def is_empty(self):
        return len(self.queue) == 0

    def get(self, idx):
        return self.queue[idx]