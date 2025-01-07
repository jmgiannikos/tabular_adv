from abc import ABC, abstractmethod
import numpy as np

# probably not going to need this, since we want to use fully implemented models (preferrably already pre-trained, but that may be a bit too optimistic). May serve as a wrapper
class Cls_model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, x):
        pass

class Random_Guesser(Cls_model):
    def __init__(self):
        super().__init__()
        self.guesser = np.random.default_rng()
    
    def predict(self, x):
        guesses = self.guesser.integers(0,2,size=x.shape[0])
        return guesses