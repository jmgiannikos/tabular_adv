from abc import ABC, abstractmethod
import numpy as np

class Adv_model(ABC):
    def __init__(self, victim, constraints, metadata, *hyperparams):
        self.victim = victim
        self.constraints = constraints
        self.hyperparams = [hyperparam for hyperparam in hyperparams]
        self.metadata = metadata

    #TODO: kinda wanna also make this a property, since it is, but python seems to be weird with abstract class properties and I dont wanna deal with that now
    @staticmethod
    @abstractmethod
    def TRAINABLE():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def SEARCH_DIMS():
        raise NotImplementedError

    @abstractmethod
    def attack(self, x):
        pass

# baseline adversarial that changes every real valued parameter by a normal distributed random amount. hyperparameters are mean and std_dev of the normal distribution
# hyperparameter optimization might not converge, due to the randomness of behavior here, but so what.
class Baseline_adv(Adv_model):
    def __init__(self, victim, constraints, metadata, *hyperparams):
        super().__init__(victim, constraints, metadata, *hyperparams)
        self.real_feature_map = np.array([feat_type=="real" for feat_type in metadata.query("mutable == True")["type"].values.tolist()]).astype(int)
        self.random_generator = np.random.default_rng()

    @staticmethod
    def HYPERPARAM_NAMES(): #names are in order
        return ["mean", "std_dev"]

    @staticmethod
    def TRAINABLE():
        return False
    
    @staticmethod
    def SEARCH_DIMS():
        return [(-4,4), (0.0001,10)] # these are chosen pretty arbitrarily for testing purposes

    def attack(self, x):
        # NOTE: For some reason hyperparameter indexes is a list with 2 dimensions. Indexing was adjusted, but I should check, why this happens
        base_perturbations = self.random_generator.normal(self.hyperparams[0][0], self.hyperparams[0][1], size=x.shape) #draws random for every val, then masks, which is inefficient, but whatever this is just a placeholder anyways
        perturbation_mask = np.expand_dims(self.real_feature_map, axis=0)
        perturbation_mask = np.reshape(np.tile(perturbation_mask, x.shape[0]), x.shape)
        perturbations = np.multiply(base_perturbations, perturbation_mask)
        adv_samples = np.add(x, perturbations)
        return adv_samples

