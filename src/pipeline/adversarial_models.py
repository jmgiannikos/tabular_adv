from abc import ABC, abstractmethod
import numpy as np
from gower_distance import Gower_dist
from tabularbench.constraints.relation_constraint import BaseRelationConstraint, EqualConstraint
from tabularbench.constraints.constraints_fixer import ConstraintsFixer
from tabularbench.constraints.constraints_checker import ConstraintChecker
import heapq as hq
import sklearn as sk
import time
from defaults import DEFAULTS as global_defaults

class Adv_model(ABC):
    DEFAULTS = None
    def __init__(self, victim, constraints, metadata, *hyperparams, profile=None):
        self.victim = victim
        self.constraints = constraints
        self.hyperparams = [hyperparam for hyperparam in hyperparams]
        self.metadata = metadata
        self.profile = profile

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
class Dummy_adv(Adv_model):
    def __init__(self, victim, constraints, metadata, *hyperparams, profile=None):
        super().__init__(victim, constraints, metadata, *hyperparams)
        self.real_feature_map = np.array([feat_type=="real" for feat_type in metadata.query("mutable == True")["type"].values.tolist()]).astype(int)
        self.random_generator = np.random.default_rng()
        self.profile = profile

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

class Baseline_adv(Adv_model, sk.base.BaseEstimator):
    DEFAULTS = {
        "target": -0.9, # negative means this managed to flip the value (we negate for min heap purposes)
        "max_iter": 10000, # maximum number of search steps before we abort and just take the best value seen so far
        "parameters": [False, 10, 500]
    }

    def __init__(self, victim, constraints, metadata, constraint_correction=DEFAULTS["parameters"][0], queue_size=DEFAULTS["parameters"][1], total_step_num=DEFAULTS["parameters"][2], profile=None):
        hyperparams = [constraint_correction, queue_size, total_step_num]
        super().__init__(victim, constraints, metadata, *hyperparams)
        self.features = metadata["feature"].tolist()
        self.mutable_feature_choices = {}
        self.feature_idxs = {}
        self.feature_types = {}
        for idx, feature in enumerate(self.features): # NOTE: really hope the order of the metadata feature list aligns with the order of x
            if metadata.query("feature == @feature")["mutable"].to_list()[0]:
                self.mutable_feature_choices[feature] = None
                self.feature_idxs[feature] = idx
                self.feature_types[feature] = metadata.query("feature == @feature")["type"].to_list()[0]

        self.constraint_correction = constraint_correction
        self.queue_size = queue_size
        self.total_step_num = total_step_num

        if self.constraint_correction:
            guard_constraints = [constraint for constraint in constraints.relation_constraints if isinstance(constraint, BaseRelationConstraint) and not isinstance(constraint, EqualConstraint)]
            fix_constraints = [constraint for constraint in constraints.relation_constraints if isinstance(constraint, EqualConstraint)]
            self.constraint_fixer = ConstraintsFixer(guard_constraints=guard_constraints, fix_constraints=fix_constraints)
            self.constraints_checker = ConstraintChecker(constraints, 0)
        self.target = self.DEFAULTS["target"]
        self.max_iterations = self.DEFAULTS["max_iter"]
        self.profile = profile

    @staticmethod
    def HYPERPARAM_NAMES(): #names are in order
        return ["constraint_correction", "queue_size", "total_step_num"]

    @staticmethod
    def TRAINABLE():
        return True
    
    @staticmethod
    def SEARCH_DIMS():
        return [[True, False], (1,100), (2,1000)] # NOTE: these are still chosen arbitrarily. May be subject to adjustment

    def fit(self, x, y): #training here just means storing the seen ranges in x so we can define reasonable step sizes. We also restrict ourselves to remain in distribution -> reasonable restriction???
        self.distance_metric = Gower_dist(x, self.metadata, dynamic=True) #set these to be dynamic so we dont have exploding gower distances
        for feature in self.mutable_feature_choices.keys():
            feature_idx = self.feature_idxs[feature]
            feature_vals = x[:,feature_idx]
            if self.feature_types[feature] == "cat":
                self.mutable_feature_choices[feature] = list(set(feature_vals.to_list()))
            elif self.feature_types[feature] == "int": 
                minimum = np.min(feature_vals)
                maximum = np.max(feature_vals)
                step_size = (maximum - minimum)/self.total_step_num
                choices = list(set([round(step_num * step_size) for step_num in range(self.total_step_num)]))
                self.mutable_feature_choices[feature] = choices
            elif self.feature_types[feature] == "real":
                minimum = np.min(feature_vals)
                maximum = np.max(feature_vals)
                step_size = (maximum - minimum)/self.total_step_num
                choices = [step_num * step_size for step_num in range(self.total_step_num)]
                self.mutable_feature_choices[feature] = choices

    def attack(self, x):
        base_labels = self.victim.predict(x) # the base labels from which we start
        adv_samples = None
        for sample_idx in range(x.shape[0]):
            adv_sample = self.adv_best_first_search(x[sample_idx], base_labels[sample_idx], self.victim) 
            if adv_samples is None:
                adv_samples = np.expand_dims(adv_sample, 0)
            else:
                adv_sample = np.expand_dims(adv_sample, 0)
                adv_samples = np.append(adv_samples, adv_sample, 0)
        return adv_samples
            
    def adv_best_first_search(self, start, base_label, victim):
        open_nodes = [(self.node_eval_function(start, start, base_label, victim).item(), 0, start)] # the attached value of the start node should always be 0, but you never know...
        hq.heapify(open_nodes)
        closed_nodes = []
        best_node = open_nodes[0]
        current_iter = 0
        expaded_node_ctr = 1
        while len(open_nodes) > 0 and current_iter < self.max_iterations:
            node_eval_result, _ ,current_node = hq.heappop(open_nodes)
            if len(closed_nodes) == 0 or not any([np.all(np.equal(current_node, closed_node)) for closed_node in closed_nodes]): #check if this node config is already present in closed nodes
                closed_nodes.append(current_node)
            if node_eval_result <= self.target: # we are trying to maximize the goal function here
                return current_node
            if node_eval_result < best_node[0]:
                best_node = (node_eval_result, current_node)
            
            perturbations = self.generate_next_nodes(current_node)

            perturbation_eval_results = self.node_eval_function(perturbations, start, base_label, victim)

            new_queue_elements = []
            for idx in range(perturbations.shape[0]):
                new_queue_elements.append((perturbation_eval_results[idx].item(), expaded_node_ctr, perturbations[idx]))
                expaded_node_ctr += 1

            for new_queue_element in new_queue_elements:
                if not any([np.all(np.equal(new_queue_element[2], closed_node)) for closed_node in closed_nodes]):
                    hq.heappush(open_nodes, new_queue_element)

            if len(open_nodes) > self.queue_size:
                open_nodes.sort()
                open_nodes = open_nodes[0:self.queue_size] #remove the largest elements to cut queue down to size again

            current_iter += 1
            if self.profile is not None:
                self.profile.dump_stats(global_defaults["results_path"]+global_defaults["performance_log_file"])
        
        return best_node[1]

    def generate_next_nodes(self, start_node, check_constraint_fix_success= True):
        if check_constraint_fix_success:
            successful_constraint_fixes = 0
            failed_constraint_fixes = 0
        if len(start_node.shape) == 1:
            start_node = np.expand_dims(start_node, 0) 
        for idx_outer, feature in enumerate(self.mutable_feature_choices):
            current_value = start_node[0][self.feature_idxs[feature]]
            if current_value in self.mutable_feature_choices[feature]:
                possible_changes = self.mutable_feature_choices[feature].copy()
                possible_changes.remove(current_value)
            else:
                possible_changes = self.mutable_feature_choices[feature]

            for idx_inner, change in enumerate(possible_changes):
                perturbation = np.copy(start_node)
                perturbation[0][self.feature_idxs[feature]] = change

                timestamp = time.time()
                if self.constraint_correction: # run this locally, so we dont fix constraints unnessecarily. Dont know if that would do something.
                    if self.constraints_checker.check_constraints(start_node, perturbation)[0] == 0: # returns False if a constraint was violated
                        perturbation = self.constraint_fixer.fix(perturbation)
                        if check_constraint_fix_success:
                            if self.constraints_checker.check_constraints(start_node, perturbation)[0] == 0:
                                # print("WARNING: Constraint fix unsusccessful")
                                failed_constraint_fixes += 1
                            else:
                                successful_constraint_fixes += 1

                if idx_inner == 0 and idx_outer == 0:
                    perturbations = perturbation
                else:
                    perturbations = np.append(perturbations, perturbation, axis=0)

        if check_constraint_fix_success:
            if successful_constraint_fixes + failed_constraint_fixes > 0:
                constraint_fix_ratio = successful_constraint_fixes / (successful_constraint_fixes + failed_constraint_fixes)
                print(f"successful constraint fixes ratio: {constraint_fix_ratio}")
        return perturbations

    def node_eval_function(self, perturbations, start, start_label, victim):
        if len(perturbations.shape) == 1:
            perturbations = np.expand_dims(perturbations, 0)
        expanded_start_label = np.array([start_label]*perturbations.shape[0])
        victim_prediction = victim.predict(perturbations)

        flip_successes = np.not_equal(expanded_start_label, victim_prediction).astype(int)

        expanded_start = np.reshape(np.repeat(start, perturbations.shape[0]), perturbations.shape)
        distance = self.distance_metric.dist_func(expanded_start, perturbations, pairwise=True)

        metrics = np.subtract(flip_successes, distance) # this is likely very much in favor of the metric, sensitive to the gower initialization and may end up negative, but should work
        return metrics*(-1) # negate metrics so we can use out of the box min heap


AVAILABLE_ATTACKERS = {
    "dummy": Dummy_adv,
    "baseline": Baseline_adv
}
