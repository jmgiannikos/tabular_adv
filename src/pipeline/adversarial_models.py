from abc import ABC, abstractmethod
import numpy as np
from gower_distance import Gower_dist
from tabularbench.constraints.relation_constraint import BaseRelationConstraint, EqualConstraint
from tabularbench.constraints.constraints_fixer import ConstraintsFixer
from tabularbench.constraints.constraints_checker import ConstraintChecker
from tabularbench.constraints.pytorch_backend import PytorchBackend
from tabularbench.constraints.constraints_backend_executor import (
    ConstraintsExecutor,
)
import heapq as hq
import sklearn as sk
import time
from defaults import DEFAULTS as global_defaults
import wandb
import torch
from bounded_prio_queue import Bounded_Priority_Queue
import utils
import json
from defaults import Log_styles
from enum import Enum
from utils import find_unused_path
import os
from functools import partial
import matplotlib.pyplot as plt

class Adv_model(ABC):
    DEFAULTS = None
    def __init__(self, victim, constraints, metadata, log_path, log_style, *hyperparams):
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
class Dummy_adv(Adv_model):
    def __init__(self, victim, constraints, metadata, log_path, log_style, *hyperparams):
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

class Baseline_adv(Adv_model, sk.base.BaseEstimator):
    DEFAULTS = {
        "target": -500, # negative means this managed to flip the value (we negate for min heap purposes)
        "max_iter": 20, # maximum number of search steps before we abort and just take the best value seen so far
        "parameters": [False, 10, 50, 10, -500],
        "eps_exp": -10,
        "expand_type": "gridstep",
        "record_search_num": 1,
        "precision": 3,
    }

    def __init__(self, victim, constraints, metadata, log_path, log_style, constraint_correction=DEFAULTS["parameters"][0], queue_size=DEFAULTS["parameters"][1], total_step_num=DEFAULTS["parameters"][2], max_expand_nodes=DEFAULTS["parameters"][3], target=DEFAULTS["parameters"][4], expand_type=DEFAULTS["expand_type"]):
        hyperparams = [constraint_correction, queue_size, total_step_num, max_expand_nodes, target]
        super().__init__(victim, constraints, metadata, *hyperparams)
        self.features = metadata["feature"].tolist()
        self.expand_type = expand_type
        self.mutable_feature_choices = {}
        self.step_sizes = {}
        self.feature_idxs = {}
        self.feature_types = {}
        self.log_results = {}
        self.log_path = log_path
        self.log_style = log_style
        self.evalued_nodes_per_node = []

        for idx, feature in enumerate(self.features): # NOTE: really hope the order of the metadata feature list aligns with the order of x
            if metadata.query("feature == @feature")["mutable"].to_list()[0]:
                self.mutable_feature_choices[feature] = None
                self.feature_idxs[feature] = idx
                self.feature_types[feature] = metadata.query("feature == @feature")["type"].to_list()[0]

        self.constraint_correction = constraint_correction
        self.constraint_correction = False # NOTE: Temporary constraint correction disable for hyperparam search
        self.queue_size = queue_size
        self.total_step_num = total_step_num
        self.max_expand_nodes = max_expand_nodes

        if self.constraint_correction:
            guard_constraints = [constraint for constraint in constraints.relation_constraints if isinstance(constraint, BaseRelationConstraint) and not isinstance(constraint, EqualConstraint)]
            fix_constraints = [constraint for constraint in constraints.relation_constraints if isinstance(constraint, EqualConstraint)]
            self.constraint_fixer = ConstraintsFixer(guard_constraints=guard_constraints, fix_constraints=fix_constraints)
            self.constraints_checker = ConstraintChecker(constraints, 0)
        self.constraint_executors = []
        self.backend = PytorchBackend()
        for constraint in constraints.relation_constraints:
            self.constraint_executors.append(ConstraintsExecutor(
                constraint=constraint,
                backend=self.backend,
                feature_names=metadata["feature"].tolist()
            ))
        self.target = target
        self.max_iterations = self.DEFAULTS["max_iter"]

        self.local_log = log_style == Log_styles.BOTH or log_style == Log_styles.LOCAL
        self.online_log = log_style == Log_styles.BOTH or log_style == Log_styles.WANDB

    @staticmethod
    def HYPERPARAM_NAMES(): #names are in order
        return ["constraint_correction", "queue_size", "total_step_num", "max_expand_nodes", "target"]

    @staticmethod
    def TRAINABLE():
        return True
    
    @staticmethod
    def SEARCH_DIMS():
        return [[True, False], (1,100), (2,1000), (3,50), (-600, -150)] # NOTE: these are still chosen arbitrarily. May be subject to adjustment

    def fit(self, x, y): #training here just means storing the seen ranges in x so we can define reasonable step sizes. We also restrict ourselves to remain in distribution -> reasonable restriction???
        self.distance_metric = Gower_dist(x, self.metadata, dynamic=False) #set these to be dynamic so we dont have exploding gower distances
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
                self.step_sizes[feature] = max(round(step_size), 1)
            elif self.feature_types[feature] == "real":
                minimum = np.min(feature_vals)
                maximum = np.max(feature_vals)
                step_size = (maximum - minimum)/self.total_step_num
                choices = [step_num * step_size for step_num in range(self.total_step_num)]
                self.mutable_feature_choices[feature] = choices
                self.step_sizes[feature] = step_size
        return self

    def store_logs(self, final_log=False):
        obj_identifier = hex(id(self))[2:].upper()
        if not final_log:
            file_path = f"{self.log_path}[{obj_identifier}]search_logs_tmp.json"
            eval_nodes_per_node_fp = f"{self.log_path}[{obj_identifier}]eval_nodes_per_node_tmp.json"
            attacks_fp = f"{self.log_path}[{obj_identifier}]attacks_tmp.pt"
        else:
            file_path = f"{self.log_path}[{obj_identifier}]search_logs.json"
            eval_nodes_per_node_fp = f"{self.log_path}[{obj_identifier}]eval_nodes_per_node.json"
            attacks_fp = f"{self.log_path}[{obj_identifier}]attacks.pt"
            os.remove(f"{self.log_path}[{obj_identifier}]search_logs_tmp.json")
            os.remove(f"{self.log_path}[{obj_identifier}]eval_nodes_per_node_tmp.json")
            os.remove(f"{self.log_path}[{obj_identifier}]attacks_tmp.pt")

            if os.path.exists(file_path):
                file_path = find_unused_path(file_path, iterator=1)
            if os.path.exists(eval_nodes_per_node_fp):
                file_path = find_unused_path(eval_nodes_per_node_fp, iterator=1)
            if os.path.exists(attacks_fp):
                file_path = find_unused_path(attacks_fp, iterator=1)

        log_obj = json.dumps(self.log_results)
        with open(file_path, "w") as outfile:
            outfile.write(log_obj)
        eval_nodes_per_node_obj = json.dumps(self.evalued_nodes_per_node)
        with open(eval_nodes_per_node_fp, "w") as outfile:
            outfile.write(eval_nodes_per_node_obj)
        torch.save(self.attacks, attacks_fp)

    def wandb_log(self):
        adv_id = hex(id(self))[2:].upper()
        for result_idx in self.log_results.keys():
            for idx in range(len(self.log_results[result_idx]["step"])):
                wandb.log(
                data={
                    f"{adv_id}_search{result_idx}/prob_diff":self.log_results[result_idx]["prob_diff"],
                    f"{adv_id}_search{result_idx}/gower_dist": self.log_results[result_idx]["gower_dist"],
                    f"{adv_id}_search{result_idx}/constraint_loss": self.log_results[result_idx]["const_loss"],
                    f"{adv_id}_search{result_idx}/cost_function": self.log_results[result_idx]["cost_func"],
                    f"{adv_id}_search{result_idx}/step": self.log_results[result_idx]["step"],
                    f"{adv_id}_search{result_idx}/total_prob": self.log_results[result_idx]["total_node_prob"]
                }
            )

    def attack(self, x):
        base_proba = self.victim.predict_proba(x)[:, global_defaults["target_label"]] # the base label probabilities from which we start
        adv_samples = None
        for sample_idx in range(x.shape[0]):
            adv_sample = self.adv_best_first_search(x[sample_idx], base_proba[sample_idx], self.victim, collect_metrics=sample_idx<self.DEFAULTS["record_search_num"], search_idx=sample_idx) 
            if adv_samples is None:
                adv_samples = np.expand_dims(adv_sample, 0)
            else:
                adv_sample = np.expand_dims(adv_sample, 0)
                adv_samples = np.append(adv_samples, adv_sample, 0)
        return adv_samples
     
    # NOTE: some loss of fidelity seems to occur with the node_eval_result not perfectly aligning with -prob_diff/gower_dist. Currently believe tha this is due to a rounding error
    def collect_metrics(self, victim, search_idx, node_eval_result, current_iter, current_node, start, base_proba):
        # classifier_id = hex(id(self))[2:].upper()
        # expanded_start_proba = torch.from_numpy(victim.predict_proba(start)[:,global_defaults["target_label"]])
        victim_proba_prediction = torch.from_numpy(victim.predict_proba(current_node)[:,global_defaults["target_label"]])
        proba_change = torch.subtract(victim_proba_prediction, base_proba)
        prob_diff = proba_change
        gower_dist = self.distance_metric.dist_func(start, current_node, pairwise=True)
        running_loss = 0
        for constraint_executor in self.constraint_executors:
            running_loss += constraint_executor.execute(torch.from_numpy(np.expand_dims(current_node, axis=0))).sum()
        const_loss = running_loss

        local_log_results = {
                "prob_diff": prob_diff,
                "gower_dist": gower_dist,
                "const_loss": const_loss,
                "cost_func": node_eval_result,
                "step": current_iter,
                "total_node_prob": victim_proba_prediction
            }
        self.append_to_log_dict(search_idx, current_iter, local_log_results)

    def append_to_log_dict(self, search_idx, current_iter, new_results):
        if search_idx in self.log_results.keys():
            if current_iter not in self.log_results[search_idx]["step"]:
                for key in self.log_results[search_idx].keys():
                    if isinstance(new_results[key], torch.Tensor):
                        self.log_results[search_idx][key].append(new_results[key].item())
                    else:
                        self.log_results[search_idx][key].append(new_results[key])
            else:
                search_idx = search_idx + self.DEFAULTS["batch_size"]
                self.append_to_log_dict(search_idx, current_iter, new_results) # not as optimal because recursion, but looks elegant in code so whatever.
        else:
            self.log_results[search_idx] = {}
            for key in new_results.keys():
                if isinstance(new_results[key], torch.Tensor):
                    self.log_results[search_idx][key] = [new_results[key].item()]
                else:
                    self.log_results[search_idx][key] = [new_results[key]]

    def adv_best_first_search(self, start, base_proba, victim, collect_metrics=False, search_idx=0):
        open_nodes = [(self.node_eval_function(start, start, base_proba, victim).item(), 0, start)] # the attached value of the start node should always be 0, but you never know...
        hq.heapify(open_nodes)
        closed_nodes = []
        best_node = (open_nodes[0][0],open_nodes[0][2])
        current_iter = 0
        expaded_node_ctr = 1
        while len(open_nodes) > 0 and current_iter < self.max_iterations:
            node_eval_result, _ ,current_node = hq.heappop(open_nodes)

            if collect_metrics:
                self.collect_metrics(victim=victim, search_idx=search_idx, node_eval_result=node_eval_result, current_iter=current_iter, current_node=current_node, start=start, base_proba=base_proba)

            if len(closed_nodes) == 0 or not any([np.all(np.equal(current_node, closed_node)) for closed_node in closed_nodes]): #check if this node config is already present in closed nodes
                closed_nodes.append(current_node)
            if node_eval_result <= self.target: # we are trying to maximize the goal function here
                return current_node
            if node_eval_result < best_node[0]:
                best_node = (node_eval_result, current_node)
            
            if self.expand_type == "fixgrid":
                perturbations = self.generate_next_nodes_fixgrid(current_node)
            else:
                perturbations = self.generate_next_nodes_gridstep(current_node)

            perturbation_eval_results = self.node_eval_function(perturbations, start, base_proba, victim)

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
        
        return best_node[1]

    def generate_next_nodes_gridstep(self, start_node):
        if len(start_node.shape) == 1:
            start_node = np.expand_dims(start_node, 0)
        for idx_outer, feature in enumerate(self.mutable_feature_choices):
            feature_idx = self.feature_idxs[feature]
            if self.feature_types[feature] == "cat":
                current_value = start_node[0][self.feature_idxs[feature]]
                if current_value in self.mutable_feature_choices[feature]:
                    possible_changes = self.mutable_feature_choices[feature].copy()
                    possible_changes.remove(current_value)
                else:
                    possible_changes = self.mutable_feature_choices[feature]
                possible_changes = np.array([possible_changes])
            else:
                if self.feature_types[feature] == "int":
                    border = int(self.step_sizes[feature]*self.max_expand_nodes/2)
                    start = -border
                    end = border
                    deltas = np.arange(start=start, stop=end, step=self.step_sizes[feature], dtype=int)
                else:
                    border = self.step_sizes[feature]*self.max_expand_nodes/2
                    start = -border
                    end = border
                    deltas = np.linspace(start=start, stop=end, num=self.max_expand_nodes)

                deltas = deltas[deltas != 0]
                possible_changes = deltas + start_node[0,feature_idx] #NOTE: this assumes we only ever pass this function a single node to expand...
                possible_changes = np.expand_dims(possible_changes, axis=0)

            perturbation = np.copy(start_node)
            perturbation = np.repeat(perturbation, possible_changes.shape[1], axis=0)
            perturbation[:,feature_idx] = possible_changes

            if idx_outer == 0:
                perturbations = perturbation
            else:
                perturbations = np.append(perturbations, perturbation, axis=0)

        return perturbations
    
    def generate_next_nodes_fixgrid(self, start_node, check_constraint_fix_success= True):
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

    def node_eval_function(self, perturbations, start, start_proba, victim):
        if len(perturbations.shape) == 1:
            perturbations = np.expand_dims(perturbations, 0)
        expanded_start_proba = np.array([start_proba]*perturbations.shape[0])
        victim_proba_prediction = victim.predict_proba(perturbations)[:,global_defaults["target_label"]]

        proba_change = np.subtract(victim_proba_prediction, expanded_start_proba)

        expanded_start = np.reshape(np.repeat(start, perturbations.shape[0]), perturbations.shape)
        distance = self.distance_metric.dist_func(expanded_start, perturbations, pairwise=True)

        safe_dist = np.max(np.append(np.expand_dims(distance, axis=1), np.reshape(np.array([10**float(self.DEFAULTS["eps_exp"])]*distance.shape[0]), (distance.shape[0], 1)), axis=1), axis=1, keepdims=False)
        metrics = np.divide(proba_change, safe_dist)
        return metrics*(-1) # negate metrics so we can use out of the box min heap


class Accel_Baseline_adv(Baseline_adv, sk.base.BaseEstimator):
    @staticmethod
    def fix_round(tensor, threshhold=0.005):
        min_val = 10**float(Accel_Baseline_adv.DEFAULTS["eps_exp"])
        return torch.where(torch.logical_or(tensor>threshhold, tensor<=0), tensor, min_val)

    @staticmethod
    def exponential(tensor, exponent=3):
        return torch.pow(tensor, exponent)
    
    @staticmethod
    def olrelu(tensor, offset=0.005, slope=0.001):
        relu = torch.nn.LeakyReLU(slope)
        return torch.add(relu(torch.add(tensor, -offset)), offset*slope)
    
    @staticmethod
    def noround(tensor):
        return tensor
    
    ACT_FUNC_MAP = {
        "fix_round": fix_round.__func__,
        "exponential": exponential.__func__,
        "offset_leaky_relu": olrelu.__func__,
        "identity": noround.__func__
    }

    DEFAULTS = {
        "target": 0.6, # negative means this managed to flip the value (we negate for min heap purposes)
        "max_iter": 20, # maximum number of search steps before we abort and just take the best value seen so far
        "parameters": [False, 1, 5000, 3000, 0.6],
        "eps_exp": -10,
        "expand_type": "gridstep",
        "record_search_num": 1000,
        "precision": 3,
        "batch_size": 1,
        "act_func": "identity", # this is an identifier and not just the function so we can easily dump this config (in a human readable form)
        "chatty": True,
        "constraint": 0.006
    }

    def __init__(self, victim, constraints, metadata, log_path, log_style, constraint_correction=DEFAULTS["parameters"][0], queue_size=DEFAULTS["parameters"][1], total_step_num=DEFAULTS["parameters"][2], max_expand_nodes=DEFAULTS["parameters"][3], target=DEFAULTS["parameters"][4], expand_type=DEFAULTS["expand_type"]):
        super().__init__(victim, constraints, metadata, log_path, log_style, constraint_correction, queue_size, total_step_num, max_expand_nodes, target, expand_type)
        self.significant_decimals = {}
        self.activation_function = self.ACT_FUNC_MAP[self.DEFAULTS["act_func"]]
        self.evalued_nodes_per_node = []
        self.attacks = None

    # TODO: translate from and to cat values (needed for other datasets)
    def fit(self, x, y): # training here means initializing gower dist for internal use and deriving step sizes
        self.distance_metric = Gower_dist(x, self.metadata, dynamic=False) #set these to be dynamic so we dont have exploding gower distances
        for feature in self.mutable_feature_choices.keys():
            feature_idx = self.feature_idxs[feature]
            feature_vals = x[:,feature_idx]
            if self.feature_types[feature] == "cat":
                self.mutable_feature_choices[feature] = list(set(feature_vals.tolist()))
            elif self.feature_types[feature] == "int": 
                minimum = np.min(feature_vals)
                maximum = np.max(feature_vals)
                step_size = (maximum - minimum)/self.total_step_num
                choices = list(set([round(step_num * step_size) for step_num in range(self.total_step_num)]))
                self.mutable_feature_choices[feature] = choices
                self.step_sizes[feature] = max(round(step_size), 1)
            elif self.feature_types[feature] == "real":
                minimum = np.min(feature_vals)
                maximum = np.max(feature_vals)
                step_size = (maximum - minimum)/self.total_step_num
                choices = [step_num * step_size for step_num in range(self.total_step_num)]
                self.mutable_feature_choices[feature] = choices
                self.step_sizes[feature] = step_size
                self.significant_decimals[feature] = self.get_significant_decimals(x, feature)
        self.perturbation_tensor = self.get_perturbation_tensor(x)
        return self

    def attack(self, x):
        if not  self.DEFAULTS["expand_type"] == "fixgrid":
            x = self.translate_categoricals(x, to_local=True)
        if self.DEFAULTS["chatty"]:
            identifier = hex(id(self))[2:].upper()
        base_proba = self.victim.predict_proba(x)[:, global_defaults["target_label"]] # the base label probabilities from which we start
        batch_num = int(x.shape[0]/self.DEFAULTS["batch_size"])+1
        batch_idx = 0
        adv_samples = []
        if self.DEFAULTS["record_search_num"] >= x.shape[0]:
            while batch_idx < batch_num:
                min_idx = batch_idx*self.DEFAULTS["batch_size"]
                max_idx = batch_idx+1*self.DEFAULTS["batch_size"]
                batch_x = x[min_idx:max_idx]
                batch_base_proba = base_proba[min_idx:max_idx]
                adv_sample_batch = self.accel_adv_best_first_search(batch_x, batch_base_proba, self.victim, collect_metrics=True)
                if self.local_log:
                    self.store_logs()
                if self.online_log:
                    self.wandb_log()
                adv_samples.extend(adv_sample_batch)
                if self.DEFAULTS["chatty"]:
                    print(f"[{identifier}]batch{batch_idx}")
                batch_idx += 1
            if self.local_log:
                self.store_logs(final_log=True)
        else:
            if self.DEFAULTS["record_search_num"] > 0:
                num_record_search = self.DEFAULTS["record_search_num"]
                record_pre_batch_x = x[:num_record_search]
                record_pre_batch_probas = base_proba[:num_record_search]
                x = x[num_record_search:]
                base_proba = base_proba[num_record_search:]
                adv_sample_batch = self.accel_adv_best_first_search(record_pre_batch_x, record_pre_batch_probas, self.victim, collect_metrics=True)
                if self.local_log:
                    self.store_logs(final_log=True)
                if self.online_log:
                    self.wandb_log()
                adv_samples.extend(adv_sample_batch)
                if self.DEFAULTS["chatty"]:
                    print(f"[{identifier}]logbatch")

            while batch_idx < batch_num:
                min_idx = batch_idx*self.DEFAULTS["batch_size"]
                max_idx = batch_idx+1*self.DEFAULTS["batch_size"]
                batch_x = x[min_idx:max_idx]
                batch_base_proba = base_proba[min_idx:max_idx]
                adv_sample_batch = self.accel_adv_best_first_search(batch_x, batch_base_proba, self.victim, collect_metrics=False)
                adv_samples.extend(adv_sample_batch)
                if self.DEFAULTS["chatty"]:
                    print(f"[{identifier}]batch{batch_idx}")
                batch_idx += 1
        return torch.stack(list(map(lambda x: x[1], adv_samples)), dim=0).numpy()

        # NOTE: this is far from perfect. Looking at the min value of the dataset and choosing based on that is functional, but it would be better if we looked at the point distances along any given feature
    
    def get_significant_decimals(self, x, feature):
        feature_idx = self.feature_idxs[feature]
        feature_vals = x[:,feature_idx]
        min_val = np.min(np.abs(feature_vals))
        if min_val == 0:
            significant_decimals = -self.DEFAULTS["eps_exp"]
        else:
            log_val = int(np.log10(min_val))
            significant_decimals = -(log_val - self.DEFAULTS["precision"])
            if significant_decimals >= self.DEFAULTS["precision"]:
                significant_decimals = self.DEFAULTS["precision"]
        return significant_decimals

    def get_perturbation_tensor(self, x):
        if self.expand_type == "fixgrid":
            index_list = []
            perturbation_list = []
            for feature in self.feature_idxs.keys():
                feature_idx = self.feature_idxs[feature]
                for choice in self.mutable_feature_choices[feature]:
                    if self.feature_types[feature] == "real":
                        choice = round(choice, self.significant_decimals[feature])
                    index_list.append(feature_idx)
                    perturbation_list.append(choice)
            self.index_tensor = torch.unsqueeze(torch.LongTensor([index_list]).to(global_defaults["device"]), dim=2)
            perturbation_tensor = torch.unsqueeze(torch.DoubleTensor([perturbation_list]).to(global_defaults["device"]), dim=2)
            return perturbation_tensor
        else:
            num_features = x.shape[1]
            self.feature_category_to_num_map = {}
            self.feature_num_to_category_map = {}
            self.feature_category_modulus = {}
            for iter, feature in enumerate(self.feature_idxs.keys()):
                feature_idx = self.feature_idxs[feature] # already only mutable features 
                if self.feature_types[feature] == "int":
                    border = int(self.step_sizes[feature]*self.max_expand_nodes/2)
                    start = -border
                    end = border
                    deltas = np.arange(start=start, stop=end, step=self.step_sizes[feature], dtype=int)
                elif self.feature_types[feature] == "real":
                    border = self.step_sizes[feature]*self.max_expand_nodes/2
                    start = -border
                    end = border
                    deltas = np.linspace(start=start, stop=end, num=self.max_expand_nodes)    
                elif self.feature_types[feature] == "cat":
                    self.feature_category_to_num_map[feature] = {}
                    self.feature_num_to_category_map[feature] = {}
                    for num_val, category in enumerate(self.mutable_feature_choices[feature]):
                        self.feature_category_to_num_map[feature][category] = num_val
                        self.feature_num_to_category_map[feature][num_val] = category
                    self.feature_category_modulus[feature] = len(self.mutable_feature_choices[feature])+1
                    
                    deltas = np.arange(start=1,stop=self.feature_category_modulus[feature], step=1)

                new_chunk = np.zeros((len(deltas), num_features))
                new_chunk[:,feature_idx] = deltas
                if iter == 0:
                    perturbations = new_chunk
                else:
                    perturbations = np.append(perturbations, new_chunk, axis=0)
            perturbation_tensor = torch.from_numpy(perturbations)
        return perturbation_tensor

    def accel_adv_best_first_search(self, startnodes, base_probas, victim, collect_metrics=False):
        if isinstance(startnodes, np.ndarray):
            startnodes = torch.from_numpy(startnodes)
        elif isinstance(startnodes, torch.Tensor):
            startnodes = startnodes
        else:
            startnodes = torch.Tensor(startnodes).to(global_defaults["device"])

        startnodes = self.round_reals(startnodes)
        idx_map = {}
        for idx in range(startnodes.shape[0]):
            idx_map[idx] = (idx, idx+1)
        start_node_evals = self.accel_eval_func(startnodes, startnodes, base_probas, victim, idx_map)

        open_nodes = {}
        closed_nodes = []
        best_nodes = []
        for sample_idx in range(startnodes.shape[0]):
            node_queue = Bounded_Priority_Queue(self.queue_size, [(start_node_evals[sample_idx], startnodes[sample_idx])])
            open_nodes[sample_idx] = node_queue
            closed_nodes.append([])
            best_nodes.append((node_queue.get(0)[0],node_queue.get(0)[1]))
        current_iter = 0
        expaded_node_ctrs = [1]*startnodes.shape[0]

        self.evalued_nodes = 0

        # quick and dirty eval code
        self.prev_best_pert = startnodes[0]

        while len(open_nodes.keys()) > 0 and current_iter < self.max_iterations:
            current_nodes = []
            for idx, node_queue in [(key, open_nodes[key]) for key in open_nodes.keys()]:
                node_eval_result, current_node = node_queue.pop()
                if collect_metrics:
                    self.collect_metrics(victim=victim, search_idx=idx, node_eval_result=node_eval_result, current_iter=current_iter, current_node=current_node, start=startnodes[idx], base_proba=base_probas[idx])

                target_prob = (-node_eval_result * self.distance_metric.dist_func(startnodes[idx], current_node, True).item())+base_probas[idx]
                if target_prob >= self.target: # we are trying to maximize the goal function here
                    best_nodes[idx] = (node_eval_result, current_node)
                    open_nodes.pop(idx) # removes queue from open nodes dictionary stopping that particular search
                else:
                    if node_eval_result < best_nodes[idx][0]:
                        best_nodes[idx] = (node_eval_result, current_node)
                    current_nodes.append(current_node)
                    # assert not any(list(map(lambda node: torch.equal(node, current_node), closed_nodes[idx]))) #NOTE: mlp cls causes this to fail
                    closed_nodes[idx].append(current_node)

            if len(open_nodes.keys()) <= 0: #this may occur if all explored nodes were found to be satisfactory. In that case we abort search immediately
                break
            current_nodes = torch.stack(current_nodes, dim=0)

            if self.expand_type == "gridstep":
                expanded_nodes = self.accel_gridstep_node_expand(current_nodes)
            elif self.expand_type == "fixgrid":
                expanded_nodes = self.accel_fixgrid_node_expand(current_nodes)

            expanded_nodes, idx_map = self.accel_check_visited(expanded_nodes, open_nodes, closed_nodes)
            eval_results = self.accel_eval_func(expanded_nodes, startnodes, base_probas, victim, idx_map)

            assert expanded_nodes.shape[0] == eval_results.shape[0]
            for idx_mapping_key in idx_map.keys(): #TODO: may want to do this with map function? Could be faster.
                idx_mapping = idx_map[idx_mapping_key]
                loc_eval_results = eval_results[idx_mapping[0]:idx_mapping[1]]
                is_viable_map = torch.logical_not(torch.isinf(loc_eval_results))
                loc_eval_results = loc_eval_results[is_viable_map]
                loc_expanded_nodes = expanded_nodes[idx_mapping[0]:idx_mapping[1]]
                loc_expanded_nodes = loc_expanded_nodes[is_viable_map]
                if len(loc_expanded_nodes) > 0:
                    new_nodes = [(loc_eval_results[idx], loc_expanded_nodes[idx]) for idx in range(loc_eval_results.shape[0])]
                    open_nodes[idx_mapping_key].push(new_nodes) 
                    expaded_node_ctrs[idx_mapping_key] += loc_eval_results.shape[0]
            
            open_nodes_list = list(open_nodes.keys()) # export to list, because dict changes (nessecarily) and that breaks .keys()
            for key in open_nodes_list: # abort condition: Queue empty
                if open_nodes[key].is_empty():
                    open_nodes.pop(key) 
            current_iter += 1

        self.evalued_nodes_per_node.append(self.evalued_nodes/startnodes.shape[0])
        if not self.DEFAULTS["expand_type"] == "fixgrid":
            best_nodes = self.translate_categoricals(best_nodes, to_local=False)

        # log local attacks
        local_attacks = torch.stack([startnodes, torch.stack([best_node[1] for best_node in best_nodes], dim=0)], dim=2)
        if self.attacks is None:
            self.attacks = local_attacks
        else:
            self.attacks = torch.cat([self.attacks, local_attacks], dim=0)
        return best_nodes

    def round_reals(self, samples):
        for feature in self.significant_decimals.keys():
            feature_idx = self.feature_idxs[feature]
            samples[:,feature_idx] = torch.round(samples[:,feature_idx], decimals=self.significant_decimals[feature])
        return samples 
    
    def modulo_cats(self, samples):
        for feature in self.feature_category_modulus.keys():
            feature_idx = self.feature_idxs[feature]
            samples[:,feature_idx] = torch.remainder(samples[:,feature_idx], self.feature_category_modulus[feature])
        return samples

    def accel_eval_func(self, perturbations, start_nodes, start_probas, victim, idx_mappings):

        if perturbations.shape == start_nodes.shape and torch.all(perturbations == start_nodes):
            return torch.DoubleTensor([0]*len(perturbations))
        
        expanded_start_probas = []
        expanded_start_nodes = []
        for key in idx_mappings.keys():
            num_entries = idx_mappings[key][1] - idx_mappings[key][0]
            expanded_start_probas.extend([start_probas[key]]*num_entries)
            expanded_start_nodes.extend([start_nodes[key]]*num_entries)
        expanded_start_nodes = torch.stack(expanded_start_nodes, dim=0)
        expanded_start_probas = torch.Tensor(expanded_start_probas).to(global_defaults["device"])

        distances = self.distance_metric.dist_func(expanded_start_nodes, perturbations, pairwise=True)
        if self.DEFAULTS["constraint"] is not None:
            viable_sample_map = distances <= self.DEFAULTS["constraint"]
        else:
            viable_sample_map = torch.BoolTensor([True]*distances.shape[0])

        self.evalued_nodes += perturbations[viable_sample_map].shape[0]
        if self.DEFAULTS["expand_type"] == "fixgrid":
            victim_proba_prediction = torch.from_numpy(victim.predict_proba(perturbations[viable_sample_map])[:,global_defaults["target_label"]])
        else:
            victim_proba_prediction = torch.from_numpy(victim.predict_proba(self.translate_categoricals(perturbations[viable_sample_map], False))[:,global_defaults["target_label"]])

        proba_change = self.activation_function(torch.subtract(victim_proba_prediction, expanded_start_probas[viable_sample_map]))

        safe_div_floor = torch.Tensor([10**float(self.DEFAULTS["eps_exp"])]).to(global_defaults["device"]).expand(distances[viable_sample_map].shape[0])
        safe_div_dist, _ = torch.max(torch.stack([distances[viable_sample_map], safe_div_floor], dim=1), dim=1, keepdim=False)
        metrics = torch.DoubleTensor([-torch.inf]*distances.shape[0])
        metrics[viable_sample_map] = torch.divide(proba_change, safe_div_dist)
        
        # quick and dirty eval code
        best_pert = torch.argmin(torch.neg(metrics)).item()
        idx_of_best_pert = torch.where(self.prev_best_pert != perturbations[best_pert])[0]
        feat_range = torch.where(perturbations[:,idx_of_best_pert] != self.prev_best_pert[idx_of_best_pert])[0].tolist()
        self.prev_best_pert = perturbations[best_pert]
        feat_range_min = min(feat_range)
        feat_range_max = max(feat_range)
        prob_diffs = torch.subtract(torch.from_numpy(victim.predict_proba(self.translate_categoricals(perturbations[feat_range_min:feat_range_max], False))[:,global_defaults["target_label"]]), expanded_start_probas[feat_range_min:feat_range_max])
        dists = distances[feat_range_min:feat_range_max]
        zero_idx = torch.argmin(dists)
        for idx in range(zero_idx):
            dists[idx] = -dists[idx]
        metr = torch.neg(metrics)[feat_range_min:feat_range_max]
        if torch.max(prob_diffs) != prob_diffs[torch.argmin(metr)]:
            fig, ax = plt.subplots(1, 1)
            ax.plot(dists, metr, label="objective function behavior")
            ax.set_xlabel("distance")
            ax.set_ylabel("objective function")
            plt.show()
            #plt.close("all")

            fig2, ax2 = plt.subplots(1, 1)
            ax2.plot(dists, prob_diffs, label="probability difference behavior")
            ax2.set_xlabel("distance")
            ax2.set_ylabel("probability difference")
            plt.show()
            plt.close("all")

        return torch.neg(metrics) # negate for overall consistency (minimize cost)

    def accel_gridstep_node_expand(self, nodes):
        nodes_shape = nodes.shape
        perts_shape = self.perturbation_tensor.shape
        target_shape = (nodes_shape[0], perts_shape[0], nodes_shape[1])
        nodes = torch.unsqueeze(nodes, dim=1)
        perturbations = torch.unsqueeze(self.perturbation_tensor, dim=0)
        nodes = nodes.expand(target_shape)
        perturbations = perturbations.expand(target_shape)
        expanded_nodes = torch.add(nodes, perturbations)
        expanded_nodes = torch.flatten(expanded_nodes, start_dim=0, end_dim=1)
        expanded_nodes = self.round_reals(expanded_nodes)
        expanded_nodes = self.modulo_cats(expanded_nodes)
        return expanded_nodes
    
    def accel_fixgrid_node_expand(self, nodes):
        desired_shape = (nodes.shape[0], self.perturbation_tensor.shape[1], nodes.shape[1])
        nodes = torch.unsqueeze(nodes, dim=1)
        nodes = nodes.expand(desired_shape)
        perturbations = self.perturbation_tensor.expand(desired_shape)
        feature_indices = self.index_tensor.expand(desired_shape)
        nodes = torch.scatter(input=nodes, dim=2, index=feature_indices, src=perturbations)
        nodes = torch.flatten(nodes, start_dim=0, end_dim=1)
        return nodes

    def accel_check_visited(self, expanded_nodes, open_nodes, closed_nodes):# NOTE: must return idx mapping that maps areas of the return tensor to the key of the open nodes dictionary!
        if self.expand_type == "gridstep":
            chunk_size = self.perturbation_tensor.shape[0]
        elif self.expand_type == "fixgrid":
            chunk_size = self.perturbation_tensor.shape[1]
        running_idx = 0
        idx_mapping = {}
        chunks = []
        for local_idx, map_idx in enumerate(open_nodes.keys()):
            chunk = expanded_nodes[chunk_size*local_idx:chunk_size*(local_idx+1)]
            open_node_queue = open_nodes[map_idx]
            chunk = chunk[open_node_queue.is_not_in(chunk)]
            chunk = chunk[utils.is_not_in(chunk, closed_nodes[map_idx])]
            idx_mapping[map_idx] = (running_idx, running_idx+chunk.shape[0])
            running_idx += chunk.shape[0]
            chunks.append(chunk)
        return torch.cat(chunks, dim=0), idx_mapping
    
    def translate_categoricals(self, samples, to_local=False):
        if "cat" in self.feature_types.values():
            if isinstance(samples, list) and len(samples[0]) == 2:
                node_samples = torch.stack([sample[1] for sample in samples])
            else:
                node_samples = samples
            if not isinstance(node_samples, np.ndarray):
                if isinstance(node_samples, list):
                    node_samples = np.array(node_samples)
                else:
                    node_samples = node_samples.numpy()
            if to_local:
                for feature in self.feature_num_to_category_map.keys():
                    feature_idx = self.feature_idxs[feature]
                    trans_func = partial(self.translate_from_cats_func, feature)
                    func = np.vectorize(trans_func, otypes="f")
                    node_samples[:,feature_idx] = func(node_samples[:,feature_idx])
            else:
                for feature in self.feature_num_to_category_map.keys():
                    feature_idx = self.feature_idxs[feature]
                    trans_func = partial(self.translate_to_cats_func, feature)
                    func = np.vectorize(trans_func, otypes="f")
                    node_samples[:,feature_idx] = func(node_samples[:,feature_idx])
            if isinstance(samples, list):
                samples = [(samples[idx][0], torch.from_numpy(node_samples[idx])) for idx in range(len(samples))]
            else:
                samples = torch.from_numpy(node_samples)
            return samples
        else:
            return samples

    def translate_to_cats_func(self, feature, val):
        if val in self.feature_num_to_category_map[feature].keys():
            return self.feature_num_to_category_map[feature][val]
        else:
            return list(self.feature_num_to_category_map[feature].values())[0]

    def translate_from_cats_func(self, feature, val):
        if val in self.feature_category_to_num_map[feature].keys():
            return self.feature_category_to_num_map[feature][val]
        else:
            return list(self.feature_category_to_num_map[feature].values())[0]


class Accel_adaptive_search(Accel_Baseline_adv, sk.base.BaseEstimator):
    @staticmethod
    def fix_round(tensor, threshhold=0.005):
        min_val = 10**float(Accel_Baseline_adv.DEFAULTS["eps_exp"])
        return torch.where(torch.logical_or(tensor>threshhold, tensor<=0), tensor, min_val)

    @staticmethod
    def exponential(tensor, exponent=3):
        return torch.pow(tensor, exponent)
    
    @staticmethod
    def olrelu(tensor, offset=0.005, slope=0.001):
        relu = torch.nn.LeakyReLU(slope)
        return torch.add(relu(torch.add(tensor, -offset)), offset*slope)
    
    @staticmethod
    def noround(tensor):
        return tensor

    ACT_FUNC_MAP = {
        "fix_round": fix_round.__func__,
        "exponential": exponential.__func__,
        "offset_leaky_relu": olrelu.__func__,
        "identity": noround.__func__
    }

    DEFAULTS = {
        "target": 0.6, # negative means this managed to flip the value (we negate for min heap purposes)
        "max_iter": 20, # maximum number of search steps before we abort and just take the best value seen so far
        "parameters": [False, 10, 1000, 6, 0.6],
        "eps_exp": -10,
        "record_search_num": 1000,
        "precision": 3,
        "batch_size": 1000,
        "act_func": "fix_round", # this is an identifier and not just the function so we can easily dump this config (in a human readable form)
        "chatty": True,
        "min_change_threshhold": 0.01,
        "step_scaling": 2, #must be whole number or else the scaling for integers breaks
        "termination_criterion": "improve",
        "termination_thershhold": 0,
        "termination_window_size": 4,
        "max_step_ratios": 0.5,
        "near_offset_ratio": 0.1, #ratio of the initial step close to zero. If this step has highest rating steps are scaled down
        "constraint": 0.019,
        "expand_type": "adaptive",
        "adjust_style": "beststep"
    }

    def __init__(self, victim, constraints, metadata, log_path, log_style, constraint_correction=DEFAULTS["parameters"][0], queue_size=DEFAULTS["parameters"][1], total_step_num=DEFAULTS["parameters"][2], max_expand_nodes=DEFAULTS["parameters"][3], target=DEFAULTS["parameters"][4], expand_type=None):
        super().__init__(victim, constraints, metadata, log_path, log_style, constraint_correction, queue_size, total_step_num, max_expand_nodes, target, expand_type)
        self.maximums = {}

    def fit(self, x, y): # training here means initializing gower dist for internal use and deriving step sizes
        self.distance_metric = Gower_dist(x, self.metadata, dynamic=False) #set these to be dynamic so we dont have exploding gower distances
        for feature in self.mutable_feature_choices.keys():
            feature_idx = self.feature_idxs[feature]
            feature_vals = x[:,feature_idx]
            if self.feature_types[feature] == "cat":
                self.mutable_feature_choices[feature] = list(set(feature_vals.tolist()))
            elif self.feature_types[feature] == "int": 
                minimum = np.min(feature_vals)
                maximum = np.max(feature_vals)
                step_size = (maximum - minimum)/self.total_step_num
                choices = list(set([round(step_num * step_size) for step_num in range(self.total_step_num)]))
                self.mutable_feature_choices[feature] = choices
                self.step_sizes[feature] = max(round(step_size), 1)
                self.maximums[feature] = max((abs(maximum), abs(minimum)))
            elif self.feature_types[feature] == "real":
                minimum = np.min(feature_vals)
                maximum = np.max(feature_vals)
                step_size = (maximum - minimum)/self.total_step_num
                choices = [step_num * step_size for step_num in range(self.total_step_num)]
                self.mutable_feature_choices[feature] = choices
                self.step_sizes[feature] = step_size
                self.significant_decimals[feature] = self.get_significant_decimals(x, feature)
                self.maximums[feature] = max((abs(maximum), abs(minimum)))
        self.perturbation_tensor, self.max_pert_values = self.get_perturbation_tensor(x)
        return self

    def accel_adv_best_first_search(self, startnodes, base_probas, victim, collect_metrics=False):
        if isinstance(startnodes, np.ndarray):
            startnodes = torch.from_numpy(startnodes)
        elif isinstance(startnodes, torch.Tensor):
            startnodes = startnodes
        else:
            startnodes = torch.Tensor(startnodes).to(global_defaults["device"])

        startnodes = self.round_reals(startnodes)
        start_node_evals = self.accel_adaptive_eval_func(startnodes, startnodes, base_probas, victim)

        is_not_cat_list = []
        for feature_name in self.feature_types.keys(): # only append for mutable features
            if "cat" != self.feature_types[feature_name]:
                is_not_cat_list.extend([True]*self.max_expand_nodes)
            else:
                is_not_cat_list.extend([False]*len(self.mutable_feature_choices[feature_name]))
        self.is_not_cat_tensor = torch.BoolTensor(is_not_cat_list)
        self.is_not_cat_tensor = torch.unsqueeze(self.is_not_cat_tensor, dim=0).expand((startnodes.shape[0], self.is_not_cat_tensor.shape[0]))
        self.is_not_cat_tensor = torch.flatten(self.is_not_cat_tensor)

        self.num_entry_dict = {}
        self.node_indexing_list = []
        for idx in range(startnodes.shape[0]):
            self.node_indexing_list.extend([idx]*self.perturbation_tensor.shape[0])
            self.num_entry_dict[idx] = self.perturbation_tensor.shape[0]
        
        self.effective_perturbation_tensor = torch.flatten(torch.unsqueeze_copy(self.perturbation_tensor, dim=0).expand((startnodes.shape[0], self.perturbation_tensor.shape[0], self.perturbation_tensor.shape[1])), start_dim=0, end_dim=1)
        self.effective_max_pert_values = torch.flatten(torch.unsqueeze_copy(self.max_pert_values, dim=0).expand((startnodes.shape[0], self.max_pert_values.shape[0], self.max_pert_values.shape[1])), start_dim=0, end_dim=1)

        open_nodes = {}
        closed_nodes = []
        best_nodes = []
        if self.DEFAULTS["termination_criterion"] != "fix":
            self.termination_windows = []
        for sample_idx in range(startnodes.shape[0]):
            node_queue = Bounded_Priority_Queue(self.queue_size, [(start_node_evals[sample_idx], startnodes[sample_idx])])
            open_nodes[sample_idx] = node_queue
            closed_nodes.append([])
            best_nodes.append((node_queue.get(0)[0],node_queue.get(0)[1]))
            if self.DEFAULTS["termination_criterion"] != "fix":
                self.termination_windows.append([])
            
        current_iter = 0
        expaded_node_ctrs = [1]*startnodes.shape[0]
        self.evalued_nodes = 0

        while len(open_nodes.keys()) > 0 and current_iter <= 20:
            current_nodes = []
            terminated_idx_list = []
            for idx, node_queue in [(key, open_nodes[key]) for key in open_nodes.keys()]:
                node_eval_result, current_node = node_queue.pop()
                if collect_metrics:
                    self.collect_metrics(victim=victim, search_idx=idx, node_eval_result=node_eval_result, current_iter=current_iter, current_node=current_node, start=startnodes[idx], base_proba=base_probas[idx])

                best_nodes, open_nodes, terminated = self.check_termination(node_eval_result, current_node, idx, best_nodes, open_nodes, startnodes, base_probas, idx)
                if not terminated:
                    if node_eval_result < best_nodes[idx][0]:
                        best_nodes[idx] = (node_eval_result, current_node)
                    current_nodes.append(current_node)
                    #assert not any(list(map(lambda node: torch.equal(node, current_node), closed_nodes[idx])))
                    closed_nodes[idx].append(current_node)
                else:
                    terminated_idx_list.append(idx)
            if len(open_nodes.keys()) <= 0: #this may occur if all explored nodes were found to be satisfactory. In that case we abort search immediately
                break
            current_nodes = torch.stack(current_nodes, dim=0)

            expanded_nodes = self.accel_adaptive_node_expand(current_nodes)

            eval_results = self.accel_adaptive_eval_func(expanded_nodes, startnodes, base_probas, victim)
            self.update_perturbation_tensor(eval_results)
            expanded_nodes, eval_results, idx_map = self.adaptive_check_visited(expanded_nodes, eval_results, open_nodes, closed_nodes)

            assert expanded_nodes.shape[0] == eval_results.shape[0]
            for idx_mapping_key in idx_map.keys(): #TODO: may want to do this with map function? Could be faster.
                idx_mapping = idx_map[idx_mapping_key]
                loc_eval_results = eval_results[idx_mapping[0]:idx_mapping[1]]
                viable_sample_map = torch.logical_not(torch.isinf(loc_eval_results))
                loc_eval_results = loc_eval_results[viable_sample_map]
                loc_expanded_nodes = expanded_nodes[idx_mapping[0]:idx_mapping[1]]
                loc_expanded_nodes = loc_expanded_nodes[viable_sample_map]
                if len(loc_eval_results) > 0:
                    new_nodes = [(loc_eval_results[idx], loc_expanded_nodes[idx]) for idx in range(loc_eval_results.shape[0])]
                    open_nodes[idx_mapping_key].push(new_nodes) 
                    expaded_node_ctrs[idx_mapping_key] += loc_eval_results.shape[0]
            
            open_nodes_key_list = list(open_nodes.keys())
            for key in open_nodes_key_list: # abort condition: Queue empty
                if open_nodes[key].is_empty():
                    open_nodes.pop(key) 
                    pruned_entry_map = np.array(self.node_indexing_list) == key
                    self.effective_perturbation_tensor = self.effective_perturbation_tensor[np.logical_not(pruned_entry_map)]
                    self.effective_max_pert_values = self.effective_max_pert_values[np.logical_not(pruned_entry_map)]
                    self.is_not_cat_tensor = self.is_not_cat_tensor[np.logical_not(pruned_entry_map)]
                    pruned_index_list_start_idx = np.where(pruned_entry_map)[0][0]
                    pruned_index_list_end_idx = np.where(pruned_entry_map)[0][-1]
                    # NOTE: assumes this list is sorted. Removes pruned index and reduces all other indexes by one
                    self.node_indexing_list = self.node_indexing_list[:pruned_index_list_start_idx] + self.node_indexing_list[pruned_index_list_end_idx+1:]
                    self.num_entry_dict.pop(key)
            current_iter += 1
        self.evalued_nodes_per_node.append(self.evalued_nodes/startnodes.shape[0])

        best_nodes = self.translate_categoricals(best_nodes, to_local=False)

        # log local attacks
        local_attacks = torch.stack([startnodes, torch.stack([best_node[1] for best_node in best_nodes], dim=0)], dim=2)
        if self.attacks is None:
            self.attacks = local_attacks
        else:
            self.attacks = torch.cat([self.attacks, local_attacks], dim=0)
        return best_nodes

    def localize_node_indexing_list(self):
        last_seen_idx = 0
        write_idx = 0
        node_indexing_list = self.node_indexing_list.copy()
        for idx, entry in enumerate(self.node_indexing_list):
            if entry != last_seen_idx:
                if idx != 0:
                    write_idx += 1
                last_seen_idx = entry
            if entry > write_idx:
                node_indexing_list[idx] = write_idx
        return node_indexing_list

    def check_termination(self, node_eval_result, current_node, search_idx, best_nodes, open_nodes, startnodes, base_probas, idx):
        terminated = False
        target_prob = (-node_eval_result * self.distance_metric.dist_func(startnodes[idx], current_node, True).item())+base_probas[idx]
        terminated = target_prob >= self.target
        if terminated:
            best_nodes[search_idx] = (node_eval_result, current_node)

        window = self.termination_windows[search_idx]
        if len(window) <= self.DEFAULTS["termination_window_size"]:
            self.termination_windows[search_idx].append(node_eval_result)
        else:
            if self.DEFAULTS["termination_criterion"] == "change":
                window.append(node_eval_result)
                max_diff = max(window) - min(window)
                terminated = max_diff <= abs(self.DEFAULTS["termination_thershhold"])
            elif self.DEFAULTS["termination_criterion"] == "improve":
                window_mean = sum(window)/len(window)
                terminated = (window_mean < node_eval_result + self.DEFAULTS["termination_thershhold"])

            window = window[1:]
            self.termination_windows[search_idx] = window

        if terminated: # we are trying to maximize the goal function here
            #best_nodes[search_idx] = (node_eval_result, current_node)
            open_nodes.pop(search_idx) # removes queue from open nodes dictionary stopping that particular search

            pruned_entry_map = np.array(self.node_indexing_list) == search_idx
            self.effective_perturbation_tensor = self.effective_perturbation_tensor[np.logical_not(pruned_entry_map)]
            self.effective_max_pert_values = self.effective_max_pert_values[np.logical_not(pruned_entry_map)]
            self.is_not_cat_tensor = self.is_not_cat_tensor[np.logical_not(pruned_entry_map)]
            pruned_index_list_start_idx = np.where(pruned_entry_map)[0][0]
            pruned_index_list_end_idx = np.where(pruned_entry_map)[0][-1]
            # NOTE: assumes this list is sorted. Removes pruned index and reduces all other indexes by one
            self.node_indexing_list = self.node_indexing_list[:pruned_index_list_start_idx] + self.node_indexing_list[pruned_index_list_end_idx+1:]
            self.num_entry_dict.pop(search_idx)

        return best_nodes, open_nodes, terminated

    def accel_adaptive_eval_func(self, perturbations, start_nodes, start_probas, victim):
        if perturbations.shape == start_nodes.shape and torch.all(perturbations == start_nodes):
            return torch.DoubleTensor([0]*len(perturbations))
        expanded_start_probas = []
        expanded_start_nodes = []
        for key in self.num_entry_dict.keys():
            num_entries = self.num_entry_dict[key]
            expanded_start_probas.extend([start_probas[key]]*num_entries)
            expanded_start_nodes.extend([start_nodes[key]]*num_entries)
        expanded_start_nodes = torch.stack(expanded_start_nodes, dim=0)
        expanded_start_probas = torch.Tensor(expanded_start_probas).to(global_defaults["device"])
        distances = self.distance_metric.dist_func(expanded_start_nodes, perturbations, pairwise=True)

        if self.DEFAULTS["constraint"] is not None:
            viable_sample_map = distances <= self.DEFAULTS["constraint"]
        else:
            viable_sample_map = torch.BoolTensor([True]*distances.shape[0])
        if torch.all(torch.logical_not(viable_sample_map)): # early abort criterion
            return torch.DoubleTensor([torch.inf]*distances.shape[0])

        self.evalued_nodes += perturbations[viable_sample_map].shape[0]
        if self.DEFAULTS["expand_type"] == "fixgrid":
            victim_proba_prediction = torch.from_numpy(victim.predict_proba(perturbations[viable_sample_map])[:,global_defaults["target_label"]])
        else:
            victim_proba_prediction = torch.from_numpy(victim.predict_proba(self.translate_categoricals(perturbations[viable_sample_map], False))[:,global_defaults["target_label"]])

        proba_change = self.activation_function(torch.subtract(victim_proba_prediction, expanded_start_probas[viable_sample_map]))

        #outside_of_distribution_map = distances > 1
        safe_div_floor = torch.Tensor([10**float(self.DEFAULTS["eps_exp"])]).to(global_defaults["device"]).expand(distances[viable_sample_map].shape[0])
        safe_div_dist, _ = torch.max(torch.stack([distances[viable_sample_map], safe_div_floor], dim=1), dim=1, keepdim=False)
        metrics = torch.DoubleTensor([-torch.inf]*distances.shape[0])
        metrics[viable_sample_map] = torch.divide(proba_change, safe_div_dist)
        #metrics[outside_of_distribution_map] = -torch.inf
        return torch.neg(metrics)
    
    def accel_adaptive_node_expand(self, nodes):
        local_node_indexing_list = self.localize_node_indexing_list()
        expanded_nodes = nodes[local_node_indexing_list] # repeats a sample by the number of expanded nodes it belongs to. Updated outside of this function
        perturbations = self.effective_perturbation_tensor
        expanded_nodes = torch.add(expanded_nodes, perturbations)
        expanded_nodes = self.round_reals(expanded_nodes)
        expanded_nodes = self.modulo_cats(expanded_nodes)
        return expanded_nodes    

    # TODO: add ceiling to adjust step size to avoid exploding distances.
    def update_perturbation_tensor(self, metrics): #assumes flat perturbation tensor of the shape the output of metrics -> needs to be updated regularly to prune finished samples and features
        num_metrics = metrics[self.is_not_cat_tensor]
        assert len(num_metrics) % self.max_expand_nodes == 0
        num_metrics = torch.reshape(num_metrics, shape=(int(num_metrics.shape[0]/self.max_expand_nodes), self.max_expand_nodes))

        if self.DEFAULTS["adjust_style"] == "beststep":
            min_tensor, argmin_tensor = torch.min(num_metrics, dim=1)
            adjust_up_tensor_map = torch.logical_and(torch.logical_or(argmin_tensor == self.max_expand_nodes-1, argmin_tensor == 0), min_tensor != torch.inf)
            adjust_up_tensor_map = torch.flatten(torch.unsqueeze(adjust_up_tensor_map, dim=1).expand((adjust_up_tensor_map.shape[0], self.max_expand_nodes)))
            # NOTE: this indexing only makes sense if the number of max_expand_nodes is EVEN. For this technique it must be EVEN
            adjust_down_tensor_map = torch.logical_or(argmin_tensor == int(self.max_expand_nodes/2), argmin_tensor == int(self.max_expand_nodes/2)+-1)
            adjust_down_tensor_map = torch.flatten(torch.unsqueeze(adjust_down_tensor_map, dim=1).expand((adjust_down_tensor_map.shape[0], self.max_expand_nodes)))
        elif self.DEFAULTS["adjust_style"] == "var_equalizer":
            inf_tensor_map = torch.isinf(torch.max(num_metrics, dim=1)[0])
            not_inf_tensor_map = torch.logical_not(inf_tensor_map)
            variances = torch.var(num_metrics[not_inf_tensor_map], dim=1)
            var_range = torch.max(variances) - torch.min(variances)
            var_median = torch.median(variances)
            highvar_map = variances > (var_median + var_range/4)
            lowvar_map = torch.logical_or(variances < (var_median - var_range/4), variances==0)
            adjust_up_tensor_map = torch.clone(not_inf_tensor_map)
            adjust_up_tensor_map[not_inf_tensor_map] = lowvar_map
            adjust_down_tensor_map = inf_tensor_map
            adjust_down_tensor_map[not_inf_tensor_map] = highvar_map
            adjust_down_tensor_map = torch.flatten(torch.unsqueeze(adjust_down_tensor_map, dim=1).expand((adjust_down_tensor_map.shape[0], self.max_expand_nodes)))
            adjust_up_tensor_map = torch.flatten(torch.unsqueeze(adjust_up_tensor_map, dim=1).expand((adjust_up_tensor_map.shape[0], self.max_expand_nodes)))

        assert not any(torch.logical_and(adjust_up_tensor_map, adjust_down_tensor_map)) #ensure we never try to both scale up and down
        if self.max_expand_nodes == 4:
            assert all(torch.logical_or(adjust_up_tensor_map, adjust_down_tensor_map)) # if we only have 4 expand steps ensure we always either scale up or down
        adjust_tensor = torch.ones(size=metrics.shape)
        exp_base = torch.Tensor([self.DEFAULTS["step_scaling"]]*adjust_tensor[self.is_not_cat_tensor].shape[0])
        # NOTE: Dont know if the double indexing here works like I think
        adjust_tensor[self.is_not_cat_tensor] = torch.mul(adjust_tensor[self.is_not_cat_tensor], torch.pow(exp_base, adjust_up_tensor_map.type(torch.FloatTensor)))
        adjust_tensor[self.is_not_cat_tensor] = torch.mul(adjust_tensor[self.is_not_cat_tensor], torch.pow(exp_base, torch.neg(adjust_down_tensor_map.type(torch.FloatTensor))))
        # expand the ajust tensor. Technically also tries to "scale" entries of a sample that are zero, but no harm done
        adjust_tensor = torch.unsqueeze(adjust_tensor, dim=1).expand((adjust_tensor.shape[0], self.effective_perturbation_tensor.shape[1]))
        
        self.effective_perturbation_tensor = torch.mul(self.effective_perturbation_tensor, adjust_tensor)
        neg_pert_map = torch.any(self.effective_perturbation_tensor < 0, dim=1)
        self.effective_perturbation_tensor[neg_pert_map] = torch.maximum(self.effective_perturbation_tensor[neg_pert_map], -self.effective_max_pert_values[neg_pert_map])
        pos_pert_map = torch.logical_not(neg_pert_map)
        self.effective_perturbation_tensor[pos_pert_map] = torch.minimum(self.effective_perturbation_tensor[pos_pert_map], self.effective_max_pert_values[pos_pert_map]) 
    
    def get_perturbation_tensor(self, x):
        num_features = x.shape[1]
        self.feature_category_to_num_map = {}
        self.feature_num_to_category_map = {}
        self.feature_category_modulus = {}
        for iter, feature in enumerate(self.feature_idxs.keys()):
            feature_idx = self.feature_idxs[feature] # already only mutable features 
            if self.feature_types[feature] == "int":
                border = int(self.step_sizes[feature]*self.max_expand_nodes/2)
                start = max(int(self.step_sizes[feature]*self.DEFAULTS["near_offset_ratio"]), 1)
                end = border+start
                pos_deltas = np.flip(np.arange(start=start, stop=end, step=self.step_sizes[feature], dtype=int),axis=0)
                neg_deltas = np.negative(np.arange(start=start, stop=end, step=self.step_sizes[feature], dtype=int))
                deltas = np.append(pos_deltas,neg_deltas,axis=0)
                maxs = np.array([max([int(self.maximums[feature]*self.DEFAULTS["max_step_ratios"]),end])]*len(deltas))
            elif self.feature_types[feature] == "real":
                border = self.step_sizes[feature]*self.max_expand_nodes/2
                start = self.step_sizes[feature]*self.DEFAULTS["near_offset_ratio"]
                end = border+start
                pos_deltas = np.flip(np.linspace(start=start, stop=end, num=int(self.max_expand_nodes/2)),axis=0)
                neg_deltas = np.negative(np.linspace(start=start, stop=end, num=int(self.max_expand_nodes/2)))  
                deltas =  np.append(pos_deltas,neg_deltas,axis=0)  
                maxs = np.array([self.maximums[feature]*self.DEFAULTS["max_step_ratios"]]*len(deltas))
            elif self.feature_types[feature] == "cat":
                self.feature_category_to_num_map[feature] = {}
                self.feature_num_to_category_map[feature] = {}
                for num_val, category in enumerate(self.mutable_feature_choices[feature]):
                    self.feature_category_to_num_map[feature][category] = num_val
                    self.feature_num_to_category_map[feature][num_val] = category
                self.feature_category_modulus[feature] = len(self.mutable_feature_choices[feature])+1
                deltas = np.arange(start=1,stop=self.feature_category_modulus[feature], step=1)
                maxs = np.array([self.feature_category_modulus[feature]]*len(deltas))

            new_chunk = np.zeros((len(deltas), num_features))
            new_chunk[:,feature_idx] = deltas
            max_chunk = np.zeros((len(maxs), num_features))
            max_chunk[:,feature_idx] = maxs
            if iter == 0:
                perturbations = new_chunk
                maximums = max_chunk
            else:
                perturbations = np.append(perturbations, new_chunk, axis=0)
                maximums = np.append(maximums, max_chunk, axis=0)
        effective_perturbation_tensor = torch.from_numpy(perturbations)
        effective_maximums = torch.from_numpy(maximums)
        return effective_perturbation_tensor, effective_maximums
    
    def adaptive_check_visited(self, expanded_nodes, eval_results, open_nodes, closed_nodes):# NOTE: must return idx mapping that maps areas of the return tensor to the key of the open nodes dictionary!
        running_idx = 0
        idx_mapping = {}
        chunks = []
        eval_chunks = []
        start_idx = 0
        assert expanded_nodes.shape[0] == eval_results.shape[0]
        for map_idx in open_nodes.keys():
            chunk_size = self.num_entry_dict[map_idx]
            chunk = expanded_nodes[start_idx:start_idx+chunk_size]
            eval_chunk = eval_results[start_idx:start_idx+chunk_size]
            start_idx += chunk_size
            open_node_queue = open_nodes[map_idx]
            # prune already open nodes and evaluations
            is_open_map = open_node_queue.is_not_in(chunk)
            chunk = chunk[is_open_map]
            eval_chunk = eval_chunk[is_open_map]
            # prune closed nodes and evaluations
            is_closed_map = utils.is_not_in(chunk, closed_nodes[map_idx])
            chunk = chunk[is_closed_map]
            eval_chunk = eval_chunk[is_closed_map]

            idx_mapping[map_idx] = (running_idx, running_idx+chunk.shape[0])
            running_idx += chunk.shape[0]
            chunks.append(chunk)
            eval_chunks.append(eval_chunk)
        return torch.cat(chunks, dim=0), torch.cat(eval_chunks, dim=0), idx_mapping

AVAILABLE_ATTACKERS = {
    "dummy": Dummy_adv,
    "baseline": Baseline_adv,
    "accel_baseline": Accel_Baseline_adv,
    "adaptive_adv": Accel_adaptive_search,
}
