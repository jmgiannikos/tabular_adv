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
import os
from bounded_prio_queue import Bounded_Priority_Queue
import utils

class Adv_model(ABC):
    DEFAULTS = None
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
class Dummy_adv(Adv_model):
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

class Baseline_adv(Adv_model, sk.base.BaseEstimator):
    DEFAULTS = {
        "target": -500, # negative means this managed to flip the value (we negate for min heap purposes)
        "max_iter": 20, # maximum number of search steps before we abort and just take the best value seen so far
        "parameters": [False, 10, 50, 10, -500],
        "eps_exp": -10,
        "expand_type": "gridstep",
        "record_search_num": 10,
        "precision": 3,
        "accel_attack": True,
        "batch_size": 5000,
    }

    def __init__(self, victim, constraints, metadata, constraint_correction=DEFAULTS["parameters"][0], queue_size=DEFAULTS["parameters"][1], total_step_num=DEFAULTS["parameters"][2], max_expand_nodes=DEFAULTS["parameters"][3], target=DEFAULTS["parameters"][4], expand_type=DEFAULTS["expand_type"]):
        hyperparams = [constraint_correction, queue_size, total_step_num, max_expand_nodes, target]
        super().__init__(victim, constraints, metadata, *hyperparams)
        self.features = metadata["feature"].tolist()
        self.expand_type = expand_type
        self.mutable_feature_choices = {}
        self.step_sizes = {}
        self.feature_idxs = {}
        self.feature_types = {}
        self.significant_decimals = {}
        for idx, feature in enumerate(self.features): # NOTE: really hope the order of the metadata feature list aligns with the order of x
            if metadata.query("feature == @feature")["mutable"].to_list()[0]:
                self.mutable_feature_choices[feature] = None
                self.feature_idxs[feature] = idx
                self.feature_types[feature] = metadata.query("feature == @feature")["type"].to_list()[0]

        self.constraint_correction = constraint_correction
        self.constraint_correction = False # NOTE: Temporary constraint correction disable
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
                self.significant_decimals[feature] = self.get_significant_decimals(x, feature)
        self.perturbation_tensor = self.get_perturbation_tensor(x)
        return self

    # NOTE: this is far from perfect. Looking at the min value of the dataset and choosing based on that is functional, but it would be better if we looked at the point distances along any given feature
    def get_significant_decimals(self, x, feature):
        feature_idx = self.feature_idxs[feature]
        feature_vals = x[:][feature_idx]
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
            self.index_tensor = torch.unsqueeze(torch.LongTensor([index_list]), dim=2)
            perturbation_tensor = torch.unsqueeze(torch.DoubleTensor([perturbation_list]), dim=2)
            return perturbation_tensor
        else:
            num_features = x.shape[1]
            self.feature_category_to_num_map = {}
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
                    for num_val, category in enumerate(self.mutable_feature_choices[feature]):
                        self.feature_category_to_num_map[feature][category] = num_val
                    self.feature_category_modulus[feature] = len(self.mutable_feature_choices[feature])
                    
                    deltas = np.arange(start=1,stop=self.feature_category_modulus[feature], step=1)

                new_chunk = np.zeros((len(deltas), num_features))
                new_chunk[:,feature_idx] = deltas
                if iter == 0:
                    perturbations = new_chunk
                else:
                    perturbations = np.append(perturbations, new_chunk, axis=0)
            perturbation_tensor = torch.from_numpy(perturbations)
        return perturbation_tensor

    def attack(self, x):
        base_proba = self.victim.predict_proba(x)[:, global_defaults["target_label"]] # the base label probabilities from which we start
        if self.DEFAULTS["accel_attack"]:
            batch_num = int(x.shape[0]/self.DEFAULTS["batch_size"])+1
            batch_idx = 0
            adv_samples = []
            if self.DEFAULTS["record_search_num"] > 0:
                num_record_search = self.DEFAULTS["record_search_num"]
                record_pre_batch_x = x[:num_record_search]
                record_pre_batch_probas = base_proba[:num_record_search]
                x = x[num_record_search:]
                base_proba = base_proba[num_record_search:]
                adv_sample_batch = self.accel_adv_best_first_search(record_pre_batch_x, record_pre_batch_probas, self.victim, collect_metrics=True)
                adv_samples.extend(adv_sample_batch)

            while batch_idx < batch_num:
                min_idx = batch_idx*self.DEFAULTS["batch_size"]
                max_idx = batch_idx+1*self.DEFAULTS["batch_size"]
                batch_x = x[min_idx:max_idx]
                batch_base_proba = base_proba[min_idx:max_idx]
                adv_sample_batch = self.accel_adv_best_first_search(batch_x, batch_base_proba, self.victim, collect_metrics=False)
                adv_samples.extend(adv_sample_batch)
                batch_idx += 1
            return torch.stack(list(map(lambda x: x[1], adv_samples)), dim=0).numpy()
        else:
            adv_samples = None
            for sample_idx in range(x.shape[0]):
                adv_sample = self.adv_best_first_search(x[sample_idx], base_proba[sample_idx], self.victim, collect_metrics=sample_idx<self.DEFAULTS["record_search_num"], search_idx=sample_idx) 
                if adv_samples is None:
                    adv_samples = np.expand_dims(adv_sample, 0)
                else:
                    adv_sample = np.expand_dims(adv_sample, 0)
                    adv_samples = np.append(adv_samples, adv_sample, 0)
            return adv_samples
    
    # TODO: translate from and to cat values (needed for other datasets)
    def accel_adv_best_first_search(self, startnodes, base_probas, victim, collect_metrics=False):
        if isinstance(startnodes, np.ndarray):
            startnodes = torch.from_numpy(startnodes)
        elif isinstance(startnodes, torch.Tensor):
            startnodes = startnodes
        else:
            startnodes = torch.Tensor(startnodes)

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

        while len(open_nodes.keys()) > 0 and current_iter < self.max_iterations:
            current_nodes = []
            for idx, node_queue in [(key, open_nodes[key]) for key in open_nodes.keys()]:
                node_eval_result, current_node = node_queue.pop()
                if collect_metrics:
                    self.collect_metrics(victim=victim, search_idx=idx, node_eval_result=node_eval_result, current_iter=current_iter, current_node=current_node, start=startnodes[idx], base_proba=base_probas[idx])

                if node_eval_result <= self.target: # we are trying to maximize the goal function here
                    best_nodes[idx] = (node_eval_result, current_node)
                    open_nodes.pop(idx) # removes queue from open nodes dictionary stopping that particular search
                else:
                    if node_eval_result < best_nodes[idx][0]:
                        best_nodes[idx] = (node_eval_result, current_node)
                    current_nodes.append(current_node)
                    assert not any(list(map(lambda node: torch.equal(node, current_node), closed_nodes[idx])))
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

            for idx_mapping_key in idx_map.keys(): #TODO: may want to do this with map function? Could be faster.
                idx_mapping = idx_map[idx_mapping_key]
                loc_eval_results = eval_results[idx_mapping[0]:idx_mapping[1]]
                loc_expanded_nodes = expanded_nodes[idx_mapping[0]:idx_mapping[1]]
                new_nodes = [(loc_eval_results[idx], loc_expanded_nodes[idx]) for idx in range(loc_eval_results.shape[0])]
                open_nodes[idx_mapping_key].push(new_nodes) 
                expaded_node_ctrs[idx_mapping_key] += loc_eval_results.shape[0]
            
            for key in open_nodes.keys(): # abort condition: Queue empty
                if open_nodes[key].is_empty():
                    open_nodes.pop[key] 
            current_iter += 1
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
        expanded_start_probas = []
        expanded_start_nodes = []
        for key in idx_mappings.keys():
            num_entries = idx_mappings[key][1] - idx_mappings[key][0]
            expanded_start_probas.extend([start_probas[key]]*num_entries)
            expanded_start_nodes.extend([start_nodes[key]]*num_entries)
        expanded_start_nodes = torch.stack(expanded_start_nodes, dim=0)
        expanded_start_probas = torch.Tensor(expanded_start_probas)

        victim_proba_prediction = torch.from_numpy(victim.predict_proba(perturbations)[:,global_defaults["target_label"]])

        proba_change = torch.subtract(victim_proba_prediction, expanded_start_probas)

        distances = self.distance_metric.dist_func(expanded_start_nodes, perturbations, pairwise=True)
        safe_div_floor = torch.Tensor([10**float(self.DEFAULTS["eps_exp"])]).expand(distances.shape[0])
        safe_div_dist, _ = torch.max(torch.stack([distances, safe_div_floor], dim=1), dim=1, keepdim=False)
        metrics = torch.divide(proba_change, safe_div_dist)
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
        
    # NOTE: some loss of fidelity seems to occur with the node_eval_result not perfectly aligning with -prob_diff/gower_dist. Currently believe tha this is due to a rounding error
    def collect_metrics(self, victim, search_idx, node_eval_result, current_iter, current_node, start, base_proba):
        classifier_id = hex(id(self))[2:].upper()
        # expanded_start_proba = torch.from_numpy(victim.predict_proba(start)[:,global_defaults["target_label"]])
        victim_proba_prediction = torch.from_numpy(victim.predict_proba(current_node)[:,global_defaults["target_label"]])
        proba_change = torch.subtract(victim_proba_prediction, base_proba)
        prob_diff = proba_change
        gower_dist = self.distance_metric.dist_func(start, current_node, pairwise=True)
        running_loss = 0
        for constraint_executor in self.constraint_executors:
            running_loss += constraint_executor.execute(torch.from_numpy(np.expand_dims(current_node, axis=0))).sum()
        const_loss = running_loss
        wandb.log(
            data={
                f"{classifier_id}_search{search_idx}/prob_diff":prob_diff,
                f"{classifier_id}_search{search_idx}/gower_dist": gower_dist,
                f"{classifier_id}_search{search_idx}/constraint_loss": const_loss,
                f"{classifier_id}_search{search_idx}/cost_function": node_eval_result,
                f"{classifier_id}_search{search_idx}/step": current_iter
            }
        )

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
                self.collect_metrics(victim, search_idx, node_eval_result, current_iter, current_node, start)

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


AVAILABLE_ATTACKERS = {
    "dummy": Dummy_adv,
    "baseline": Baseline_adv
}
