import sklearn as sk
from sklearn.model_selection import cross_validate
from tabularbench.constraints.constraints_checker import ConstraintChecker
import skopt as sko
import numpy as np
import torch
from defaults import DEFAULTS
from hyperparameter_search_policies import opt_types, fixed_hyperparam, random_search

class HyperparamOptimizerWrapper(sk.base.BaseEstimator):
    def __init__(self, model_class, search_dimensions, victim, constraints, metadata, distance_metric, opt_strat, check_constraints, results_path, tolerance=DEFAULTS["tolerance"]): #TODO: passing around the whole dataset like this is cumbersome, instead just pass ranges and feature types
        self.model_class = model_class # model should be a class and have a static method that marks it as trainable or non-trainable called trainable
        self.search_dimensions = search_dimensions # see skopt.gp_minimize for the specifics
        self.best_params = None 
        self.victim = victim
        self.tolerance = tolerance
        self.constraints = constraints
        self.constraints_checker = ConstraintChecker(self.constraints, tolerance=tolerance)
        self.distance_metric = distance_metric
        self.metadata = metadata
        self.opt_strat = opt_strat
        self.hyperparam_result_dict = None
        self.results_path = results_path
        if model_class.TRAINABLE():
            self.fitted_model = None
        self.check_constraints = check_constraints

    def attack(self, X):
        if self.model_class.TRAINABLE():
            attacker_model = self.fitted_model
        else:
            attacker_model = self.model_class(self.victim, self.constraints, self.metadata, self.results_path, *self.best_params)
        adv_samples = attacker_model.attack(X)
        return adv_samples

    def fit(self, X, y, ncalls=DEFAULTS["ncalls"]):
        # these are made object variables, so that other functions in this class can access them without needing to have them passed as args, which sk.crossvalidate doesnt do for instance
        self.X = X 
        self.y = y

        if self.model_class.TRAINABLE():
            objective = self.trainable_objective
        else:
            objective = self.static_objective

        if self.opt_strat == opt_types.GRID:
            pass
        elif self.opt_strat == opt_types.FIXED:
            result_dict = fixed_hyperparam(objective, self.model_class.DEFAULTS["parameters"])
        elif self.opt_strat == opt_types.RANDOM:
            result_dict = random_search(objective, self.search_dimensions, n_calls=ncalls)
        elif self.opt_strat == opt_types.GPMIN:
            result_dict = sko.gp_minimize(objective, self.search_dimensions, n_calls=ncalls)

        if self.model_class.TRAINABLE():
            self.fitted_model = self.model_class(self.victim, self.constraints, self.metadata, self.results_path, *result_dict.x).fit(X, y)
        
        self.hyperparam_result_dict = result_dict # store result dict for later analysis
        self.best_params = [result_dict.x] # this is wrapped in a list, because of the weird parameter shape the optimization function uses internally. This allows consistency between the modes
        return self
        
    def trainable_objective(self, *args):
        if len(args) == 1 and isinstance(args[0], list):
            args = args[0]
        model = self.model_class(self.victim, self.constraints, self.metadata, self.results_path, *args) #args (hyperparameters) that the model takes need to align with the search dimensions
        scores = cross_validate(model, self.X, self.y, scoring=self.objective, cv=DEFAULTS["crossval_folds"])
        score = np.mean(scores["test_score"])
        return score

    def static_objective(self, *args):
        args = args[0]
        model = self.model_class(self.victim, self.constraints, self.metadata, self.results_path, *args) #args (hyperparameters) that the model takes need to align with the search dimensions
        score = self.objective(model, self.X, self.y)
        return score

    def objective(self, model, val_X, val_y, imperceptability_weighting=DEFAULTS["imperceptability_weight"]): # X here are the adv samples (the naming scheme is so we can use the crossval function from sklearn for trainable_objective)
        adv_samples = model.attack(val_X)
        pruned_adv_samples, pruned_X, pruned_y = self.prune_constraint_violations(adv_samples, val_X, val_y)
        #num_constraint_violations = np.shape(val_y)[0] - np.shape(pruned_y)[0] # value mostly irrellevant currently
        victim_predictions = self.victim.predict(pruned_adv_samples) # assumes predict produces binary labels
        victim_orig_predictions = self.victim.predict(pruned_X)
        flipped_labels = victim_orig_predictions != victim_predictions
        success_rate = np.sum(flipped_labels.astype(int))/np.shape(self.y)[0] # assumes 0 is the main index of y. Samples that violate constraints are not counted, even if the label flips
        imperceptability = self.get_imperceptability(pruned_adv_samples, pruned_X)
        score = (1-imperceptability_weighting)*success_rate + (imperceptability_weighting)*imperceptability #starting with equal weight should be fine since both values are between 0 and 1
        return score

    def get_imperceptability(self, adv_samples, base_samples):
        gower_distances = self.distance_metric.dist_func(adv_samples, base_samples, pairwise=True)
        # calculate mean distance to original sample in absence of a better idea 
        if gower_distances.shape[0] != 0:
            mean_dist = torch.sum(gower_distances).item()/gower_distances.shape[0]
        else:
            mean_dist = 0
        return -mean_dist # return negative mean dist, so minimizing gower distance has a positive effect

    def prune_constraint_violations(self, adv_samples, X, y):
        if isinstance(adv_samples, torch.Tensor):
            adv_samples = adv_samples.numpy(True)
        if isinstance(X,  torch.Tensor):
            X = X.numpy(True)
        if self.check_constraints:
            sample_viability_map = self.constraints_checker.check_constraints(X, adv_samples) # True for every sample that breaches no constraints
            bool_sample_viability_map = sample_viability_map == 1
            selected_adv_samples = adv_samples[bool_sample_viability_map] #assumes shape nxm where n is number of samples and m is number of features per sample
            selected_x = X[bool_sample_viability_map] #assumes shape nxm where n is number of samples and m is number of features per sample
            selected_y = y[bool_sample_viability_map] # assumes y to be one dimensional 
            return selected_adv_samples, selected_x, selected_y
        else:
            return adv_samples, X, y