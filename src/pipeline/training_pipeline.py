import sklearn as sk
from scipy.optimize import OptimizeResult
from tabularbench.datasets import dataset_factory 
from sklearn.model_selection import cross_validate, KFold
from tabularbench.constraints.constraints_checker import ConstraintChecker
import skopt as sko
import numpy as np
import torch
import classifier_models as cls_model
import adversarial_models as adv_model
from enum import Enum
from sklearn.metrics import f1_score
import argparse
from gower_distance import Gower_dist
from defaults import DEFAULTS
import wandb
import subprocess
import os
import datetime
import pickle
import cProfile

class opt_types(Enum):
    GPMIN = "Gaussian Process (Minimize)"
    RANDOM = "Random sampling"
    GRID = "Grid Search"
    FIXED = "no hyperparameter optimization"

# I admit this translation of str to enum is dumb, but whatever
AVAILABLE_OPT_STRATS = {
    "random": opt_types.RANDOM,
    "gpmin": opt_types.GPMIN,
    "fix": opt_types.FIXED,
    #"grid": opt_types.GRID
}

DATASET_ALIASES = [
        "ctu_13_neris",
        "lcld_time",
        "malware",
        "url",
        "wids",
    ]

DEFAULT_CONFIG = {
    "dataset_name": "url",
    "attacker_cls": "dummy",
    "victim_cls": "random",
    "opt_strat": "random",
    "cls_epochs": 1,
    "cls_batch_size": -1
}

# TODO: Handling of mutable and non mutable features -> seems like constraint checker already checks mutability constraints
# TODO: may have to introduce special handelling to translate dataframe to numpy array (or even tensor, if we are too cpu constrained?) -> seems to run fine currently. May have to revisit when using real models

# pipeline doing crossval for an adversarial against a victim on a dataset
def pipeline(dataset_name, attacker_cls, victim_cls, opt_strat, eval_hyperparameters, tune_cls, cls_epochs, cls_batch_size, profile, wandb_log=False):
    attacker_cls = adv_model.AVAILABLE_ATTACKERS[attacker_cls]
    victim_cls = cls_model.AVAILABLE_VICTIMS[victim_cls]
    opt_strat = AVAILABLE_OPT_STRATS[opt_strat]

    dataset = dataset_factory.get_dataset(dataset_name)
    x, y = dataset.get_x_y()

    metadata = dataset.get_metadata(only_x=True)
    distance_metric = Gower_dist(x=x, metadata=metadata) # TODO: may want to make this adjustable?
    constraints = dataset.get_constraints()

    # TODO: Insert a crossval for victim training here
    kfold_splitter = KFold(n_splits=2)

    result_dict = {}
    for fold_idx, (train_index, test_index) in enumerate(kfold_splitter.split(x,y)):
        # split used for victim training (and testing)
        x_victim = x.to_numpy()[train_index] 
        y_victim = y[train_index]
        #split used for adversarial training (and testing)
        x_adv = x.to_numpy()[test_index]
        y_adv = y[test_index]

        victim_scores, victim = victim_training_pipeline(x_victim, y_victim, x_adv, y_adv, victim_cls, tune_cls, wandb_log, epochs=cls_epochs, cls_batch_size=cls_batch_size)
        if profile is not None:
            profile.dump_stats(DEFAULTS["results_path"]+DEFAULTS["performance_log_file"])

        adversarial_scores, adversarial = adversarial_training_pipeline(x_adv, y_adv, attacker_cls, victim, constraints, metadata, distance_metric, opt_strat, eval_hyperparameters, wandb_log, profile)
        if profile is not None:
            profile.dump_stats(DEFAULTS["results_path"]+DEFAULTS["performance_log_file"])

        result_dict[f"fold {fold_idx}"] = {
            "victim_scores": victim_scores,
            "victim": victim,
            "adversarial_scores": adversarial_scores,
            "adversarial": adversarial
        }

    return result_dict

# NOTE: in current state doing this for trainable adv_cls is very costly, since we retrain the adv model from the ground. This is because we throw the previously trained models away!
# TODO: make more efficient for trainable adv_cls. Will require touching up the entire pipeline
def evaluate_hyperparams(adv_cls, victim, X, y, estimators, splits, constraints, metadata):
    results = {}
    hyperparam_resolver = {}

    for fold_idx in range(splits["train"].shape[0]):
        results[f"fold_{fold_idx}"] = {}
        hyperparam_resolver[f"fold_{fold_idx}"] = {}

        X_train = X[splits["train"][fold_idx]]
        y_train = y[splits["train"][fold_idx]]
        X_test = X[splits["test"][fold_idx]]
        y_test = y[splits["test"][fold_idx]]
        estimator = estimators[fold_idx] #NOTE: this is the BEST estimator found by the hyperparameter search. If we want another parametrization we need to re-train 

        # extract results on train set from estimator and create hyperparam resolver
        results[f"fold_{fold_idx}"]["train"] = {}
        for hyperparam_idx in range(estimator.hyperparam_result_dict["x_iters"].shape[0]):
            hyperparam_resolver[f"fold_{fold_idx}"][hyperparam_idx] = estimator.hyperparam_result_dict["x_iters"][hyperparam_idx]

        for hyperparam_idx in hyperparam_resolver[f"fold_{fold_idx}"].keys():
            hyperparam = hyperparam_resolver[f"fold_{fold_idx}"][hyperparam_idx]
            adv = adv_cls(victim, constraints, metadata, *hyperparam) #NOTE: Handing over the gower distance measure like this feels bad...
            if adv_cls.TRAINABLE:
                adv.fit(X_train, y_train)

            # evaluate on test set

            hyperparam_result_test = scorer(adv, X_test, y_test, wandb_log=False) # logging for these scores will be done manually
            hyperparam_result_train = scorer(adv, X_train, y_train, wandb_log=False) # logging for these scores will be done manually

            results[f"fold_{fold_idx}"]["train"][hyperparam_idx] = hyperparam_result_train
            results[f"fold_{fold_idx}"]["test"][hyperparam_idx] = hyperparam_result_test

    return results, hyperparam_resolver

def wandb_log_scatter(results_dict, hyperparameter_resolver, estimators, fix_constraints_log=False):
    for fold_name in results_dict.keys():
        val_dicts = {
            "test": results_dict[fold_name]["test"],
            "train": results_dict[fold_name]["train"] 
        }
        for val_dict_name in val_dicts.keys():
            plot_name = f"{fold_name}:{val_dict_name}"
            rows = []
            for hyperparam_idx in val_dicts[val_dict_name].keys():
                hyperparam_results = val_dicts[val_dict_name][hyperparam_idx]
                row = []
                row.append(hyperparam_results["imperceptability"])
                row.append(hyperparam_results["success rate"])
                if fix_constraints_log:
                    estimator = estimators[int(fold_name.split("_")[-1])]
                    hyperparam_names = estimator.HYPERPARAM_NAMES()
                    if "constraint_correction" in hyperparam_names:
                        fix_constraints_idx = hyperparam_names.index("constraint_correction")
                        constraint_fixing_val = hyperparameter_resolver[fold_name][hyperparam_idx][fix_constraints_idx]
                        row.append(constraint_fixing_val)
                rows.append(row)
            
            if fix_constraints_log:
                table = wandb.Table(data=rows, columns=["imperceptability", "success rate", "fixed constraints"])
            else:
                table = wandb.Table(data=rows, columns=["imperceptability", "success rate"])
            wandb.log({plot_name: wandb.plot.scatter(table, "imperceptability", "success rate", title=plot_name)})
        
def adversarial_training_pipeline(x, y, adv_class, victim, constraints, metadata, distance_metric, opt_strat, eval_hyperparameters, wandb_log, profile):
    x, y = select_non_target_labels(x,y)
    search_dimensions = adv_class.SEARCH_DIMS()
    optimization_wrapper = HyperparamOptimizerWrapper(adv_class, search_dimensions, victim, constraints, metadata, distance_metric, opt_strat, profile) 
    crossval_scorer = scorer("outer_crossval_scorer", wandb_log)
    scores = cross_validate(optimization_wrapper, x, y, scoring=crossval_scorer.score, error_score="raise", return_estimator=eval_hyperparameters, return_indices=eval_hyperparameters)
    
    if eval_hyperparameters:
        hyperparam_results, hyperparam_resolver = evaluate_hyperparams(adv_class, victim, x, y, scores["estimator"], scores["indices"], constraints, metadata)
        if wandb_log:

            wandb_log_scatter(hyperparam_results, hyperparam_resolver)
        scores["hyperparam_results"] = hyperparam_results
        scores["hyperparam_resolver"] = hyperparam_resolver

    return scores, optimization_wrapper # NOTE: unsure how corss_validate affects optimization wrapper so this may be weird with regards to things set in fit function. Better to use estimators returned in scores

def victim_training_pipeline(x, y, x_test, y_test, victim_class, tune_model, wandb_log, epochs=DEFAULTS["victim_training_epochs"], cls_batch_size=-1): # tune model is generally False. If its ever true, we need to adjust this pipeline section
    victim = victim_class(tune_model) # TODO: May want to pass an initialized object as parameter in some way, so we dont have to initialize here. Handeling class not nessecary since we dont do hyperparam optimization here
    if cls_batch_size != -1:
        batches = batch(x,y,cls_batch_size)
    else:
        batches = [(x, y)]
    best_score = None
    best_model = None
    
    for _ in range(epochs):
        for x_batch, y_batch in batches:
            victim = victim.fit(x_batch, y_batch)
        
        y_pred = victim.predict(x_test)
        score = f1_score(y_test, y_pred)
        if wandb_log:
            wandb.log({"victim_f1": score})
        best_score, best_model = update_best_model(best_model, best_score, victim, score)

    return best_score, best_model

def update_best_model(best_model, best_score, model, score):
    if best_score is None:
        ret_score = score
        ret_model = model
    elif best_score >= score:
        ret_score = best_score
        ret_model = best_model
    else:
        ret_score = score
        ret_model = model

    return ret_score, ret_model

def batch(x,y,num_batch):
    batches = []
    batch_size = int(x.shape[0]/num_batch)
    for idx in range(num_batch):
        start_idx = idx*batch_size
        end_idx = (idx+1)*batch_size-1
        batches.append((x[start_idx:end_idx], y[start_idx:end_idx]))
    start_idx = num_batch*batch_size
    batches.append((x[start_idx:-1], y[start_idx:-1]))
    return batches


def select_non_target_labels(x,y,target_label=DEFAULTS["target_label"]):
    not_target_label_map = y != target_label
    x = x[not_target_label_map]
    y = y[not_target_label_map] #kinda redundant because this should now all be the same label, but whatever
    return x, y

# NOTE: this is kinda unclean, since we read a lot of dataset properties from the attacker wrapper, which is unintuitive. Should however be clean technically.
class scorer():
    def __init__(self, name="", wandb_log=False):
        self.name=name
        self.call_counter = 0
        self.wandb_log = wandb_log
        self.logged_tables = {}
        
    def score(self, attacker, X, y): #takes an initialized model and gives it a score. Must be compatible with sklearns crossvalidate
        adv_samples = attacker.attack(X)
        pruned_adv_samples, pruned_X, pruned_y = attacker.prune_constraint_violations(adv_samples, X, y)
        num_constraint_violations = np.shape(y)[0] - np.shape(pruned_y)[0] # value mostly irrellevant currently
        victim_predictions = attacker.victim.predict(pruned_adv_samples) # assumes predict produces binary labels
        flipped_labels = pruned_y == victim_predictions
        success_rate = np.sum(flipped_labels.astype(int))/np.shape(y)[0] # assumes 0 is the main index of y. Samples that violate constraints are not counted, even if the label flips
        imperceptability = attacker.get_imperceptability(pruned_adv_samples, pruned_X)

        if self.wandb_log:
            self.log_bar_chart(f"{self.name}: constraint violations", self.call_counter, num_constraint_violations)
            self.log_bar_chart(f"{self.name}: success rate", self.call_counter, success_rate)
            self.log_bar_chart(f"{self.name}: imperceptability", self.call_counter, imperceptability)

        return {"constraint violations": num_constraint_violations,
                "success rate": success_rate,
                "imperceptability": imperceptability}
    
    def log_bar_chart(self, chart_name, fold, value):
        fold_name = F"fold_{fold}"
        if chart_name in self.logged_tables.keys():
            table = self.logged_tables[chart_name]
            table.add_data(fold_name, value)
        else:
            table = wandb.Table(columns=["fold", "value"], data=[[fold_name, value]])

        wandb.log(
            {
                chart_name: wandb.plot.bar(
                    table, "fold", "value", title=chart_name
                )
            }
        )
        

# TODO: maybe implement this? Is more complicated than it seems (especially efficiently) if we wanna do exactly n evenly spaced calls
def grid_search(objective, search_dimensions, n_calls):
    pass

def fixed_hyperparam(objective, default_params):
    score = objective()
    hyperparameters = default_params

    optresult = OptimizeResult() 

    optresult.update([{"x": hyperparameters, "fun": score, "x_iters": np.array([hyperparameters]), "func_vals": np.array([score])}])
    return optresult

def random_search(objective, search_dimensions, n_calls):
    generator = np.random.default_rng()
    for idx, feature in enumerate(search_dimensions):
        if isinstance(feature, list):
            feature_vals = generator.choice(feature, size=(n_calls,1))
        elif isinstance(feature, tuple):
            if isinstance(feature[0], int):
                feature_vals = generator.integers(feature[0], feature[1]+1, size=(n_calls,1))
            elif isinstance(feature[0], float):
                feature_vals = generator.uniform(feature[0], feature[1], size=(n_calls,1))
        if idx == 0:
            parameter_mat = feature_vals
        else:
            parameter_mat = np.append(parameter_mat, feature_vals, axis=1)

    kwargs = {"objective": objective}
    objective_scores = np.apply_along_axis(objective_wrapper, 1, parameter_mat, **kwargs)
 
    best_score = np.min(objective_scores)
    best_params = parameter_mat[np.argmin(objective_scores)]

    optresult = OptimizeResult() 

    optresult.update([{"x": best_params, "fun": best_score, "x_iters": parameter_mat, "func_vals": objective_scores}])

    return optresult

def objective_wrapper(np_args, objective):
    args = np_args.tolist()
    return objective(*args)

class HyperparamOptimizerWrapper(sk.base.BaseEstimator):
    def __init__(self, model_class, search_dimensions, victim, constraints, metadata, distance_metric, opt_strat, tolerance=DEFAULTS["tolerance"], profile=None): #TODO: passing around the whole dataset like this is cumbersome, instead just pass ranges and feature types
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
        if model_class.TRAINABLE():
            self.fitted_model = None
        self.profile = None

    def attack(self, X):
        if self.model_class.TRAINABLE():
            attacker_model = self.fitted_model
        else:
            attacker_model = self.model_class(self.victim, self.constraints, self.metadata, *self.best_params, profile)
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
        elif self.fixed == opt_types.FIXED:
            result_dict = fixed_hyperparam(objective, self.model_class.DEFAULTS["parameters"])
        elif self.opt_strat == opt_types.RANDOM:
            result_dict = random_search(objective, self.search_dimensions, n_calls=ncalls)
        elif self.opt_strat == opt_types.GPMIN:
            result_dict = sko.gp_minimize(objective, self.search_dimensions, n_calls=ncalls)

        if self.model_class.TRAINABLE():
            self.fitted_model = self.model_class(self.victim, self.constraints, self.metadata, *result_dict.x, profile).fit(X)
        
        self.hyperparam_result_dict = result_dict # store result dict for later analysis
        self.best_params = [result_dict.x] # this is wrapped in a list, because of the weird parameter shape the optimization function uses internally. This allows consistency between the modes
        return self
        
    def trainable_objective(self, *args):
        model = self.model_class(self.victim, self.constraints, self.metadata, *args, profile) #args (hyperparameters) that the model takes need to align with the search dimensions
        scores = cross_validate(model, self.X, self.y, scoring=self.objective)
        score = np.mean(scores["test_score"])
        return score

    def static_objective(self, *args):
        model = self.model_class(self.victim, self.constraints, self.metadata, *args, profile) #args (hyperparameters) that the model takes need to align with the search dimensions
        score = self.objective(model, self.X, self.y)
        return score

    def objective(self, model, val_X, val_y, imperceptability_weighting=DEFAULTS["imperceptability_weight"]): # X here are the adv samples (the naming scheme is so we can use the crossval function from sklearn for trainable_objective)
        adv_samples = model.attack(val_X)
        pruned_adv_samples, pruned_X, pruned_y = self.prune_constraint_violations(adv_samples, val_X, val_y)
        #num_constraint_violations = np.shape(val_y)[0] - np.shape(pruned_y)[0] # value mostly irrellevant currently
        victim_predictions = self.victim.predict(pruned_adv_samples) # assumes predict produces binary labels
        flipped_labels = pruned_y == victim_predictions
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
        sample_viability_map = self.constraints_checker.check_constraints(X.to_numpy(), adv_samples.to_numpy()) # True for every sample that breaches no constraints
        bool_sample_viability_map = sample_viability_map == 1
        selected_adv_samples = adv_samples[bool_sample_viability_map] #assumes shape nxm where n is number of samples and m is number of features per sample
        selected_x = X[bool_sample_viability_map] #assumes shape nxm where n is number of samples and m is number of features per sample
        selected_y = y[bool_sample_viability_map] # assumes y to be one dimensional 
        return selected_adv_samples, selected_x, selected_y

def main(profile= None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-victim_cls", "-v", type=str, choices=cls_model.AVAILABLE_VICTIMS.keys(), default=None)
    parser.add_argument("-attacker_cls", "-a", type=str, choices=adv_model.AVAILABLE_ATTACKERS.keys(), default=None)
    parser.add_argument("-dataset_name", "-d", type=str, choices=DATASET_ALIASES, default=None)
    parser.add_argument("-opt_strat", "-o", type=str, choices=AVAILABLE_OPT_STRATS.keys(), default=None)
    parser.add_argument("-config", "-c", type=dict, default=None)
    parser.add_argument("-cls_epochs", "-clse", type=int, default=1)
    parser.add_argument("-tune_cls", "-tcls", action="store_true")
    parser.add_argument("-wandb_log", "-log", action="store_true")
    parser.add_argument("-cls_batch_size", "-clsb", type=int, default=-1)
    parser.add_argument("-eval_hyper", "-evalh", action="store_true")
    args = parser.parse_args()
    
    if args.config is not None:
        config=args.config
        for key in DEFAULT_CONFIG.keys():
            if key not in config.keys():
                config[key] = DEFAULT_CONFIG[key]
    else:
        config=DEFAULT_CONFIG
    
    args_dict = vars(args)
    for key in config.keys():
        if key in args_dict.keys():
            if args_dict[key] is not None:
                config[key] = args_dict[key]

    # manually add/override the action="store_true" fields to the config dict, because they are not captured by vargs()
    config["tune_cls"] = args.tune_cls
    config["wandb_log"] = args.wandb_log
    config["eval_hyperparameters"] = args.eval_hyper

    current_git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    log_config = {
                "victim": config["victim_cls"],
                "attacker_cls": config["attacker_cls"],
                "dataset_name": config["dataset_name"],
                "opt_strat": config["opt_strat"],
                "tune_cls": config["tune_cls"],
                "cls_epochs": config["cls_epochs"],
                "cls_batch_size": config["cls_batch_size"],
                "pipeline_defaults": DEFAULTS,
                "adv_defaults": adv_model.AVAILABLE_ATTACKERS[config["attacker_cls"]].DEFAULTS,
                "cls_defaults": cls_model.AVAILABLE_VICTIMS[config["victim_cls"]].DEFAULTS,
                "git_hash": current_git_hash
            }
    
    if "config" in config.keys():
        log_config["file_loaded_config"] = config["config"]

    if args.wandb_log:
        wandb.init(
            project="tabular_adv",
            config=log_config
        )

    results_path = DEFAULTS["results_path"]
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    config["profile"] = profile

    results = pipeline(**config)
    
    current_time = datetime.datetime.now()
    run_results_path = f"/[run]{current_time.year}-{current_time.month}-{current_time.day}_{current_time.hour}:{current_time.minute}:{current_time.second}/"
    os.mkdir(results_path+run_results_path)

    with open(results_path+run_results_path+'configs.pkl', 'wb+') as f:
        pickle.dump(log_config, f)

    with open(results_path+run_results_path+'results.pkl', 'wb+') as f:
        pickle.dump(results, f)
    
    profile.dump_stats(results_path)

        

if __name__ == "__main__":
    profile = cProfile.Profile()
    profile.enable()
    main(profile)
    profile.disable()
    
        

