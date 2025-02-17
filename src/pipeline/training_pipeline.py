from tabularbench.datasets import dataset_factory 
from sklearn.model_selection import cross_validate, KFold
import classifier_models as cls_model
import adversarial_models as adv_model
from sklearn.metrics import f1_score
import argparse
from gower_distance import Gower_dist
from defaults import DEFAULTS, Log_styles, AVAILABLE_LOG_STYLES
import wandb
import subprocess
import os
import datetime
import pickle
import cProfile
from hyperparam_opt_wrapper import HyperparamOptimizerWrapper
from hyperparameter_search_policies import AVAILABLE_OPT_STRATS
from scorer import Scorer
from hyperparameter_eval import wandb_log_scatter, evaluate_hyperparams

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
    "cls_batch_size": -1,
    "dataset_cap": None
}

# TODO: Handling of mutable and non mutable features -> seems like constraint checker already checks mutability constraints
# TODO: may have to introduce special handelling to translate dataframe to numpy array (or even tensor, if we are too cpu constrained?) -> seems to run fine currently. May have to revisit when using real models

# pipeline doing crossval for an adversarial against a victim on a dataset
def pipeline(dataset_name, attacker_cls, victim_cls, opt_strat, eval_hyperparameters, tune_cls, cls_epochs, cls_batch_size, profile, check_constraints, results_path, dataset_cap=None, log_style=Log_styles.NOLOG):
    attacker_cls = adv_model.AVAILABLE_ATTACKERS[attacker_cls]
    victim_cls = cls_model.AVAILABLE_VICTIMS[victim_cls]
    opt_strat = AVAILABLE_OPT_STRATS[opt_strat]

    dataset = dataset_factory.get_dataset(dataset_name)
    x, y = dataset.get_x_y()
    if dataset_cap is not None and dataset_cap < x.shape[0]:
        x = x[:dataset_cap]
        y = y[:dataset_cap]

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

        victim_scores, victim = victim_training_pipeline(x=x_victim, y=y_victim, x_test=x_adv, y_test=y_adv, victim_class=victim_cls, tune_model=tune_cls, log_style=log_style ,epochs=cls_epochs, cls_batch_size=cls_batch_size)
        if profile is not None:
            profile.dump_stats(DEFAULTS["results_path"]+DEFAULTS["performance_log_file"])

        x_adv, y_adv = select_non_target_labels(x_adv,y_adv)
        adversarial_scores, adversarial = adversarial_training_pipeline(x_adv, y_adv, attacker_cls, victim, constraints, metadata, distance_metric, opt_strat, eval_hyperparameters, check_constraints=check_constraints, results_path=results_path, log_style=log_style)
        if profile is not None:
            profile.dump_stats(DEFAULTS["results_path"]+DEFAULTS["performance_log_file"])

        result_dict[f"fold {fold_idx}"] = {
            "victim_scores": victim_scores,
            "victim": victim,
            "adversarial_scores": adversarial_scores,
            "adversarial": adversarial,
            "adv_set_size": y_adv.shape[0]
        }

    return result_dict
        
def adversarial_training_pipeline(x, y, adv_class, victim, constraints, metadata, distance_metric, opt_strat, eval_hyperparameters, check_constraints, results_path, log_style):
    search_dimensions = adv_class.SEARCH_DIMS()
    optimization_wrapper = HyperparamOptimizerWrapper(adv_class, search_dimensions, victim, constraints, metadata, distance_metric, opt_strat, results_path=results_path, check_constraints=check_constraints) 
    crossval_scorer = Scorer(name="outer_crossval_scorer", results_path=results_path, log_style=log_style  ,feature_names=metadata["feature"].tolist())
    scores = cross_validate(optimization_wrapper, x, y, scoring=crossval_scorer.score, error_score="raise", return_estimator=eval_hyperparameters, return_indices=eval_hyperparameters, cv=DEFAULTS["crossval_folds"])      
    if log_style == Log_styles.BOTH or log_style == Log_styles.WANDB:
        crossval_scorer.log_bar_chart()
        crossval_scorer.wandb_log_feat_analysis() 
    if log_style == Log_styles.BOTH or log_style == Log_styles.LOCAL:
        crossval_scorer.dump_logs()

    if eval_hyperparameters:
        hyperparam_results, hyperparam_resolver = evaluate_hyperparams(adv_class, victim, x, y, scores["estimator"], scores["indices"], constraints, metadata, check_constraints=check_constraints, distance_metric=distance_metric)
        if log_style == Log_styles.BOTH or log_style == Log_styles.WANDB:
            wandb_log_scatter(hyperparam_results, hyperparam_resolver, scores["estimator"], fix_constraints_log=True)
        scores["hyperparam_results"] = hyperparam_results
        scores["hyperparam_resolver"] = hyperparam_resolver

    return scores, optimization_wrapper # NOTE: unsure how corss_validate affects optimization wrapper so this may be weird with regards to things set in fit function. Better to use estimators returned in scores

def victim_training_pipeline(x, y, x_test, y_test, victim_class, tune_model, log_style, epochs=DEFAULTS["victim_training_epochs"], cls_batch_size=-1): # tune model is generally False. If its ever true, we need to adjust this pipeline section
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
        if log_style == Log_styles.BOTH or log_style == Log_styles.WANDB:
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

def main(profile= None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-victim_cls", "-v", type=str, choices=cls_model.AVAILABLE_VICTIMS.keys(), default=None)
    parser.add_argument("-attacker_cls", "-a", type=str, choices=adv_model.AVAILABLE_ATTACKERS.keys(), default=None)
    parser.add_argument("-dataset_name", "-d", type=str, choices=DATASET_ALIASES, default=None)
    parser.add_argument("-opt_strat", "-o", type=str, choices=AVAILABLE_OPT_STRATS.keys(), default=None)
    parser.add_argument("-config", "-c", type=dict, default=None)
    parser.add_argument("-cls_epochs", "-clse", type=int, default=1)
    parser.add_argument("-tune_cls", "-tcls", action="store_true")
    parser.add_argument("-cls_batch_size", "-clsb", type=int, default=-1)
    parser.add_argument("-eval_hyper", "-evalh", action="store_true")
    parser.add_argument("-dataset_cap", "-dcap", type=int, default=None)
    parser.add_argument("-check_constraints", "-con", action="store_true")
    parser.add_argument("-log_style", "-lsty", type=str, choices=AVAILABLE_LOG_STYLES.keys(), default="nolog" )
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
    config["eval_hyperparameters"] = args.eval_hyper
    config["check_constraints"] = args.check_constraints
    config["log_style"] = AVAILABLE_LOG_STYLES[args.log_style]

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
                "scorer_defaults": Scorer.DEFAULTS,
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

    current_time = datetime.datetime.now()
    run_results_path = f"/[run]{current_time.year}-{current_time.month}-{current_time.day}_{current_time.hour}:{current_time.minute}:{current_time.second}/"
    os.mkdir(results_path+run_results_path)

    config["results_path"] = results_path+run_results_path

    with open(results_path+run_results_path+'configs.pkl', 'wb+') as f:
        pickle.dump(log_config, f)

    results = pipeline(**config)

    with open(results_path+run_results_path+'results.pkl', 'wb+') as f:
        pickle.dump(results, f)
    
    #profile.dump_stats(results_path + run_results_path + "time_profile") # currently useless, since only main thread is profiled. Most work happens in side threads     

if __name__ == "__main__":
    profile = cProfile.Profile()
    profile.enable()
    main(profile)
    profile.disable()
    
        

