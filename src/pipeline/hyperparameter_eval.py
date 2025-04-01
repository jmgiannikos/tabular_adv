import adversarial_models as adv_model
from sklearn.metrics import f1_score
from defaults import DEFAULTS, Log_styles, AVAILABLE_LOG_STYLES
import wandb
from hyperparam_opt_wrapper import HyperparamOptimizerWrapper
from hyperparameter_search_policies import AVAILABLE_OPT_STRATS
from scorer import Scorer
from utils import prune_labels

# NOTE: in current state doing this for trainable adv_cls is very costly, since we retrain the adv model from the ground. This is because we throw the previously trained models away!
# TODO: make more efficient for trainable adv_cls. Will require touching up the entire pipeline
def evaluate_hyperparams(adv_cls, victim, X, y, estimators, splits, constraints, metadata, results_path, check_constraints=True, distance_metric=None):    
    results = {}
    hyperparam_resolver = {}

    for fold_idx in range(len(splits["train"])):
        results[f"fold_{fold_idx}"] = {}
        hyperparam_resolver[f"fold_{fold_idx}"] = {}

        X_train = X[splits["train"][fold_idx]]
        y_train = y[splits["train"][fold_idx]]
        X_test = X[splits["test"][fold_idx]]
        y_test = y[splits["test"][fold_idx]]

        X_train, y_train = prune_labels(X_train, y_train, victim) # local pruning so that this is handeled uniformly. Other methods also need full X and y
        X_test, y_test = prune_labels(X_test, y_test, victim) 
        
        estimator = estimators[fold_idx] #NOTE: this is the BEST estimator found by the hyperparameter search. If we want another parametrization we need to re-train 

        # extract results on train set from estimator and create hyperparam resolver
        results[f"fold_{fold_idx}"]["train"] = {}
        results[f"fold_{fold_idx}"]["test"] = {}
        
        for hyperparam_idx in range(len(estimator.hyperparam_result_dict["x_iters"])):
            hyperparam_resolver[f"fold_{fold_idx}"][hyperparam_idx] = estimator.hyperparam_result_dict["x_iters"][hyperparam_idx]

        for hyperparam_idx in hyperparam_resolver[f"fold_{fold_idx}"].keys():
            hyperparam = hyperparam_resolver[f"fold_{fold_idx}"][hyperparam_idx]
            adv = adv_cls(victim, constraints, metadata, *hyperparam) #NOTE: Handing over the gower distance measure like this feels bad...
            if adv_cls.TRAINABLE:
                adv.fit(X_train, y_train)

            # evaluate on test set
            if check_constraints:
                scorer_obj = Scorer(results_path=results_path, log_style=Log_styles.NOLOG, constraints=constraints, distance_metric=distance_metric)
            else:
                scorer_obj = Scorer(results_path=results_path, log_style=Log_styles.NOLOG, distance_metric=distance_metric)

            hyperparam_result_test = scorer_obj.score(adv, X_test, y_test) # logging for these scores will be done manually

            hyperparam_result_train = scorer_obj.score(adv, X_train, y_train) # logging for these scores will be done manually

            results[f"fold_{fold_idx}"]["train"][hyperparam_idx] = hyperparam_result_train
            results[f"fold_{fold_idx}"]["test"][hyperparam_idx] = hyperparam_result_test

    return results, hyperparam_resolver

def wandb_log_scatter(results_dict, hyperparameter_resolver, estimators=None, fix_constraints_log=False):
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
                    if isinstance(estimator, HyperparamOptimizerWrapper):
                        estimator = estimator.model_class

                    if isinstance(estimator, adv_model.Adv_model) or issubclass(estimator, adv_model.Adv_model):
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