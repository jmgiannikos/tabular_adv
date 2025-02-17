from scipy.optimize import OptimizeResult
import numpy as np
from enum import Enum

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

# TODO: maybe implement this? Is more complicated than it seems (especially efficiently) if we wanna do exactly n evenly spaced calls
def grid_search(objective, search_dimensions, n_calls):
    pass

def fixed_hyperparam(objective, default_params):
    score = objective(default_params)
    hyperparameters = default_params

    optresult = OptimizeResult({"x": hyperparameters, "fun": score, "x_iters": np.array([hyperparameters]), "func_vals": np.array([score])}) 
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

    optresult = OptimizeResult({"x": best_params, "fun": best_score, "x_iters": parameter_mat, "func_vals": objective_scores}) 

    #optresult.update([{"x": best_params, "fun": best_score, "x_iters": parameter_mat, "func_vals": objective_scores}])

    return optresult

def objective_wrapper(np_args, objective):
    args = (np_args.tolist(),)
    return objective(*args)