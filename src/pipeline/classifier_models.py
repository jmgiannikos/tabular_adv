from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
import xgboost as xgb
import skopt as sko

# probably not going to need this, since we want to use fully implemented models (preferrably already pre-trained, but that may be a bit too optimistic). May serve as a wrapper
class Cls_model(ABC):
    DEFAULTS = None
    def __init__(self, tune_model):
        pass

    @abstractmethod
    def tune_hyperparameters(self, x, y, ncalls):
        pass

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

class Random_Guesser(Cls_model):
    def __init__(self, tune_model):
        super().__init__()
        self.guesser = np.random.default_rng()

    def tune_hyperparameters(self, x, y, ncalls):
        return self

    def fit(self, x, y):
        return self
    
    def predict(self, x):
        guesses = self.guesser.integers(0,2,size=x.shape[0])
        return guesses

# TODO: Early stopping not implemented 
# TODO: Make all parameters searchable by the hyperparameter optimization method
class XGB(Cls_model):
    DEFAULTS = {
        "ncalls": 10,
        "xgb_rounds": 1
    }
    def __init__(self, tune_model=False):
        super().__init__(tune_model)
        self.parameter_list = ["n_estimators", "max_depth", "max_leaves", "grow_policy", "booster"] # learning rate intentionally left out, as this may interact with XGB_ROUNDS in unintended ways
        # left out a good chunk of the parameters, when I had no idea how to judge what a reasonable setting range would be
        self.search_dimensions = [(1,10), (1,40), (1,500), ["depthwise", "lossguide"], ["gbtree", "gblinear", "dart"]]
        self.best_parameters = None
        self.tune_model = tune_model

    def tune_hyperparameters(self, x, y, ncalls=DEFAULTS["ncalls"]):
        self.x = x
        self.y = y 
        result_dict = sko.gp_minimize(self.objective, self.search_dimensions, n_calls=ncalls)
        self.best_parameters = [result_dict.x]
        self.fit(x,y)
        return self
    
    def fit(self, x, y, hyperparameters=None, num_rounds=DEFAULTS["xgb_rounds"]):
        if not self.tune_model:
            model = xgb.XGBClassifier()
            self.model = model.fit(x,y)
        else:
            if hyperparameters is None:
                if self.best_parameters is None:
                    raise ValueError("hyperparameters not initialized")
                else:
                    hyperparameters = self.best_parameters
            
            dtrain = xgb.DMatrix(x, label=y)
            self.model = xgb.train(hyperparameters, dtrain, num_rounds)
        return self

    def objective(self, *args):
        parameters = {}
        parameters = self.dictify_params(*args)
        model = xgb.XGBClassifier(**parameters)
        scores = cross_validate(model, self.x, self.y, scoring="f1")
        score = np.mean(scores["test_score"])
        return score

    def dictify_params(self, *parameters):
        for i, parameter_name in enumerate(self.parameter_list): # had to use native python for loop. sadge :(
            parameters[parameter_name] = parameters[i]
        return parameters
    
    def predict(self, x):
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        if isinstance(self.model, xgb.XGBClassifier):
            ypred = self.model.predict(x)
        else:
            dtest = xgb.DMatrix(x)
            ypred = self.model.predict(dtest, training=False)
        return ypred
    
    def predict_proba(self, x):
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        if isinstance(self.model, xgb.XGBClassifier):
            ypob = self.model.predict_proba(x)
        return ypob

AVAILABLE_VICTIMS = {
    "random": Random_Guesser,
    "xgb": XGB
}