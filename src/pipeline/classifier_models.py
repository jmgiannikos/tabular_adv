from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
import xgboost as xgb
import skopt as sko
import tabpfn
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
import torch
from defaults import DEFAULTS as global_defaults

# probably not going to need this, since we want to use fully implemented models (preferrably already pre-trained, but that may be a bit too optimistic). May serve as a wrapper
class Cls_model(ABC):
    DEFAULTS = None
    def __init__(self, tune_model, scaler):
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

    @abstractmethod
    def predict_proba(self, x):
        pass

class Random_Guesser(Cls_model):
    def __init__(self, tune_model, scaler):
        super().__init__()
        self.guesser = np.random.default_rng()

    def tune_hyperparameters(self, x, y, ncalls):
        return self

    def fit(self, x, y):
        return self
    
    def predict(self, x):
        guesses = self.guesser.integers(0,2,size=x.shape[0])
        return guesses
    
    def predict_proba(self, x):
        return super().predict_proba(x)

# TODO: Early stopping not implemented 
# TODO: Make all parameters searchable by the hyperparameter optimization method
class XGB(Cls_model):
    DEFAULTS = {
        "ncalls": 10,
        "xgb_rounds": 1
    }
    def __init__(self, tune_model=False, scaler=None):
        super().__init__(tune_model, scaler)
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
    
class TabPFN(Cls_model):
    DEFAULTS = {
        "ncalls": 10,
        "batch_size": 1000,
        "batch_process": False
    }
    def __init__(self, tune_model=False, scaler=None):
        super().__init__(tune_model, scaler)
        self.parameter_list = []
        self.search_dimensions = []
        self.best_parameters = None
        self.tune_model = tune_model

    def fit(self, x, y, hyperparameters=None):
        if self.DEFAULTS["batch_process"] and len(x) > self.DEFAULTS["batch_size"]: # training sets get reduced to fit
            x = x[:self.DEFAULTS["batch_size"]]
            y = y[:self.DEFAULTS["batch_size"]]
        if not self.tune_model:
            if x.shape[1] > 100:
                model = tabpfn.TabPFNClassifier(device=global_defaults["device"], subsample_features=True)
            else:
                model = tabpfn.TabPFNClassifier(device=global_defaults["device"])
            self.model = model.fit(x,y)
        else:
            if hyperparameters is None:
                if self.best_parameters is None:
                    raise ValueError("hyperparameters not initialized")
                else:
                    hyperparameters = self.best_parameters
            
            if x.shape[1] > 100:
                model = tabpfn.TabPFNClassifier(device=global_defaults["device"], subsample_features=True, *hyperparameters)
            else:
                model = tabpfn.TabPFNClassifier(device=global_defaults["device"], *hyperparameters)
            self.model = model.fit(x,y)
        return self
    
    def predict(self, x):
        if self.DEFAULTS["batch_process"] and len(x) > self.DEFAULTS["batch_size"]: # test sets get processed batch wise
            ypreds = self.batch_process_wrapper(x, self.predict, y=None, kwargs={})
            return torch.cat(ypreds, dim=0).numpy()
        else:
            if len(x.shape) == 1:
                x = np.expand_dims(x, 0)
            ypred = self.model.predict(x)
            return ypred
    
    def predict_proba(self, x):
        if self.DEFAULTS["batch_process"] and len(x) > self.DEFAULTS["batch_size"]: # test sets get processed batch wise
            ypreds = self.batch_process_wrapper(x, self.predict_proba, y=None, kwargs={})
            return torch.cat(ypreds, dim=0).numpy()
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        ypob = self.model.predict_proba(x)
        return ypob
    
    def tune_hyperparameters(self, x, y, ncalls=DEFAULTS["ncalls"]):
        if self.DEFAULTS["batch_process"] and len(x) > self.DEFAULTS["batch_size"]: # training sets get reduced to fit
            x = x[:self.DEFAULTS["batch_size"]]
            y = y[:self.DEFAULTS["batch_size"]]
        self.x = x
        self.y = y 
        result_dict = sko.gp_minimize(self.objective, self.search_dimensions, n_calls=ncalls)
        self.best_parameters = [result_dict.x]
        self.fit(x,y)
        return self
    
    def objective(self, *args):
        parameters = {}
        parameters = self.dictify_params(*args)
        model = tabpfn.TabPFNClassifier(**parameters)
        scores = cross_validate(model, self.x, self.y, scoring="f1")
        score = np.mean(scores["test_score"])
        return score

    def dictify_params(self, *parameters):
        for i, parameter_name in enumerate(self.parameter_list): # had to use native python for loop. sadge :(
            parameters[parameter_name] = parameters[i]
        return parameters
    
    def batch_process_wrapper(self, x, func, y, kwargs):
        fin = False
        start_idx = 0
        results = []
        while not fin:
            end_idx = start_idx + self.DEFAULTS["batch_size"]
            if end_idx >= x.shape[0]:
                fin = True
                kwargs["x"] = x[start_idx:]
                if y is not None:
                    kwargs["y"] = y[start_idx:]
            else:
                kwargs["x"] = x[start_idx:end_idx]
                if y is not None:
                    kwargs["y"] = y[start_idx:end_idx]
            
            results.append(torch.from_numpy(func(**kwargs)))
            start_idx += self.DEFAULTS["batch_size"]
        return results

class RidgeCls(Cls_model):
    DEFAULTS = None
    def __init__(self, tune_model=False, scaler=None,sigmoidize_confidence=True):
        super().__init__(tune_model, scaler)
        self.parameter_list = []
        self.search_dimensions = []
        self.best_parameters = None
        self.tune_model = tune_model
        self.sigmoidize_confidence = sigmoidize_confidence
        self.scaler = scaler

    def tune_hyperparameters(self, x, y, ncalls):
        self.x = x
        self.y = y 
        result_dict = sko.gp_minimize(self.objective, self.search_dimensions, n_calls=ncalls)
        self.best_parameters = [result_dict.x]
        self.fit(self.scaler.transform(x),y)
        return self

    def label_translate(self, labels, to_ridge=False):
        if to_ridge:
            zero_map = (labels == 0).astype(np.int32)
            labels = labels+(zero_map*(-1))
        else:
            labels = (labels == 1).astype(np.int32)
        return labels

    def fit(self, x, y):
        y = self.label_translate(y, to_ridge=True)
        if not self.tune_model:
            model = RidgeClassifier()
            self.model = model.fit(self.scaler.transform(x),y)
        else:
            if hyperparameters is None:
                if self.best_parameters is None:
                    raise ValueError("hyperparameters not initialized")
                else:
                    hyperparameters = self.best_parameters
            
            model = RidgeClassifier(*hyperparameters)
            self.model = model.fit(self.scaler.transform(x),y)
        return self

    def predict(self, x):
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        ypred = self.model.predict(self.scaler.transform(x))
        ypred = self.label_translate(ypred, to_ridge=False)
        return ypred

    def predict_proba(self, x):
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        confidence_scores = self.model.decision_function(self.scaler.transform(x))
        confidence_scores = torch.from_numpy(confidence_scores)
        one_prob_scores = torch.special.expit(confidence_scores) #map confidence scores to scale between 1 and 0. Treat as probability of class 1. (steepness of sigmoid curve is kept as default, but may impact behavior)
        zero_prob_scores = torch.subtract(torch.Tensor([1]*one_prob_scores.shape[0]).to(global_defaults["device"]), one_prob_scores)
        return torch.stack((zero_prob_scores, one_prob_scores), dim=1).numpy() # normalize so sum is one. May already happen through sigmoid, but better safe than sorry

    def objective(self, *args):
        parameters = {}
        parameters = self.dictify_params(*args)
        model = RidgeClassifier(**parameters)
        scores = cross_validate(model, self.x, self.y, scoring="f1")
        score = np.mean(scores["test_score"])
        return score
    
    def dictify_params(self, *parameters):
        for i, parameter_name in enumerate(self.parameter_list): # had to use native python for loop. sadge :(
            parameters[parameter_name] = parameters[i]
        return parameters
    
class MlpCls(Cls_model):
    DEFAULTS = None
    def __init__(self, tune_model=False, scaler=None):
        super().__init__(tune_model, scaler)
        self.parameter_list = []
        self.search_dimensions = []
        self.best_parameters = None
        self.tune_model = tune_model
        self.scaler = scaler

    def tune_hyperparameters(self, x, y, ncalls):
        self.x = x
        self.y = y 
        result_dict = sko.gp_minimize(self.objective, self.search_dimensions, n_calls=ncalls)
        self.best_parameters = [result_dict.x]
        self.fit(self.scaler.transform(x),y)
        return self

    def fit(self, x, y):
        if not self.tune_model:
            model = MLPClassifier(solver="lbfgs", alpha=0.00001)
            self.model = model.fit(self.scaler.transform(x),y)
        else:
            if hyperparameters is None:
                if self.best_parameters is None:
                    raise ValueError("hyperparameters not initialized")
                else:
                    hyperparameters = self.best_parameters
            
            model = MLPClassifier(*hyperparameters)
            self.model = model.fit(self.scaler.transform(x),y)
        return self

    def predict(self, x):
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        ypred = self.model.predict(self.scaler.transform(x))
        return ypred

    def predict_proba(self, x):
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        y_proba = self.model.predict_proba(self.scaler.transform(x))
        return y_proba

    def objective(self, *args):
        parameters = {}
        parameters = self.dictify_params(*args)
        model = MLPClassifier(**parameters)
        scores = cross_validate(model, self.x, self.y, scoring="f1")
        score = np.mean(scores["test_score"])
        return score
    
    def dictify_params(self, *parameters):
        for i, parameter_name in enumerate(self.parameter_list): # had to use native python for loop. sadge :(
            parameters[parameter_name] = parameters[i]
        return parameters

AVAILABLE_VICTIMS = {
    "random": Random_Guesser,
    "xgb": XGB,
    "tabpfn": TabPFN,
    "ridgecls": RidgeCls,
    "mlpcls": MlpCls
}