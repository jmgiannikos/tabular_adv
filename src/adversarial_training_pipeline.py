import sklearn as sk
from tabularbench.datasets import dataset_factory 
from sklearn.model_selection import cross_validate
from tabularbench.constraints.constraints_checker import ConstraintChecker
import skopt as sko
import numpy as np
import torch
import classifier_models as cls_model
import model_training_pipeline
import adversarial_models as adv_model

DEFAULT_CUDA_DEVICE = "cpu"
DEFAULT_SAFE_DIV_FACTOR = 0.00000001
DEFAULT_TOLERANCE = 0
DEFAULT_IMPERCEPTABILITY_WEIGHT = 0.5
DEFAULT_CONFIG = {
    "dataset_name": "url",
    "adv_cls": adv_model.Baseline_adv,
    "victim": cls_model.Random_Guesser
}
DEFAULT_TARGET_LABEL = 1
DEFAULT_NCALLS = 10

# TODO: Handling of mutable and non mutable features -> seems like constraint checker already checks mutability constraints
# TODO: may have to introduce special handelling to translate dataframe to numpy array (or even tensor, if we are too cpu constrained?) -> seems to run fine currently. May have to revisit when using real models

# pipeline doing crossval for an adversarial against a victim on a dataset
def pipeline(dataset_name, adv_class, victim):
    dataset = dataset_factory.get_dataset(dataset_name)
    search_dimensions = adv_class.SEARCH_DIMS()
    x, y = dataset.get_x_y()
    x, y = select_non_target_labels(x,y)
    optimization_wrapper = HyperparamOptimizerWrapper(adv_class, search_dimensions, victim, dataset)
    scores = cross_validate(optimization_wrapper, x, y, scoring=scorer, error_score="raise")
    return scores, optimization_wrapper

def select_non_target_labels(x,y,target_label=DEFAULT_TARGET_LABEL):
    not_target_label_map = y != target_label
    x = x[not_target_label_map]
    y = y[not_target_label_map] #kinda redundant because this should now all be the same label, but whatever
    return x, y

# NOTE: this is kinda unclean, since we read a lot of dataset properties from the attacker wrapper, which is unintuitive. Should however be clean technically.
def scorer(attacker, X, y): #takes an initialized model and gives it a score. Must be compatible with sklearns crossvalidate
    adv_samples = attacker.attack(X)
    pruned_adv_samples, pruned_X, pruned_y = attacker.prune_constraint_violations(adv_samples, X, y)
    num_constraint_violations = np.shape(y)[0] - np.shape(pruned_y)[0] # value mostly irrellevant currently
    victim_predictions = attacker.victim.predict(pruned_adv_samples) # assumes predict produces binary labels
    flipped_labels = pruned_y == victim_predictions
    success_rate = np.sum(flipped_labels.astype(int))/np.shape(y)[0] # assumes 0 is the main index of y. Samples that violate constraints are not counted, even if the label flips
    imperceptability = attacker.get_imperceptability(pruned_adv_samples, pruned_X)
    return {"constraint violations": num_constraint_violations,
            "success rate": success_rate,
            "imperceptability": imperceptability}

class HyperparamOptimizerWrapper(sk.base.BaseEstimator):
    def __init__(self, model_class, search_dimensions, victim, dataset, tolerance=DEFAULT_TOLERANCE): #TODO: passing around the whole dataset like this is cumbersome, instead just pass ranges and feature types
        self.model_class = model_class # model should be a class and have a static method that marks it as trainable or non-trainable called trainable
        self.search_dimensions = search_dimensions # see skopt.gp_minimize for the specifics
        self.best_params = None 
        self.victim = victim
        self.dataset = dataset
        self.tolerance = tolerance
        self.constraints = dataset.get_constraints()
        self.constraints_checker = ConstraintChecker(self.constraints, tolerance=tolerance)
        self.gower_dist = Gower_dist(dataset)
        self.metadata = dataset.get_metadata(only_x=True)
        if model_class.TRAINABLE():
            self.fitted_model = None

    def attack(self, X):
        if self.model_class.TRAINABLE():
            attacker_model = self.fitted_model
        else:
            attacker_model = self.model_class(self.victim, self.constraints, self.metadata,*self.best_params)
        adv_samples = attacker_model.attack(X)
        return adv_samples

    def fit(self, X, y, ncalls=DEFAULT_NCALLS):
        # these are made object variables, so that other functions in this class can access them without needing to have them passed as args, which sk.crossvalidate doesnt do for instance
        self.X = X 
        self.y = y
        if self.model_class.TRAINABLE():
            result_dict = sko.gp_minimize(self.trainable_objective, self.search_dimensions)
            self.fitted_model = self.model_class(self.victim, self.constraints, self.metadata, n_calls=ncalls, *result_dict.x).fit(X)
        else:
            result_dict = sko.gp_minimize(self.static_objective, self.search_dimensions, n_calls=ncalls)
        self.best_params = [result_dict.x] # this is wrapped in a list, because of the weird parameter shape the optimization function uses internally. This allows consistency between the modes
        return self
        
    def trainable_objective(self, *args):
        model = self.model_class(self.victim, self.constraints, self.metadata, *args) #args (hyperparameters) that the model takes need to align with the search dimensions
        scores = cross_validate(model, self.X, self.y, scoring=self.objective)
        score = np.mean(scores["test_score"])
        return score

    def static_objective(self, *args):
        model = self.model_class(self.victim, self.constraints, self.metadata, *args) #args (hyperparameters) that the model takes need to align with the search dimensions
        score = self.objective(model, self.X, self.y)
        return score

    def objective(self, model, val_X, val_y, imperceptability_weighting=DEFAULT_IMPERCEPTABILITY_WEIGHT): # X here are the adv samples (the naming scheme is so we can use the crossval function from sklearn for trainable_objective)
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
        gower_distances = self.gower_dist.dist_func(adv_samples, base_samples, pairwise=True)
        # calculate mean distance to original sample in absence of a better idea 
        if gower_distances.shape[0] != 0:
            mean_dist = torch.sum(gower_distances).item()/gower_distances.shape[0]
        else:
            mean_dist = 0
        return mean_dist

    def prune_constraint_violations(self, adv_samples, X, y):
        sample_viability_map = self.constraints_checker.check_constraints(X.to_numpy(), adv_samples.to_numpy())
        bool_sample_viability_map = sample_viability_map != 0
        selected_adv_samples = adv_samples[bool_sample_viability_map] #assumes shape nxm where n is number of samples and m is number of features per sample
        selected_x = X[bool_sample_viability_map] #assumes shape nxm where n is number of samples and m is number of features per sample
        selected_y = y[bool_sample_viability_map] # assumes y to be one dimensional 
        return selected_adv_samples, selected_x, selected_y

class Gower_dist: #TODO: passing around the whole dataset like this is cumbersome, instead just pass ranges and feature types
    # expecting weighting dict of form feature_name: weight
    # expecting x and y to be numpy arrays of shape (samples, features)
    def __init__(self, dataset, weighting_dict=None, cuda_device=DEFAULT_CUDA_DEVICE, safe_div_factor=DEFAULT_SAFE_DIV_FACTOR):
        self.cuda_device = cuda_device

        metadata = dataset.get_metadata(only_x=True)
        self.feature_list, self.num_idxs, self.num_features, self.cat_idxs, self.cat_features = self.calculate_index_lists(metadata)
        
        self.x, _ = dataset.get_x_y()
        if len(self.cat_idxs) > 0:
            self.x_num, self.x_cat = self.split_num_and_cat(self.x, self.num_idxs, self.cat_idxs)
        else:
            x = self.x
            if not isinstance(x, np.ndarray):
                x = x.to_numpy()
            self.x_num = torch.from_numpy(x)
            self.x_cat = None
        
        self.num_ranges = self.get_ranges(self.x_num, safe_div_factor=safe_div_factor)

        if weighting_dict is not None:
            self.weighting_dict=weighting_dict
            self.weighting_tensor = torch.Tensor([weighting_dict[feature] for feature in self.feature_list]).to(self.cuda_device)
        else:
            self.weighting_dict = {}
            for feature in self.feature_list:
                self.weighting_dict[feature] = 1/len(self.feature_list)
            self.weighting_tensor = torch.Tensor([1/len(self.feature_list)]*len(self.feature_list)).to(self.cuda_device)
        
    def get_ranges(self, num_features, safe_div_factor=DEFAULT_SAFE_DIV_FACTOR):
        max_vals, _ = torch.max(num_features, axis=0)
        min_vals, _ = torch.min(num_features, axis=0)
        dist = torch.abs(torch.sub(max_vals, min_vals))
        
        zero_mask = torch.eq(dist, 0)
        safe_adder = torch.mul(zero_mask.type(dist.dtype), safe_div_factor)
        dist = torch.add(dist, safe_adder)
        
        return dist
        
    def calculate_index_lists(self, metadata):
        feature_list = metadata["feature"].tolist()
        cat_idxs = []
        cat_features = []
        for idx in range(0,len(feature_list)):
            feature = feature_list[idx]
            if metadata.query("feature == @feature")["type"].to_list()[0] == "cat":
                cat_idxs.append(idx)
                cat_features.append(feature)

        _num_feat_list = [(enum, x) for (enum, x) in enumerate(feature_list) if enum not in cat_idxs]
        num_idxs = [idx for idx in range(0, len(feature_list)) if idx not in cat_idxs]
        num_features = [x for (enum, x) in enumerate(feature_list) if enum not in cat_idxs]
         # reorganize feature list
        feature_list = num_features + cat_features

        return feature_list, num_idxs, num_features, cat_idxs, cat_features
    
    def split_num_and_cat(self, data, num_idxs, cat_idxs):         
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()
        data_num = torch.from_numpy(data[:, np.array(num_idxs)]).to(self.cuda_device)
        data_cat = torch.from_numpy(data[:, np.array(cat_idxs)]).to(self.cuda_device)
        return data_num, data_cat

    def expand_tensors(self, x, y):
        n = x.size(dim=0)
        m = y.size(dim=0)
        assert x.size(dim=1) == y.size(dim=1)
        f = x.size(dim=1)

        x = torch.unsqueeze(x, dim=1)
        y = torch.unsqueeze(y, dim=0)

        x = x.expand((n,m,f))
        y = y.expand((n,m,f))

        return x, y

    def expand_to(self, target, source):
        target_dims = target.size()
        source_dims = source.size()
        assert len(source_dims) <= len(target_dims)
        for dim in range(0, len(target_dims)-len(source_dims)):
            source = torch.unsqueeze(source, dim=0)
        source = source.expand(target_dims)
        return source
    
    def get_num_dists(self, x, y, pairwise):
        if not pairwise:
            x, y = self.expand_tensors(x, y)

        divisor = self.expand_to(x,self.num_ranges)
        dist = torch.div(torch.abs(torch.sub(x, y)), divisor)
        return dist

    def get_cat_dists(self, x, y, pairwise):
        if not pairwise:
            x, y = self.expand_tensors(x, y)

        eq_tensor = torch.eq(x,y)
        dist = eq_tensor.type(x.dtype)
        return dist

    def dist_func(self, y, x=None, pairwise=False):
        if len(self.cat_idxs) != 0:
            if x is None:
                x_num = self.x_num
                x_cat = self.x_cat
            else:
                x_num, x_cat = self.split_num_and_cat(x, self.num_idxs, self.cat_idxs)
    
            y_num, y_cat = self.split_num_and_cat(y, self.num_idxs, self.cat_idxs)
    
            if pairwise:
                assert x_num.shape == y_num.shape
                assert y_cat.shape == x_cat.shape

            dists_num = self.get_num_dists(x_num, y_num, pairwise)
            dists_cat = self.get_cat_dists(x_cat, y_cat, pairwise)
    
            dists = torch.cat([dists_num, dists_cat], dim=2)

        else: 
            if x is None:
                x_num = self.x_num
            else:
                if not isinstance(x, np.ndarray):
                    x = x.to_numpy()
                x_num = torch.from_numpy(x)
            
            if not isinstance(y, np.ndarray):
                y = y.to_numpy()
            y_num = torch.from_numpy(y).to(self.cuda_device)
            
            if pairwise:
                assert x_num.shape == y_num.shape

            dists = self.get_num_dists(x_num, y_num, pairwise)
            
        weights = self.expand_to(dists,self.weighting_tensor)

        weighted_dists = torch.mul(dists, weights)
        if not pairwise:
            dist_mat = torch.sum(weighted_dists, dim=2, keepdim=False)
        else:
            dist_mat = torch.sum(weighted_dists, dim=1, keepdim=False)

        return dist_mat
    
if __name__ == "__main__":
    config=DEFAULT_CONFIG
    dataset_name = config["dataset_name"]
    adversarial_class = config["adv_cls"]
    victim = config["victim"]
    if not isinstance(victim, cls_model.Cls_model):
        victim = model_training_pipeline.train_pipeline(victim, dataset_name)
    results = pipeline(dataset_name, adversarial_class, victim)
    print(results[0])
        
        

