import torch 
import numpy as np
from defaults import DEFAULTS

class Gower_dist:
    # expecting weighting dict of form feature_name: weight
    # expecting x and y to be numpy arrays of shape (samples, features)
    def __init__(self, x, metadata, weighting_dict=None, cuda_device=DEFAULTS["cuda_device"], safe_div_factor=DEFAULTS["safe_div_factor"], dynamic=False):
        self.cuda_device = cuda_device
        self.dynamic = dynamic
        self.safe_div_factor = safe_div_factor
        metadata = metadata
        self.feature_list, self.num_idxs, self.num_features, self.cat_idxs, self.cat_features = self.calculate_index_lists(metadata)
        
        if len(self.cat_idxs) > 0:
            x_num, x_cat = self.split_num_and_cat(x, self.num_idxs, self.cat_idxs)
        else:
            x_num = x
            if not isinstance(x, np.ndarray):
                x_num = x_num.to_numpy()
            x_cat = None

        if not dynamic:
            x_num = torch.from_numpy(x_num)
            self.num_ranges = self.get_ranges(x_num, safe_div_factor=safe_div_factor)
        else:
            self.x_num = x_num

        if weighting_dict is not None:
            self.weighting_dict=weighting_dict
            self.weighting_tensor = torch.Tensor([weighting_dict[feature] for feature in self.feature_list]).to(self.cuda_device)
        else:
            self.weighting_dict = {}
            for feature in self.feature_list:
                self.weighting_dict[feature] = 1/len(self.feature_list)
            self.weighting_tensor = torch.Tensor([1/len(self.feature_list)]*len(self.feature_list)).to(self.cuda_device)
        
    def get_ranges(self, num_features, safe_div_factor=DEFAULTS["safe_div_factor"]):
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

        if not self.dynamic:
            divisor = self.expand_to(x,self.num_ranges)
        else:
            x_tensor = np.append(self.x_num, x, axis=0) #collect all seen data points
            x_tensor = torch.from_numpy(np.append(x_tensor, y, axis=0))
            divisor = self.get_ranges(x_tensor, safe_div_factor=self.safe_div_factor) 

        dist = torch.div(torch.abs(torch.sub(x, y)), divisor)
        return dist

    def get_cat_dists(self, x, y, pairwise):
        if not pairwise:
            x, y = self.expand_tensors(x, y)

        eq_tensor = torch.eq(x,y)
        dist = eq_tensor.type(x.dtype)
        return dist

    def dist_func(self, y, x, pairwise=False):
        if len(self.cat_idxs) != 0:
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