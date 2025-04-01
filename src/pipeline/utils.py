import torch
import numpy as np
from defaults import DEFAULTS
import os
from itertools import permutations
import random
import pandas
import math

def is_not_in(samples, check_against_samples):
    if len(check_against_samples) == 0:
        return torch.BoolTensor([True]*samples.shape[0])
    check_against_samples = torch.stack(check_against_samples, dim=0)
    desired_shape = (samples.shape[0], check_against_samples.shape[0], samples.shape[1])
    check_against_samples = torch.unsqueeze(check_against_samples, dim=0).expand(desired_shape)
    samples = torch.unsqueeze(samples, dim=1).expand(desired_shape)
    equality_matrix = torch.all(torch.eq(samples, check_against_samples), dim=2)
    is_not_in_map = torch.logical_not(torch.any(equality_matrix, dim=1))
    return is_not_in_map

def prune_labels(x,y,victim,target_label=DEFAULTS["target_label"], get_prune_map=False):
    if not get_prune_map:
        not_target_label_map = y != target_label
        x = x[not_target_label_map]
        y = y[not_target_label_map] #kinda redundant because this should now all be the same label, but whatever
        
        # get victim predictions and prune every sample where the prediction is already the target label (a misclassification occurred)
        y_preds = victim.predict(x)
        not_target_predict_map = y_preds != target_label
        x = x[not_target_predict_map]
        y = y[not_target_predict_map]
        return x, y
    else:
        not_target_label_map = y != target_label
        y_preds = victim.predict(x)
        not_target_predict_map = y_preds != target_label
        prune_map = np.logical_and(not_target_predict_map, not_target_label_map)
        x=x[prune_map]
        y=y[prune_map]
        return x, y, prune_map
    
def find_unused_path(base_path, iterator):
    found = False
    while not found:
        path = f"{base_path}_{iterator}"
        if os.path.exists(path):
            iterator += 1
        else:
            found=True
    return path

def log_dataset_metrics(dist_metric, path, x, y):
    class_zero_samples = x[y==0]
    class_one_samples = x[y==1]

    feature_wise_dists_zero = get_feature_wise_dists(class_zero_samples, class_zero_samples, dist_metric)
    feature_wise_dists_one = get_feature_wise_dists(class_one_samples, class_one_samples, dist_metric)
    feature_wise_dists_mixed = get_feature_wise_dists(class_one_samples, class_zero_samples, dist_metric)

    torch.save(feature_wise_dists_mixed, f"{path}mixed_dataset_feat_dists")
    torch.save(feature_wise_dists_zero, f"{path}zero_dataset_feat_dists")
    torch.save(feature_wise_dists_one, f"{path}one_dataset_feat_dists")


#NOTE: currently just very rough subsampling
def get_feature_wise_dists(data_a, data_b, dist_metric):
    if isinstance(data_a, pandas.DataFrame):
        data_a = torch.from_numpy(data_a.values)
    if isinstance(data_b, pandas.DataFrame):
        data_b = torch.from_numpy(data_b.values)

    idx_list = list(range(len(data_a)))
    feature_wise_distances = []
    for _ in range(DEFAULTS["dataset_metric_subsampling_steps"]):
        random.shuffle(idx_list)
        idx_perm = torch.Tensor(idx_list).long()
        permutation = data_a[idx_perm]

        if len(permutation) > len(data_b):
            permutation = permutation[:len(data_b)]
        elif len(data_b) > len(permutation):
            data_b = data_b[:len(permutation)]

        feature_wise_distances.append(dist_metric.get_feature_wise_distances(data_b, permutation))
    feature_wise_distance_tensor = torch.cat(feature_wise_distances, dim=0)
    return feature_wise_distance_tensor

def sample_even(x, y, size):
    x_ones = x[y==1]
    x_zeros = x[y==0]

    x_ones = x_ones[:math.ceil(size/2)]
    x_zeros = x_zeros[:math.floor(size/2)]
    x = np.append(x_ones, x_zeros, axis=0)
    y = np.array([1]*len(x_ones) + [0]*len(x_zeros))

    permutation = list(range(len(y)))
    random.shuffle(permutation)
    x = x[permutation]
    y = y[permutation]
    return x, y


