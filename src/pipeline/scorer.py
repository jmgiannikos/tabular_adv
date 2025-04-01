from tabularbench.constraints.constraints_checker import ConstraintChecker
import numpy as np
import torch
import wandb
from hyperparam_opt_wrapper import HyperparamOptimizerWrapper
from defaults import Log_styles
from utils import prune_labels, find_unused_path
from sklearn.metrics import confusion_matrix
import os
import json
from adversarial_models import Adv_model

# NOTE: this is kinda unclean, since we read a lot of dataset properties from the attacker wrapper, which is unintuitive. Should however be clean technically.
class Scorer():
    DEFAULTS = {
        "tolerance": 0
    }
    def __init__(self, results_path, name="", log_style=Log_styles.NOLOG, constraints=None, distance_metric=None, feature_names=None):
        self.name=name
        self.call_counter = 0
        self.constraints=constraints
        self.distance_metric = distance_metric

        self.log_style = log_style

        if log_style != Log_styles.NOLOG:
            self.feature_wise_distances = []
            self.feature_names = feature_names

        self.wandb_log = log_style == Log_styles.WANDB or log_style == Log_styles.BOTH
        if self.wandb_log:
            self.logged_tables = {}

        self.local_log = log_style == Log_styles.LOCAL or log_style == Log_styles.BOTH
        if self.local_log:
            self.results_path = results_path

    def dump_logs(self, lastcall=False):
        if self.local_log:
            tensor_elements = []
            for element in self.feature_wise_distances: # NOTE: this may be redundant, but better safe than sorry (?)
                if not isinstance(element, torch.Tensor):
                    element = torch.Tensor(element)
                if len(element.shape) < 2:
                    element = torch.unsqueeze(element, dim=0)
                tensor_elements.append(element)

            feat_dist_tensor = torch.cat(tensor_elements, dim=0)

            scorer_id = hex(id(self))[2:].upper()
            if lastcall:
                os.remove(f"{self.results_path}[{scorer_id}]tmp_feat_dists.pt")
                log_path = f"{self.results_path}[{scorer_id}]feat_dists.pt"
                if os.path.exists(log_path):
                    log_path = find_unused_path(log_path, iterator=1)
            else:
                log_path = f"{self.results_path}[{scorer_id}]tmp_feat_dists.pt"

            torch.save(feat_dist_tensor, log_path)
        else:
            raise UserWarning("tried to dump scorer logs, when local logging was disabled")

    def score(self, attacker, X, y): #takes an initialized model and gives it a score. Must be compatible with sklearns crossvalidate
        full_x, full_y = X, y
        X, y, label_pruning_map = prune_labels(X, y, attacker.victim, get_prune_map=True)
        
        if self.log_style != Log_styles.NOLOG and self.feature_names is None:
            self.feature_names = range(X.shape[1])

        adv_samples = attacker.attack(X)   

        # LOG FEATURE WISE DISTANCES #
        if self.log_style != Log_styles.NOLOG:
            if isinstance(attacker, HyperparamOptimizerWrapper):
                self.feature_wise_distances.append(attacker.distance_metric.get_feature_wise_distances(X, adv_samples))
            elif self.distance_metric is not None:
                self.feature_wise_distances.append(self.distance_metric.get_feature_wise_distances(X, adv_samples))

        if isinstance(attacker, HyperparamOptimizerWrapper):
            pruned_adv_samples, pruned_X, pruned_y, const_violation_prune_map = attacker.prune_constraint_violations(adv_samples, X, y, get_prune_map=True)
        elif self.constraints is not None:
            if isinstance(adv_samples, torch.Tensor):
                adv_samples = adv_samples.numpy(True)
            if isinstance(X,  torch.Tensor):
                X = X.numpy(True)
            constraints_checker = ConstraintChecker(self.constraints, tolerance=self.DEFAULTS["tolerance"])
            sample_viability_map = constraints_checker.check_constraints(X, adv_samples) # True for every sample that breaches no constraints
            bool_sample_viability_map = sample_viability_map == 1
            pruned_adv_samples = adv_samples[bool_sample_viability_map] #assumes shape nxm where n is number of samples and m is number of features per sample
            pruned_X = X[bool_sample_viability_map] #assumes shape nxm where n is number of samples and m is number of features per sample
            pruned_y = y[bool_sample_viability_map] # assumes y to be one dimensional 
            const_violation_prune_map = bool_sample_viability_map
        else:
            pruned_adv_samples = adv_samples
            pruned_X = X
            pruned_y = y
            const_violation_prune_map = np.array([True]*len(y)) 
        
        num_constraint_violations = np.shape(y)[0] - np.shape(pruned_y)[0] # value mostly irrellevant currently
        constraint_violation_ratio = num_constraint_violations / np.shape(y)[0]
        victim_predictions = attacker.victim.predict(pruned_adv_samples) # assumes predict produces binary labels
        victim_original_predictions = attacker.victim.predict(pruned_X) # get predictions of the model on the original samples #NOTE: it may be reasonable to prune the data so this only contains samples that the model predicts as 0 (correctly)
        flipped_labels = victim_original_predictions != victim_predictions
        success_rate = np.sum(flipped_labels.astype(int))/np.shape(y)[0] # assumes 0 is the main index of y. Samples that violate constraints are not counted, even if the label flips
        if isinstance(attacker, HyperparamOptimizerWrapper):
            imperceptability = -attacker.get_imperceptability(pruned_adv_samples, pruned_X) # NOTE: negated because imperceptability is negated gower
        elif self.distance_metric is not None:
            gower_distances = self.distance_metric.dist_func(adv_samples, X, pairwise=True)
            # calculate mean distance to original sample in absence of a better idea 
            if gower_distances.shape[0] != 0:
                imperceptability = torch.sum(gower_distances).item()/gower_distances.shape[0]
            else:
                imperceptability = 0
        else:
            imperceptability = np.nan

        if self.wandb_log:
            self.update_tables(f"{self.name}: constraint violations", self.call_counter, constraint_violation_ratio)
            self.update_tables(f"{self.name}: success rate", self.call_counter, success_rate)
            self.update_tables(f"{self.name}: imperceptability", self.call_counter, imperceptability)

        if self.local_log:
            self.dump_logs() # regularly dump logs. Should overwrite old logs, when updating (all logs are also held locally)

        pruning_map = self.merge_pruning_maps(label_pruning_map, const_violation_prune_map)
        before_conf_mat, after_conf_mat = self.get_conf_mats(full_x, full_y, attacker.victim, pruned_adv_samples, pruning_map)
        
        if isinstance(attacker, Adv_model):
            attacker_id = hex(id(attacker))[2:].upper()
        else:
            attacker_id = hex(id(attacker.attacker_model))[2:].upper() #should allow linking between adversarial logs and conf mats
        self.log_conf_mats(before_conf_mat, after_conf_mat, attacker_id)

        self.call_counter += 1
        return {"constraint violations": num_constraint_violations,
                "success rate": success_rate,
                "imperceptability": imperceptability}
    
    def merge_pruning_maps(self, label_map, constraint_map):
        assert np.sum(label_map) == len(constraint_map) #pre execution sanity check
        merged_map = []
        constraint_map_idx = 0
        for item in label_map:
            if item:
                merged_map.append(constraint_map[constraint_map_idx])
                constraint_map_idx += 1
            else:
                merged_map.append(False)
        merged_map = np.array(merged_map)
        assert constraint_map_idx == len(constraint_map) #post execution sanity check
        return merged_map

    def log_conf_mats(self, before, after, attacker_id, iterator=0):
        if self.local_log:
            if iterator == 0:
                file_path = f"{self.results_path}[{attacker_id}]confusion_matrices"
            else:
                file_path = f"{self.results_path}[{attacker_id}_{iterator}]confusion_matrices"

            if os.path.exists(file_path):
                iterator += 1
                self.log_conf_mats(before, after, attacker_id, iterator)
            else:
                conf_mat_dict = {
                    "before": before.tolist(),
                    "after": after.tolist()
                }

                log_obj = json.dumps(conf_mat_dict)
                with open(file_path, "w") as outfile:
                    outfile.write(log_obj)



    def get_conf_mats(self, x, y, victim, adv_samples, pruning_map):
        # get before conf mat
        victim_preds = victim.predict(x)
        before_conf_mat = confusion_matrix(y, victim_preds)

        # get after conf mat
        pruned_samples_map = np.logical_not(pruning_map)
        base_x = x[pruned_samples_map]
        base_y = y[pruned_samples_map]
        adv_y = y[pruning_map]
        joint_x = np.append(base_x, adv_samples, axis=0)
        joint_y = np.append(base_y, adv_y, axis=0)
        attacked_victim_preds = victim.predict(joint_x)
        after_conf_mat = confusion_matrix(joint_y, attacked_victim_preds)
        return before_conf_mat, after_conf_mat


    def update_tables(self, chart_name, fold, value):
        fold_name = F"fold_{fold}"
        if chart_name in self.logged_tables.keys():
            table = self.logged_tables[chart_name]
            table.add_data(fold_name, value)
        else:
            table = wandb.Table(columns=["fold", "value"], data=[[fold_name, value]])
            self.logged_tables[chart_name] = table
    
    def log_bar_chart(self):
        if self.wandb_log:
            for chart_name in self.logged_tables.keys():
                wandb.log(
                    {
                        chart_name: wandb.plot.bar(
                            self.logged_tables[chart_name], "fold", "value", title=chart_name
                        )
                    }
                )
        else:
            raise UserWarning("tried to wandb log bar charts, when wandb logging was disabled")
        
    def wandb_log_feat_analysis(self):
        if self.wandb_log:
            tensor_elements = []
            for element in self.feature_wise_distances: # NOTE: this may be redundant, but better safe than sorry (?)
                if not isinstance(element, torch.Tensor):
                    element = torch.Tensor(element)
                if len(element.shape) < 2:
                    element = torch.unsqueeze(element, dim=0)
                tensor_elements.append(element)

            feat_dist_tensor = torch.cat(tensor_elements, dim=0)
            feat_adjusted_tensor = torch.logical_not(torch.eq(feat_dist_tensor, 0)).long()

            adjusted_feature_counts = torch.sum(feat_adjusted_tensor, dim=0)
            sample_number = feat_dist_tensor.shape[0]
            adjusted_feature_ratios = torch.div(adjusted_feature_counts, sample_number)
            avg_adjusted_tensor = torch.mean(feat_dist_tensor, dim=0)

            adjusted_feature_counts_table_values = [[self.feature_names[idx], adjusted_feature_counts[idx]] for idx in range(len(self.feature_names))]
            adjusted_feature_avgs_table_values = [[self.feature_names[idx], avg_adjusted_tensor[idx]] for idx in range(len(self.feature_names))]

            adjusted_feature_counts_table = wandb.Table(columns=["feature", "count"], data=adjusted_feature_counts_table_values)
            adjusted_feature_ratios_table = wandb.Table(columns=["feature", "percentage"], data=adjusted_feature_ratios)
            avg_adjusted_tensor_table = wandb.Table(columns=["feature", "avg_dist"], data=adjusted_feature_avgs_table_values)

            wandb.log({"adjusted_features": wandb.plot.bar(adjusted_feature_ratios_table, "feature", "count", title="adjusted_features")})
            wandb.log({"feature_adjustments": wandb.plot.bar(avg_adjusted_tensor_table, "feature", "avg_dist", title="feature_adjustments")})
        else:
            raise UserWarning("tried to wandb log feature analysis, when wandb logging was disabled")
