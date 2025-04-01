import torch as t
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pandas
import pandas
from hyperparam_opt_wrapper import HyperparamOptimizerWrapper
import json
from hyperparameter_search_policies import opt_types
import statistics
import torch

FULL = True
FILE_PATH = "/workspace/results/mlpcls_lcld/[run]2025-3-7_18:47:6_fixgrid_mlpcls_lcld_bounded_noround/"
FEATURE_NAMES = ['loan_amnt', 'term', 'int_rate', 'installment', 'sub_grade', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'purpose', 'dti', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status', 'application_type', 'mort_acc', 'pub_rec_bankruptcies', 'fico_score', 'month_of_year', 'ratio_loan_amnt_annual_inc', 'ratio_open_acc_total_acc', 'month_since_earliest_cr_line', 'ratio_pub_rec_month_since_earliest_cr_line', 'ratio_pub_rec_bankruptcies_month_since_earliest_cr_line', 'ratio_pub_rec_bankruptcies_pub_rec']
#['length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens', 'nb_at', 'nb_qm', 'nb_and', 'nb_or', 'nb_eq', 'nb_underscore', 'nb_tilde', 'nb_percent', 'nb_slash', 'nb_star', 'nb_colon', 'nb_comma', 'nb_semicolumn', 'nb_dollar', 'nb_space', 'nb_www', 'nb_com', 'nb_dslash', 'http_in_path', 'https_token', 'ratio_digits_url', 'ratio_digits_host', 'punycode', 'port', 'tld_in_path', 'tld_in_subdomain', 'abnormal_subdomain', 'nb_subdomains', 'prefix_suffix', 'random_domain', 'shortening_service', 'path_extension', 'nb_redirection', 'nb_external_redirection', 'length_words_raw', 'char_repeat', 'shortest_words_raw', 'shortest_word_host', 'shortest_word_path', 'longest_words_raw', 'longest_word_host', 'longest_word_path', 'avg_words_raw', 'avg_word_host', 'avg_word_path', 'phish_hints', 'domain_in_brand', 'brand_in_subdomain', 'brand_in_path', 'suspecious_tld', 'statistical_report', 'whois_registered_domain', 'domain_registration_length', 'domain_age', 'web_traffic', 'dns_record', 'google_index', 'page_rank']
#['loan_amnt', 'term', 'int_rate', 'installment', 'sub_grade', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'purpose', 'dti', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status', 'application_type', 'mort_acc', 'pub_rec_bankruptcies', 'fico_score', 'month_of_year', 'ratio_loan_amnt_annual_inc', 'ratio_open_acc_total_acc', 'month_since_earliest_cr_line', 'ratio_pub_rec_month_since_earliest_cr_line', 'ratio_pub_rec_bankruptcies_month_since_earliest_cr_line', 'ratio_pub_rec_bankruptcies_pub_rec']
#['length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens', 'nb_at', 'nb_qm', 'nb_and', 'nb_or', 'nb_eq', 'nb_underscore', 'nb_tilde', 'nb_percent', 'nb_slash', 'nb_star', 'nb_colon', 'nb_comma', 'nb_semicolumn', 'nb_dollar', 'nb_space', 'nb_www', 'nb_com', 'nb_dslash', 'http_in_path', 'https_token', 'ratio_digits_url', 'ratio_digits_host', 'punycode', 'port', 'tld_in_path', 'tld_in_subdomain', 'abnormal_subdomain', 'nb_subdomains', 'prefix_suffix', 'random_domain', 'shortening_service', 'path_extension', 'nb_redirection', 'nb_external_redirection', 'length_words_raw', 'char_repeat', 'shortest_words_raw', 'shortest_word_host', 'shortest_word_path', 'longest_words_raw', 'longest_word_host', 'longest_word_path', 'avg_words_raw', 'avg_word_host', 'avg_word_path', 'phish_hints', 'domain_in_brand', 'brand_in_subdomain', 'brand_in_path', 'suspecious_tld', 'statistical_report', 'whois_registered_domain', 'domain_registration_length', 'domain_age', 'web_traffic', 'dns_record', 'google_index', 'page_rank']
FILE_DIRS = [
    "/workspace/results/mlpcls_lcld/[run]2025-3-7_18:47:6_fixgrid_mlpcls_lcld_bounded_noround/",
    "/workspace/results/mlpcls_lcld/[run]2025-3-7_21:25:59_gridstep_mplcls_lcld_bounded/",
    "/workspace/results/mlpcls_lcld/[run]2025-3-8_22:22:38_adaptive_mlpcls_lcld_bounded/"
]
EXPERIMENT_NAME = "mlpcls_lcld_bounded"

def select_by_identifier(file_names, identifier):
    remaining_file_names = []
    selected_file_names = []
    for file_name in file_names:
        if identifier in file_name:
            selected_file_names.append(file_name)
        else:
            remaining_file_names.append(file_name)
    return selected_file_names, remaining_file_names

def update_search_run_grid(data, axes=None, quali_printed=None, adversarial_id=""):
    if axes is None:
        axes = {}
        label = ["prob_diff", "total_node_prob", "gower_dist", "const_loss", "cost_func"]
        titles = ["probability difference", "total target probability", "gower distance", "constraint loss", "cost function"]
        for idx in range(5):
            _, axis = plt.subplots(1,1)
            axes[label[idx]] = axis
            #axes[label[idx]].set_title(titles[idx])
            axes[label[idx]].set_xlabel("search step")
            axes[label[idx]].set_ylabel(titles[idx])

    for label in axes.keys():
        axes[label].plot(data["step"], data[label])
        axes[label].set_xticks(range(21))

        if quali_printed is not None:
            #plt.show(block=True)
            axes[label].figure.savefig(f"{FILE_PATH}[{adversarial_id}]{label}_quali_search_plot_{quali_printed}")
    if quali_printed is not None:
        plt.close("all")

    return axes

def display_search_runs(logs, qualitative_samples = 2):
    for log in logs:
        with open(log, 'r') as file:
            log_dict = json.load(file)
        adversarial_id = log.split("/")[-1].split("]")[0][1:]
        quali_printed = 0
        while quali_printed < qualitative_samples:
            search_data = log_dict[list(log_dict.keys())[quali_printed]]
            plot_ax = update_search_run_grid(search_data, quali_printed=quali_printed, adversarial_id=adversarial_id)
            quali_printed += 1
            
        plot_ax = update_search_run_grid(log_dict[list(log_dict.keys())[0]])
        for key in list(log_dict.keys())[1:]:
            plot_ax = update_search_run_grid(log_dict[key], plot_ax)
        
        for key in plot_ax.keys():
            ax = plot_ax[key]
            #plt.show(block=False)
            ax.figure.savefig(f"{FILE_PATH}[{adversarial_id}]{key}_quanti_search_plot_lines")   
            plt.close("all")

def get_best_node_metrics(log_dict, res_dict=None):
    if res_dict is None:
        gower_dists = []
        total_probs = []
        prob_diffs = []
        for key in log_dict.keys():
            eval_func_search_trace = log_dict[key]["cost_func"]
            best_idx = eval_func_search_trace.index(min(eval_func_search_trace))
            gower_dists.append(log_dict[key]["gower_dist"][best_idx])
            total_probs.append(log_dict[key]["total_node_prob"][best_idx])
            prob_diffs.append(log_dict[key]["prob_diff"][best_idx])
        return gower_dists, total_probs, prob_diffs
    else:
        for key in log_dict.keys():
            eval_func_search_trace = log_dict[key]["cost_func"]
            best_idx = eval_func_search_trace.index(min(eval_func_search_trace))
            res_dict["gower_dists"].append(log_dict[key]["gower_dist"][best_idx])
            res_dict["total_prob"].append(log_dict[key]["total_node_prob"][best_idx])
            res_dict["prob_diff"].append(log_dict[key]["prob_diff"][best_idx])
        return res_dict

def display_result_scatter(logs):
    for log in logs:
        with open(log, 'r') as file:
            log_dict = json.load(file)
        adversarial_id = log.split("/")[-1].split("]")[0][1:]
        gower_dists, total_probs, prob_diffs = get_best_node_metrics(log_dict)
        figure, axes = plt.subplots(1, 1)
        axes.scatter(gower_dists, total_probs)
        axes.set_title('gower by total probability')
        axes.set_xlabel("gower distance")
        axes.set_ylabel("total target probability")
        #plt.show(block=False)
        plt.savefig(f"{FILE_PATH}[{adversarial_id}]quanti_search_plot_scatter_total")   
        plt.close("all")
        figure, axes = plt.subplots(1, 1)
        axes.scatter(gower_dists, prob_diffs)
        axes.set_title('gower by probability change')
        axes.set_xlabel("gower distance")
        axes.set_ylabel("probability difference")
        #plt.show(block=False)
        plt.savefig(f"{FILE_PATH}[{adversarial_id}]quanti_search_plot_scatter_change")   
        plt.close("all")

def display_conf_mats(logs):
    for idx, log in enumerate(logs):
        with open(log, 'r') as file:
            log_dict = json.load(file)
        if idx == 0:
            conf_mat_before = np.asarray(log_dict["before"])
            conf_mat_after = np.asarray(log_dict["after"])
        else:
            conf_mat_before = conf_mat_before + np.asarray(log_dict["before"])
            conf_mat_after = conf_mat_after + np.asarray(log_dict["after"])
        
    before_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat_before)
    before_disp.plot()
    #plt.show(block=False)
    plt.savefig(f"{FILE_PATH}conf_mat_before")   
    plt.close("all")

    after_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat_after)
    after_disp.plot()
    #plt.show(block=False)
    plt.savefig(f"{FILE_PATH}conf_mat_after")   
    plt.close("all")

def process_search_logs_runs(file_names):
    search_logs, remainder = select_by_identifier(file_names, "search_logs")
    display_search_runs(search_logs)
    return remainder

def process_search_logs_scatter(file_names):
    search_logs, remainder = select_by_identifier(file_names, "search_logs")
    display_result_scatter(search_logs)
    return remainder

def process_conf_mats(file_names):
    conf_mat_logs, remainder = select_by_identifier(file_names, "confusion_matrices")
    display_conf_mats(conf_mat_logs)
    return remainder

def get_feature_analysis_metrics(feature_distance_tensors):
    adj_feat_ratio_list = []
    mean_adj_list = []
    adj_var_list = []
    adv_id_map = {}
    tensor_idx = 0
    for tensor_path in feature_distance_tensors:
        feature_distance_tensor = t.load(tensor_path)
        tensor_name = tensor_path.split("/")[-1]
        if "]" in tensor_name:
            adversarial_id = tensor_name.split("]")[0][1:]
        else:
            identifier_list = tensor_name.split("_")
            adversarial_id = f"{identifier_list[0]}_{identifier_list[1]}"
        feat_adjusted_tensor = t.logical_not(t.eq(feature_distance_tensor, 0)).long()

        adjusted_feature_counts = t.sum(feat_adjusted_tensor, dim=0)
        adj_feat_ratio_list.append(t.div(adjusted_feature_counts, feature_distance_tensor.shape[0]))
        mean_adj_list.append(t.mean(feature_distance_tensor, dim=0))
        adj_var_list.append(t.var(feature_distance_tensor, dim=0))
        adv_id_map[tensor_idx] = adversarial_id
        tensor_idx += 1

    adj_feat_ratios = t.stack(adj_feat_ratio_list, dim=0)
    mean_adjs = t.stack(mean_adj_list, dim=0)
    adj_vars = t.stack(adj_var_list, dim=0)
    return adj_feat_ratios, mean_adjs, adj_vars, adv_id_map

def process_feature_analysis(file_names):
    feature_distances, remainder = select_by_identifier(file_names, "feat_dists")
    feat_ratios, mean_dists, dist_vars, adv_id_map = get_feature_analysis_metrics(feature_distances)
    result_dict = {}
    for idx in adv_id_map.keys():
        adv_id = adv_id_map[idx]
        sort_idxs = t.argsort(feat_ratios[idx])
        feat_ratios_sorted = feat_ratios[idx][sort_idxs]
        mean_dists_sorted = mean_dists[idx][sort_idxs]
        dist_vars_sorted = dist_vars[idx][sort_idxs]
        result_dict[adv_id] = {}
        for feat_idx in range(len(sort_idxs)):
            feat_ratio = feat_ratios_sorted[feat_idx].item()
            mean_dist = mean_dists_sorted[feat_idx].item()
            dist_var = dist_vars_sorted[feat_idx].item()
            feat_name = FEATURE_NAMES[feat_idx]
            result_dict[adv_id][feat_name] = {"feat_ratio": feat_ratio, "mean_dist": mean_dist, "dist_var": dist_var}
    result_dict_obj = json.dumps(result_dict)
    with open(f"{FILE_PATH}feature_adj_analysis", "w") as outfile:
        outfile.write(result_dict_obj)
    return result_dict, remainder

def stringify(dataframe):
    for key in dataframe.keys():
        if not isinstance(dataframe[key], dict) and not isinstance(dataframe[key], list):
            dataframe[key] = str(dataframe[key])
        elif isinstance(dataframe[key], dict):
            dataframe[key] = stringify(dataframe[key])
        else:
            dataframe[key] = [str(elem) for elem in dataframe[key]]
    return dataframe

def read_result_pkl(result_pkl_path, results=True):
    dictionary = pandas.read_pickle(result_pkl_path)
    print(type(dictionary))
    dictionary = stringify(dictionary)
    if results:
        with open(f"{FILE_PATH}results.json", "w") as outfile: 
            json.dump(dictionary, outfile)
    else:
        with open(f"{FILE_PATH}configs.json", "w") as outfile: 
            json.dump(dictionary, outfile)
    print(dictionary)

def get_results_with_var(logs):
    for idx, log in enumerate(logs):
        with open(log, 'r') as file:
            log_dict = json.load(file)
        if idx == 0:
            gower_dists, total_probs, prob_diffs = get_best_node_metrics(log_dict)
            res_dict = {"gower_dists": gower_dists, "total_prob": total_probs, "prob_diff":prob_diffs}
        else:
            res_dict = get_best_node_metrics(log_dict, res_dict)

    total_probs = torch.Tensor(res_dict["total_prob"])
    is_flipped_map = total_probs > 0.5
    success_rate = torch.sum(is_flipped_map)/total_probs.shape[0]
    gower_dists = torch.Tensor(res_dict["gower_dists"])[is_flipped_map]
    mean_gower_dists = torch.mean(gower_dists)
    var_gower_dists = torch.var(gower_dists)
    
    with open(f"{FILE_PATH}final_results.json", "w") as outfile: 
        json.dump({"success_rate": success_rate.item(), "mean_gower_dists": mean_gower_dists.item(), "var_gower_dists": var_gower_dists.item()}, outfile)
        
def process_results_with_var(file_names):
    search_logs, remainder = select_by_identifier(file_names, "search_logs")
    get_results_with_var(search_logs)
    return remainder

def get_search_lengths(search_logs, parent_dir):
    active_searches = [0]*20
    for search_log in search_logs:
        with open(f"{parent_dir}{search_log}", 'r') as file:
            log_dict = json.load(file)
        for search_key in log_dict.keys():
            search_length = len(log_dict[search_key]["step"])
            for idx in range(search_length):
                if idx >= len(active_searches):
                    active_searches.append(1)
                else:
                    active_searches[idx] += 1
    total_searches = active_searches[0]
    for idx in range(len(active_searches)):
        active_searches[idx] = active_searches[idx]/total_searches
    return active_searches

def process_search_lenght(file_dirs, experiment_name=""):
    fig, ax = plt.subplots(1, 1)
    for file_dir in file_dirs:
        adv_name = file_dir.split("/")[-2].split("_")[2]
        result_files = os.listdir(file_dir)
        search_logs, remainder = select_by_identifier(result_files, "search_logs")
        search_lengths = get_search_lengths(search_logs, file_dir)
        ax.plot(range(len(search_lengths)), search_lengths, label=adv_name)
        ax.set_xticks(range(21))
        ax.set_xlabel("search step")
        ax.set_ylabel("running searches ratio")

    ax.legend()
    #plt.show(block=True)
    fig.savefig(f"{FILE_PATH}search_behavior_{experiment_name}")
    plt.close("all")

def main():
    result_files = os.listdir(FILE_PATH)
    result_paths = []
    for file in result_files:
        result_paths.append(f"{FILE_PATH}{file}")
    result_dict, remainder = process_feature_analysis(result_paths)
    process_conf_mats(result_paths)
    process_search_logs_runs(result_paths)
    process_search_logs_scatter(result_paths)
    process_results_with_var(result_paths)
    read_result_pkl(f"{FILE_PATH}results.pkl")
    read_result_pkl(f"{FILE_PATH}configs.pkl", False)
    if FULL:
        process_search_lenght(FILE_DIRS, EXPERIMENT_NAME)

if __name__ == "__main__":
    main()