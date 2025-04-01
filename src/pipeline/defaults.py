from enum import Enum

DEFAULTS = {
    "target_label": 0,
    "ncalls": 50,
    "imperceptability_weight": 0.5,
    "tolerance": 0,
    "cuda_device": "cpu",
    "safe_div_factor": 0.00000001,
    "victim_training_epochs": 1,
    "performance_log_file": "perf_log",
    "results_path": "./results",
    "check_constraints": False,
    "crossval_folds": 3,
    "dataset_metric_subsampling_steps": 3,
    "device": "cpu",
    "sample_even": True
}

class Log_styles(Enum):
    NOLOG = "No Logging"
    WANDB = "only wandb logging"
    LOCAL = "only local logging"
    BOTH = "wandb and local logging"

AVAILABLE_LOG_STYLES = {
    "nolog": Log_styles.NOLOG,
    "wandb": Log_styles.WANDB,
    "local": Log_styles.LOCAL,
    "both": Log_styles.BOTH
}