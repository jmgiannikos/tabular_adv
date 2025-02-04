import torch

def is_not_in(samples, check_against_samples):
    check_against_samples = torch.stack(check_against_samples, dim=0)
    desired_shape = (samples.shape[0], check_against_samples.shape[0], samples.shape[1])
    check_against_samples = torch.unsqueeze(check_against_samples, dim=0).expand(desired_shape)
    samples = torch.unsqueeze(samples, dim=1).expand(desired_shape)
    equality_matrix = torch.all(torch.eq(samples, check_against_samples), dim=2)
    is_not_in_map = torch.logical_not(torch.any(equality_matrix, dim=1))
    return is_not_in_map