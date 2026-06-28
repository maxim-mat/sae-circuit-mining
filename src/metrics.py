import torch

def next_token_loss_metric(logits, tokens, **kwargs):
    """
    Standard next-token prediction loss over the sequence.
    Lower is better, so direction is -1.
    """
    shift_logits = logits[0, :-1, :].contiguous()
    shift_labels = tokens[0, 1:].contiguous()
    
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), 
        shift_labels.view(-1), 
        reduction='sum'
    )
    return loss, -1


def logit_diff_metric(correct_logits=None, corrupted_logits=None, **kwargs):
    """
    Logit difference metric for tasks like Indirect Object Identification.
    Assumes `correct_logits` and `corrupted_logits` are provided by the caller.
    Higher is better, so direction is 1.
    """
    if correct_logits is None or corrupted_logits is None:
        raise ValueError("logit_diff_metric requires `correct_logits` and `corrupted_logits` to be passed.")
        
    logit_diff = correct_logits - corrupted_logits
    return logit_diff, 1


# Map dataset names to their corresponding metric functions
DATASET_METRICS = {
    'NeelNanda/pile-10k': next_token_loss_metric,
    # Example for an IOI dataset:
    # 'some/ioi-dataset': logit_diff_metric,
    
    'default': next_token_loss_metric
}

def get_metric_function(dataset_name):
    """
    Returns the appropriate metric function for the given dataset.
    """
    if dataset_name in DATASET_METRICS:
        return DATASET_METRICS[dataset_name]
    return DATASET_METRICS['default']
