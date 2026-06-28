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


def mib_bench_logit_diff_metric(logits, item, model, **kwargs):
    """
    Logit difference metric for mib-bench datasets.
    Extracts correct and corrupted tokens from `item['choices']` and `item['answerKey']`.
    """
    if 'choices' not in item or 'answerKey' not in item:
        raise ValueError("mib_bench_logit_diff_metric requires `choices` and `answerKey` in the item.")
        
    choices = item['choices']
    answer_key = item['answerKey']
    
    correct_str = choices[answer_key]
    corrupted_str = choices[1 - answer_key]
    
    # Tokenize (assuming standard leading space for names in GPT-2)
    correct_token = model.to_single_token(" " + correct_str)
    corrupted_token = model.to_single_token(" " + corrupted_str)
    
    # Usually logit diff is measured at the last position of the prompt
    pos = -1
    
    correct_logits = logits[0, pos, correct_token]
    corrupted_logits = logits[0, pos, corrupted_token]
    
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
    if dataset_name.startswith('mib-bench/'):
        return mib_bench_logit_diff_metric
        
    if dataset_name in DATASET_METRICS:
        return DATASET_METRICS[dataset_name]
    return DATASET_METRICS['default']
