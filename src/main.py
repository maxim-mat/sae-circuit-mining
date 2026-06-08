import os
import hashlib
import torch
import pandas as pd
import argparse
from datasets import load_dataset
from sae_lens import SAE
from transformer_lens import HookedTransformer
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
from transformer_lens.utils import tokenize_and_concatenate

def parse_args():
    parser = argparse.ArgumentParser(description="SAE Circuit Mining event generator.")
    parser.add_argument(
        "--release", 
        type=str, 
        default="gpt2-small-res-jb", 
        help="SAE release to load."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu", 
        help="Device to run on."
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.1, 
        help="Activation threshold for logging events (applied to relative or absolute depending on --threshold_type)."
    )
    parser.add_argument(
        "--threshold_type",
        type=str,
        choices=["relative", "absolute"],
        default="relative",
        help="Whether the threshold is applied to 'relative' or 'absolute' activations."
    )
    parser.add_argument(
        "--num_prompts", 
        type=int, 
        default=None, 
        help="Number of prompts to process (default: None, meaning all of the dataset)."
    )
    parser.add_argument(
        "--events_output", 
        type=str, 
        default="sae_events.parquet", 
        help="Output path for the events Parquet file."
    )
    parser.add_argument(
        "--cases_output", 
        type=str, 
        default="sae_cases.parquet", 
        help="Output path for the cases Parquet file."
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="NeelNanda/pile-10k", 
        help="HuggingFace dataset path to load."
    )
    parser.add_argument(
        "--use_top_k", 
        action="store_true", 
        help="Use top-k activations per token."
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=1, 
        help="Number of top activations to keep per token when --use_top_k is set."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Load models and configurations
    release_saes = get_pretrained_saes_directory()[args.release].saes_map
    model = HookedTransformer.from_pretrained("gpt2-small", device=args.device)

    sorted_sae_items = sorted(
        release_saes.items(), 
        key=lambda x: int(x[0].split('.')[1]) if '.' in x[0] else 0
    )

    sae_objects = {sid: SAE.from_pretrained(release=args.release, sae_id=sid, device=args.device)[0] for sid, _ in sorted_sae_items}
    context_size = getattr(list(sae_objects.values())[0].cfg, "context_size", 1024)
    prepend_bos = getattr(list(sae_objects.values())[0].cfg, "prepend_bos", True)

    dataset = load_dataset(path=args.dataset, split="train", streaming=False)
    token_dataset = tokenize_and_concatenate(dataset, model.tokenizer, streaming=True, max_length=context_size, add_bos_token=prepend_bos)

    # Global lists to accumulate data for batch writing
    all_events = []
    all_cases = []

    print("Processing dataset...")
    for i, item in enumerate(token_dataset):
        if args.num_prompts is not None and i >= args.num_prompts:
            break
            
        token_ids = item["tokens"]
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids)
        tokens = token_ids.unsqueeze(0).to(args.device)
        
        prompt_str = model.to_string(tokens[0])
        str_tokens = model.to_str_tokens(tokens[0])
        trace_id = hashlib.md5(prompt_str.encode("utf-8")).hexdigest()[:8]
        
        # Store case metadata exactly once per prompt
        all_cases.append({
            "trace_id": trace_id,
            "prompt": prompt_str
        })
        
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)
            
            for sid, hook_point in sorted_sae_items:
                layer_idx = int(sid.split('.')[1])
                sae = sae_objects[sid]
                
                acts = cache[hook_point] 
                feature_acts = sae.encode(acts)[0] 
                
                sum_vals = torch.clamp(feature_acts, min=0.0).sum(dim=1, keepdim=True)
                sum_vals[sum_vals == 0] = 1.0
                rel_activation_matrix = feature_acts / sum_vals
                
                # Determine which matrix to use for thresholding
                if args.threshold_type == "relative":
                    activation_matrix = rel_activation_matrix
                else:
                    activation_matrix = feature_acts
                
                # We require that the activation is >= threshold and the feature is active (> 0)
                mask = (activation_matrix >= args.threshold) & (feature_acts > 0.0)
                
                # If top-k is also enabled, further restrict the mask
                if args.use_top_k:
                    k = min(args.top_k, activation_matrix.shape[1])
                    _, topk_indices = torch.topk(activation_matrix, k=k, dim=1)
                    
                    topk_mask = torch.zeros_like(activation_matrix, dtype=torch.bool)
                    topk_mask.scatter_(1, topk_indices, True)
                    
                    mask = mask & topk_mask
                
                passing_indices = mask.nonzero(as_tuple=False)
                
                for t, idx in passing_indices:
                    t_val = t.item()
                    idx_val = idx.item()
                    
                    # Lean event row: completely omits the prompt string
                    all_events.append({
                        "trace_id": trace_id,
                        "timestamp": layer_idx,
                        "activity": f"L{layer_idx}_{idx_val}",
                        "token_idx": t_val,
                        "token": str_tokens[t_val],
                        "relative_activation_value": rel_activation_matrix[t_val, idx_val].item(),
                        "absolute_activation_value": feature_acts[t_val, idx_val].item()
                    })

    # Save to compressed Parquet files
    if all_events:
        pd.DataFrame(all_events).to_parquet(args.events_output, compression="snappy", index=False)
    else:
        cols = ["trace_id", "timestamp", "activity", "token_idx", "token", "relative_activation_value", "absolute_activation_value"]
        pd.DataFrame(columns=cols).to_parquet(args.events_output, compression="snappy", index=False)

    if all_cases:
        pd.DataFrame(all_cases).to_parquet(args.cases_output, compression="snappy", index=False)
    else:
        cols = ["trace_id", "prompt"]
        pd.DataFrame(columns=cols).to_parquet(args.cases_output, compression="snappy", index=False)

    print(f"Saved relational logs: {args.events_output} and {args.cases_output}")

if __name__ == "__main__":
    main()
