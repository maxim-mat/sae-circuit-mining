import os
import shutil
import hashlib
import torch
import pandas as pd
import argparse
from datasets import load_dataset
from sae_lens import SAE
from transformer_lens import HookedTransformer
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
from transformer_lens.utils import tokenize_and_concatenate
import pyarrow as pa
import pyarrow.parquet as pq

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
    parser.add_argument(
        "--save_raw_activations",
        action="store_true",
        help="If set, creates a dataset of raw SAE activation values (> 0.0) optimized for future global statistical analysis instead of process-mining event traces."
    )
    parser.add_argument(
        "--raw_activations_dir",
        type=str,
        default="sae_activations",
        help="Output directory path for the raw activations partitioned Parquet dataset."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="Number of prompts to process before flushing raw activations to disk."
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
    if args.save_raw_activations:
        # Clear old output directory if it exists
        if os.path.exists(args.raw_activations_dir):
            print(f"Output directory '{args.raw_activations_dir}' already exists. Clearing old files...")
            shutil.rmtree(args.raw_activations_dir)
        os.makedirs(args.raw_activations_dir, exist_ok=True)

        # PyArrow Schema for optimized storage & future statistical queries
        raw_schema = pa.schema([
            ('prompt_idx', pa.int32()),
            ('trace_id', pa.string()),
            ('token_idx', pa.int16()),
            ('token_id', pa.int32()),
            ('token', pa.string()),
            ('feature_idx', pa.int32()),
            ('activation', pa.float32()),
            ('layer', pa.int16()),
        ])

        # Batch accumulator dictionary
        batch_data = {
            'prompt_idx': [],
            'trace_id': [],
            'token_idx': [],
            'token_id': [],
            'token': [],
            'feature_idx': [],
            'activation': [],
            'layer': []
        }

        def flush_batch():
            if len(batch_data['prompt_idx']) == 0:
                return
            print(f"Flushing {len(batch_data['prompt_idx'])} activation values to partitioned dataset...")
            table = pa.Table.from_pydict(batch_data, schema=raw_schema)
            pq.write_to_dataset(
                table, 
                root_path=args.raw_activations_dir, 
                partition_cols=['layer'], 
                use_dictionary=['token', 'trace_id']
            )
            # Clear data
            for k in batch_data:
                batch_data[k].clear()

        total_elements_written = 0
    else:
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
        
        if args.save_raw_activations:
            print(f"Prompt {i+1} | Length: {len(token_ids)} | Trace ID: {trace_id}")
        else:
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
                
                if args.save_raw_activations:
                    # Raw activations mode: Extract ALL activations > 0.0 without thresholding/top_k
                    mask = feature_acts > 0.0
                    passing_indices = mask.nonzero(as_tuple=False)
                    
                    if passing_indices.numel() > 0:
                        token_indices = passing_indices[:, 0]
                        feature_indices = passing_indices[:, 1]
                        vals = feature_acts[token_indices, feature_indices]
                        
                        # Move to CPU / NumPy
                        token_indices_cpu = token_indices.cpu().numpy()
                        feature_indices_cpu = feature_indices.cpu().numpy()
                        vals_cpu = vals.cpu().numpy()
                        
                        num_events = len(token_indices_cpu)
                        total_elements_written += num_events
                        
                        batch_data['prompt_idx'].extend([i] * num_events)
                        batch_data['trace_id'].extend([trace_id] * num_events)
                        batch_data['token_idx'].extend(token_indices_cpu.tolist())
                        batch_data['token_id'].extend([token_ids[t].item() for t in token_indices_cpu])
                        batch_data['token'].extend([str_tokens[t] for t in token_indices_cpu])
                        batch_data['feature_idx'].extend(feature_indices_cpu.tolist())
                        batch_data['activation'].extend(vals_cpu.tolist())
                        batch_data['layer'].extend([layer_idx] * num_events)
                else:
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

        if args.save_raw_activations and (i + 1) % args.chunk_size == 0:
            flush_batch()

    if args.save_raw_activations:
        # Flush any remaining items in buffer
        flush_batch()
        print("--------------------------------------------------")
        print(f"Execution complete. Total raw activation values saved: {total_elements_written}")
        print(f"Output directory dataset: {args.raw_activations_dir}")
    else:
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
