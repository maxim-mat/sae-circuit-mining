import os
import shutil
import hashlib
# pyrefly: ignore [missing-import]
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
    parser = argparse.ArgumentParser(description="Save raw SAE activation values to an optimized partitioned Parquet dataset.")
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
        default=0.0, 
        help="Activation threshold (features must be > threshold and > 0.0)."
    )
    parser.add_argument(
        "--num_prompts", 
        type=int, 
        default=None, 
        help="Number of prompts to process (default: None, meaning all of the dataset)."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="sae_activations", 
        help="Output directory path for the partitioned Parquet dataset."
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="NeelNanda/pile-10k", 
        help="HuggingFace dataset path to load."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="Number of prompts to process before flushing to disk."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Clear old output directory if it exists to avoid mixing data
    if os.path.exists(args.output_dir):
        print(f"Output directory '{args.output_dir}' already exists. Clearing old files...")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models and configurations
    print(f"Loading SAE release '{args.release}' directory map...")
    release_saes = get_pretrained_saes_directory()[args.release].saes_map
    
    print("Loading HookedTransformer 'gpt2-small'...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=args.device)

    sorted_sae_items = sorted(
        release_saes.items(), 
        key=lambda x: int(x[0].split('.')[1]) if '.' in x[0] else 0
    )

    print(f"Loading {len(sorted_sae_items)} SAE objects...")
    sae_objects = {}
    for sid, _ in sorted_sae_items:
        print(f"  Loading SAE: {sid}")
        sae_objects[sid] = SAE.from_pretrained(release=args.release, sae_id=sid, device=args.device)[0]
    
    context_size = getattr(list(sae_objects.values())[0].cfg, "context_size", 1024)
    prepend_bos = getattr(list(sae_objects.values())[0].cfg, "prepend_bos", True)

    print(f"Loading HuggingFace dataset: {args.dataset}")
    dataset = load_dataset(path=args.dataset, split="train", streaming=False)
    token_dataset = tokenize_and_concatenate(dataset, model.tokenizer, streaming=True, max_length=context_size, add_bos_token=prepend_bos)

    # PyArrow Schema for optimized storage & future statistical queries
    schema = pa.schema([
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
        table = pa.Table.from_pydict(batch_data, schema=schema)
        pq.write_to_dataset(
            table, 
            root_path=args.output_dir, 
            partition_cols=['layer'], 
            use_dictionary=['token', 'trace_id']
        )
        # Clear data
        for k in batch_data:
            batch_data[k].clear()

    total_elements_written = 0
    
    print("Processing dataset prompts...")
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
        
        print(f"Prompt {i+1} | Length: {len(token_ids)} | Trace ID: {trace_id}")
        
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)
            
            for sid, hook_point in sorted_sae_items:
                layer_idx = int(sid.split('.')[1])
                sae = sae_objects[sid]
                
                acts = cache[hook_point] 
                feature_acts = sae.encode(acts)[0] # Shape: [seq_len, num_features]
                
                # Active filter (ReLU activations are positive)
                mask = (feature_acts >= args.threshold) & (feature_acts > 0.0)
                
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

        if (i + 1) % args.chunk_size == 0:
            flush_batch()

    # Flush any remaining items in buffer
    flush_batch()

    print("--------------------------------------------------")
    print(f"Execution complete. Total raw activation values saved: {total_elements_written}")
    print(f"Output directory dataset: {args.output_dir}")

if __name__ == "__main__":
    main()
