import os
import shutil
import hashlib
import torch
import pandas as pd
import argparse
import logging
from datasets import load_dataset
from sae_lens import SAE
from transformer_lens import HookedTransformer
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
from transformer_lens.utils import tokenize_and_concatenate
import pyarrow as pa
import pyarrow.parquet as pq
from metrics import get_metric_function

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
        "--num_prompts", 
        type=int, 
        default=None,
        help="Number of prompts to process (default: None, meaning all of the dataset)."
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="NeelNanda/pile-10k", 
        help="HuggingFace dataset path to load."
    )
    parser.add_argument(
        "--save_attributions",
        action="store_true",
        help="If set, creates a dataset of zero-ablation attributions optimized for future global statistical analysis."
    )
    parser.add_argument(
        "--attributions_dir",
        type=str,
        default="sae_attributions",
        help="Output directory path for the attributions partitioned Parquet dataset."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="Number of prompts to process before flushing attributions to disk."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (e.g. 'train', 'validation', 'test')."
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="Name of the text column in the dataset. If not set, auto-detects from common names (text, prompt, sentence, content)."
    )
    parser.add_argument(
        "--no_concat",
        action="store_true",
        help="Tokenize each prompt individually instead of concatenating into fixed-length windows. Use for short-prompt datasets like mib-bench."
    )
    return parser.parse_args()


def _detect_text_column(dataset, logger):
    """Auto-detect the text column from a dataset's column names."""
    columns = dataset.column_names
    # Ordered by priority
    candidates = ["text", "prompt", "sentence", "content", "input", "question"]
    for c in candidates:
        if c in columns:
            logger.info(f"Auto-detected text column: '{c}'")
            return c
    raise ValueError(
        f"Could not auto-detect a text column from dataset columns: {columns}. "
        f"Please specify --text_column explicitly."
    )


def main():
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("sae_circuit_mining")

    logger.info("Initializing SAE Circuit Mining event generator...")
    logger.info(f"Parameters: release={args.release}, device={args.device}, dataset={args.dataset}")

    # --- Load dataset first (before model/SAE) ---
    logger.info(f"Loading HuggingFace dataset: {args.dataset} (split={args.split})...")
    dataset = load_dataset(path=args.dataset, split=args.split, streaming=False)

    # Resolve the text column
    text_column = args.text_column if args.text_column else _detect_text_column(dataset, logger)
    logger.info(f"Using text column: '{text_column}'")

    logger.info(f"Retrieving metric function for dataset: {args.dataset}")
    metric_fn = get_metric_function(args.dataset)

    # --- Load models and configurations ---
    logger.info(f"Retrieving pre-trained SAE directory map for release: {args.release}...")
    release_saes = get_pretrained_saes_directory()[args.release].saes_map
    
    logger.info("Loading transformer model 'gpt2-small'...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=args.device)

    sorted_sae_items = sorted(
        release_saes.items(), 
        key=lambda x: int(x[0].split('.')[1]) if '.' in x[0] else 0
    )

    logger.info(f"Loading {len(sorted_sae_items)} SAEs for layers...")
    sae_objects = {sid: SAE.from_pretrained(release=args.release, sae_id=sid, device=args.device)[0] for sid, _ in sorted_sae_items}
    logger.info("All SAEs loaded successfully.")
    
    context_size = getattr(list(sae_objects.values())[0].cfg, "context_size", 1024)
    prepend_bos = getattr(list(sae_objects.values())[0].cfg, "prepend_bos", True)

    # --- Tokenize dataset ---
    if args.no_concat:
        # Tokenize each prompt individually (for short-prompt datasets like mib-bench)
        logger.info("Tokenizing dataset in individual-prompt mode (--no_concat)...")
        tokenizer = model.tokenizer
        token_dataset = []
        for row in dataset:
            prompt_text = row[text_column]
            encoding = tokenizer(
                prompt_text,
                return_tensors="pt",
                max_length=context_size,
                truncation=True,
                padding=False,
                add_special_tokens=prepend_bos,
            )
            new_row = row.copy()
            new_row["tokens"] = encoding["input_ids"].squeeze(0)
            token_dataset.append(new_row)
        logger.info(f"Tokenized {len(token_dataset)} individual prompts.")
    else:
        logger.info("Tokenizing and concatenating dataset...")
        token_dataset = tokenize_and_concatenate(
            dataset, model.tokenizer, streaming=True,
            max_length=context_size, column_name=text_column,
            add_bos_token=prepend_bos
        )

    # Set up attributions output when enabled
    if args.save_attributions:
        # Clear old output directory if it exists
        if os.path.exists(args.attributions_dir):
            logger.info(f"Output directory '{args.attributions_dir}' already exists. Clearing old files...")
            shutil.rmtree(args.attributions_dir)
        os.makedirs(args.attributions_dir, exist_ok=True)

        # PyArrow Schema — compact: no string columns, float16 activations
        raw_schema = pa.schema([
            ('counterfactual', pa.string()),
            ('prompt_idx', pa.int32()),
            ('token_idx', pa.int16()),
            ('token_id', pa.int32()),
            ('feature_idx', pa.int32()),
            ('attribution', pa.float16()),
            ('layer', pa.int16()),
        ])

        # Batch accumulator dictionary
        batch_data = {
            'counterfactual': [],
            'prompt_idx': [],
            'token_idx': [],
            'token_id': [],
            'feature_idx': [],
            'attribution': [],
            'layer': []
        }

        def flush_batch():
            if len(batch_data['prompt_idx']) == 0:
                return
            logger.info(f"Flushing {len(batch_data['prompt_idx'])} attribution values to partitioned dataset...")
            table = pa.Table.from_pydict(batch_data, schema=raw_schema)
            pq.write_to_dataset(
                table, 
                root_path=args.attributions_dir, 
                partition_cols=['counterfactual', 'layer'],
                compression='zstd'
            )
            # Clear data
            for k in batch_data:
                batch_data[k].clear()

        total_elements_written = 0

    logger.info("Processing dataset...")
    for p_idx, item in enumerate(token_dataset):
        if args.num_prompts is not None and p_idx >= args.num_prompts:
            logger.info(f"Reached prompt limit ({args.num_prompts}). Stopping.")
            break
            
        token_ids = item["tokens"]
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids)
        tokens = token_ids.unsqueeze(0).to(args.device)
        
        prompt_str = model.to_string(tokens[0])
        trace_id = hashlib.md5(prompt_str.encode("utf-8")).hexdigest()[:8]
        
        logger.info(f"Processing prompt {p_idx+1} | Length: {len(token_ids)} | Trace ID: {trace_id}")
        
        acts_dict = {}
        def make_fwd_hook(hook_name):
            def fwd_hook(tensor, hook):
                tensor.retain_grad()
                acts_dict[hook_name] = tensor
                return tensor
            return fwd_hook
            
        for sid, hook_point in sorted_sae_items:
            model.add_hook(hook_point, make_fwd_hook(hook_point), dir='fwd')
            
        logits = model(tokens)
        
        # Find all counterfactuals in the item
        counterfactuals = {}
        if 'choices' in item and 'answerKey' in item:
            counterfactuals['clean'] = item
            
        for key, value in item.items():
            if isinstance(value, dict) and 'choices' in value and 'answerKey' in value:
                counterfactuals[key] = value

        if not counterfactuals:
            # Fallback for standard datasets without counterfactuals
            counterfactuals['default'] = item

        for cf_name, cf_item in counterfactuals.items():
            model.zero_grad()
            for hook_point in acts_dict:
                if acts_dict[hook_point].grad is not None:
                    acts_dict[hook_point].grad.zero_()
            
            metric_kwargs = {
                'logits': logits,
                'tokens': tokens,
                'item': cf_item,
                'model': model
            }
            
            metric_tensor, direction = metric_fn(**metric_kwargs)
            metric_tensor.backward(retain_graph=True)
            
            for sid, hook_point in sorted_sae_items:
                layer_idx = int(sid.split('.')[1])
                sae = sae_objects[sid]
                
                acts = acts_dict[hook_point] # shape: [1, seq_len, d_model]
                grad_acts = acts.grad[0]     # shape: [seq_len, d_model]
                acts_0 = acts[0]             # shape: [seq_len, d_model]
                
                # encode to get feature activations
                feature_acts = sae.encode(acts_0.unsqueeze(0))[0] # shape: [seq_len, d_sae]
                
                # W_dec transpose handling
                if sae.W_dec.shape[0] == sae.cfg.d_sae:
                    W_dec_T = sae.W_dec.T
                else:
                    W_dec_T = sae.W_dec
                    
                # grad_f = grad_x @ W_dec^T
                grad_f = torch.matmul(grad_acts, W_dec_T)
                
                # zero-ablation attribution = direction * grad_f * feature_acts
                # Helpful features will have positive attribution scores.
                attributions = direction * grad_f * feature_acts
                
                if args.save_attributions:
                    # We filter by feature_acts > 0, so attributions can be positive or negative
                    raw_mask = feature_acts > 0.0
                    raw_passing = raw_mask.nonzero(as_tuple=False)
                    
                    if raw_passing.numel() > 0:
                        token_indices = raw_passing[:, 0]
                        feature_indices = raw_passing[:, 1]
                        vals = attributions[token_indices, feature_indices]
                        
                        # Move to CPU / NumPy
                        token_indices_cpu = token_indices.cpu().numpy()
                        feature_indices_cpu = feature_indices.cpu().numpy()
                        vals_cpu = vals.cpu().detach().to(torch.float16).numpy()  # float16 for size reduction
                        
                        num_events = len(token_indices_cpu)
                        total_elements_written += num_events
                        
                        batch_data['counterfactual'].extend([cf_name] * num_events)
                        batch_data['prompt_idx'].extend([p_idx] * num_events)
                        batch_data['token_idx'].extend(token_indices_cpu.tolist())
                        batch_data['token_id'].extend([token_ids[t].item() for t in token_indices_cpu])
                        batch_data['feature_idx'].extend(feature_indices_cpu.tolist())
                        batch_data['attribution'].extend(vals_cpu.tolist())
                        batch_data['layer'].extend([layer_idx] * num_events)
                        
        model.reset_hooks()
        # acts_dict is captured by reference, clear it for the next prompt
        acts_dict.clear()

        if args.save_attributions and (p_idx + 1) % args.chunk_size == 0:
            flush_batch()

    # Finalize attributions if enabled
    if args.save_attributions:
        flush_batch()
        logger.info(f"Attribution values saved: {total_elements_written} total")
        logger.info(f"Output directory dataset: {args.attributions_dir}")

    logger.info("Execution complete.")

if __name__ == "__main__":
    main()
