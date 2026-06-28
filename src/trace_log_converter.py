import os
import hashlib
import argparse
import logging
import pandas as pd
import pyarrow.dataset as ds
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate

def parse_args():
    parser = argparse.ArgumentParser(description="Convert attributions dataset to trace logs.")
    parser.add_argument("--attributions_dir", type=str, default="sae_attributions")
    parser.add_argument("--events_output", type=str, default="sae_events.parquet")
    parser.add_argument("--cases_output", type=str, default="sae_cases.parquet")
    parser.add_argument("--dataset", type=str, default="NeelNanda/pile-10k")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default=None)
    parser.add_argument("--no_concat", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--threshold_type", type=str, choices=["relative", "absolute"], default="absolute")
    parser.add_argument("--use_top_k", action="store_true")
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="gpt2-small")
    parser.add_argument("--context_size", type=int, default=1024)
    parser.add_argument("--prepend_bos", action="store_true", default=True)
    return parser.parse_args()

def _detect_text_column(dataset, logger):
    columns = dataset.column_names
    candidates = ["text", "prompt", "sentence", "content", "input", "question"]
    for c in candidates:
        if c in columns:
            return c
    raise ValueError("Could not auto-detect text column.")

def discretize_causal_log(parquet_path, mass_threshold=0.90):
    print(f"Loading raw attributions from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Take absolute value to treat negative suppressive mass equally to positive generative mass
    df['abs_attr'] = df['attribution'].abs()
    
    # Sort to ensure highest causal drivers are at the top of each token/layer group
    df = df.sort_values(
        ['prompt_idx', 'token_idx', 'layer', 'abs_attr'], 
        ascending=[True, True, True, False]
    )
    
    print("Calculating cumulative causal mass...")
    # Calculate relative mass contribution per token per layer
    df['total_group_mass'] = df.groupby(['prompt_idx', 'token_idx', 'layer'])['abs_attr'].transform('sum')
    df['cumulative_mass'] = df.groupby(['prompt_idx', 'token_idx', 'layer'])['abs_attr'].cumsum()
    
    df['mass_ratio'] = df['cumulative_mass'] / (df['total_group_mass'] + 1e-9)
    
    print(f"Filtering for top {mass_threshold*100}% causal mass...")
    # Keep only the features necessary to reach the target causal mass
    discrete_log = df[df['mass_ratio'] <= mass_threshold].copy()
    
    # Drop calculation columns to clean the output
    discrete_log = discrete_log.drop(columns=['abs_attr', 'total_group_mass', 'cumulative_mass', 'mass_ratio'])
    
    return discrete_log

# Execute
# discrete_df = discretize_causal_log("sae_attributions", mass_threshold=0.90)

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("trace_log_converter")

    logger.info("Loading model tokenizer...")
    model = HookedTransformer.from_pretrained(args.model_name, device="cpu")
    
    logger.info(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(args.dataset, split=args.split, streaming=False)
    text_column = args.text_column if args.text_column else _detect_text_column(dataset, logger)

    if args.no_concat:
        tokenizer = model.tokenizer
        token_dataset = []
        for row in dataset:
            encoding = tokenizer(
                row[text_column],
                return_tensors="pt",
                max_length=args.context_size,
                truncation=True,
                padding=False,
                add_special_tokens=args.prepend_bos,
            )
            token_dataset.append({"tokens": encoding["input_ids"].squeeze(0)})
    else:
        token_dataset = tokenize_and_concatenate(
            dataset, model.tokenizer, streaming=True,
            max_length=args.context_size, column_name=text_column,
            add_bos_token=args.prepend_bos
        )

    logger.info(f"Loading attributions dataset from {args.attributions_dir}...")
    if not os.path.exists(args.attributions_dir):
        logger.error(f"Directory {args.attributions_dir} does not exist. Run main.py first.")
        return

    dataset_pa = ds.dataset(args.attributions_dir, format="parquet")
    df = dataset_pa.to_table().to_pandas()
    
    if df.empty:
        logger.warning("Attributions dataset is empty. Exiting.")
        return

    df['abs_activation'] = df['activation'].abs()
    
    if args.threshold_type == "relative":
        sum_vals = df.groupby(['prompt_idx', 'token_idx', 'layer'])['abs_activation'].transform('sum')
        sum_vals = sum_vals.replace(0, 1.0) 
        df['relative_activation_value'] = df['abs_activation'] / sum_vals
        df['metric_for_threshold'] = df['relative_activation_value']
    else:
        df['relative_activation_value'] = None 
        df['metric_for_threshold'] = df['abs_activation']
        
    df = df[df['metric_for_threshold'] >= args.threshold]
    
    if args.use_top_k:
        df = df.sort_values(['prompt_idx', 'token_idx', 'layer', 'abs_activation'], ascending=[True, True, True, False])
        df = df.groupby(['prompt_idx', 'token_idx', 'layer']).head(args.top_k)
        
    all_cases = []
    unique_prompts = df['prompt_idx'].unique()
    
    prompt_to_trace = {}
    for p_idx in unique_prompts:
        tokens = token_dataset[p_idx]["tokens"]
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens)
            
        prompt_str = model.to_string(tokens)
        trace_id = hashlib.md5(prompt_str.encode("utf-8")).hexdigest()[:8]
        prompt_to_trace[p_idx] = trace_id
        
        all_cases.append({
            "trace_id": trace_id,
            "prompt": prompt_str
        })
        
    cases_df = pd.DataFrame(all_cases)
    
    df['trace_id'] = df['prompt_idx'].map(prompt_to_trace)
    df['activity'] = "L" + df['layer'].astype(str) + "_" + df['feature_idx'].astype(str)
    df['timestamp'] = df['layer']
    
    logger.info("Mapping tokens to strings...")
    unique_tokens = df[['prompt_idx', 'token_idx', 'token_id']].drop_duplicates()
    unique_tokens['token'] = unique_tokens['token_id'].apply(lambda tid: model.tokenizer.decode([int(tid)]))
    
    df = df.merge(unique_tokens[['prompt_idx', 'token_idx', 'token']], on=['prompt_idx', 'token_idx'], how='left')
    
    if args.threshold_type == "absolute":
        sum_vals = df.groupby(['prompt_idx', 'token_idx', 'layer'])['abs_activation'].transform('sum')
        sum_vals = sum_vals.replace(0, 1.0)
        df['relative_activation_value'] = df['abs_activation'] / sum_vals
        
    events_df = df[['trace_id', 'timestamp', 'activity', 'token_idx', 'token', 'relative_activation_value', 'activation']]
    events_df = events_df.rename(columns={'activation': 'absolute_activation_value'})
    
    logger.info(f"Saving {len(events_df)} events to {args.events_output}")
    events_df.to_parquet(args.events_output, compression="snappy", index=False)
    
    logger.info(f"Saving {len(cases_df)} cases to {args.cases_output}")
    cases_df.to_parquet(args.cases_output, compression="snappy", index=False)
    
if __name__ == "__main__":
    main()
