import os
import hashlib
import torch
import pandas as pd
from datasets import load_dataset
from sae_lens import SAE
from transformer_lens import HookedTransformer
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
from transformer_lens.utils import tokenize_and_concatenate

# --- Configuration ---
RELEASE = "gpt2-small-res-jb"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOGGING_THRESHOLD = 0.1
NUM_PROMPTS_TO_PROCESS = 3  
EVENTS_OUTPUT_PATH = "sae_events.parquet"
CASES_OUTPUT_PATH = "sae_cases.parquet"

# Load models and configurations
release_saes = get_pretrained_saes_directory()[RELEASE].saes_map
model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

sorted_sae_items = sorted(
    release_saes.items(), 
    key=lambda x: int(x[0].split('.')[1]) if '.' in x[0] else 0
)

sae_objects = {sid: SAE.from_pretrained(release=RELEASE, sae_id=sid, device=DEVICE)[0] for sid, _ in sorted_sae_items}
context_size = getattr(list(sae_objects.values())[0].cfg, "context_size", 1024)
prepend_bos = getattr(list(sae_objects.values())[0].cfg, "prepend_bos", True)

dataset = load_dataset(path="NeelNanda/pile-10k", split="train", streaming=False)
token_dataset = tokenize_and_concatenate(dataset, model.tokenizer, streaming=True, max_length=context_size, add_bos_token=prepend_bos)

# Global lists to accumulate data for batch writing
all_events = []
all_cases = []

print("Processing dataset...")
for i, item in enumerate(token_dataset):
    if NUM_PROMPTS_TO_PROCESS is not None and i >= NUM_PROMPTS_TO_PROCESS:
        break
        
    token_ids = item["tokens"]
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids)
    tokens = token_ids.unsqueeze(0).to(DEVICE)
    
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
            
            passing_indices = (rel_activation_matrix >= LOGGING_THRESHOLD).nonzero(as_tuple=False)
            
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
                    "relative_activation_value": rel_activation_matrix[t_val, idx_val].item()
                })

# Save to compressed Parquet files
pd.DataFrame(all_events).to_parquet(EVENTS_OUTPUT_PATH, compression="snappy", index=False)
pd.DataFrame(all_cases).to_parquet(CASES_OUTPUT_PATH, compression="snappy", index=False)
print(f"Saved relational logs: {EVENTS_OUTPUT_PATH} and {CASES_OUTPUT_PATH}")
