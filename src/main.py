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
OUTPUT_CSV_PATH = "sae_traces.csv"

if os.path.exists(OUTPUT_CSV_PATH):
    os.remove(OUTPUT_CSV_PATH)

# Load pretrained SAEs configurations and weights
release_saes = get_pretrained_saes_directory()[RELEASE].saes_map
model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

# Fix Structural Bug: Ensure layers are sorted monotonically (Layer 0 -> Layer 11)
sorted_sae_items = sorted(
    release_saes.items(), 
    key=lambda x: int(x[0].split('.')[1]) if '.' in x[0] else 0
)

sae_objects = {}
for sid, _ in sorted_sae_items:
    sae, _, _ = SAE.from_pretrained(release=RELEASE, sae_id=sid, device=DEVICE)
    sae_objects[sid] = sae

first_sae = list(sae_objects.values())[0]
context_size = getattr(first_sae.cfg, "context_size", 1024)
prepend_bos = getattr(first_sae.cfg, "prepend_bos", True)

dataset = load_dataset(path="NeelNanda/pile-10k", split="train", streaming=False)

token_dataset = tokenize_and_concatenate(
    dataset=dataset,  # type: ignore
    tokenizer=model.tokenizer,  # type: ignore
    streaming=True,
    max_length=context_size,
    add_bos_token=prepend_bos,
)

print(f"Starting processing. Traces will be saved to {OUTPUT_CSV_PATH}")

for i, item in enumerate(token_dataset):
    if NUM_PROMPTS_TO_PROCESS is not None and i >= NUM_PROMPTS_TO_PROCESS:
        print(f"Processed the set limit of {NUM_PROMPTS_TO_PROCESS} prompts. Stopping.")
        break
        
    print(f"Processing prompt {i+1}...")
    
    token_ids = item["tokens"]
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids)
    tokens = token_ids.unsqueeze(0).to(DEVICE)
    
    prompt_str = model.to_string(tokens[0])
    str_tokens = model.to_str_tokens(tokens[0])
    trace_id = hashlib.md5(prompt_str.encode("utf-8")).hexdigest()[:8]
    
    prompt_rows = []
    
    # Fix Memory Leak: Enforce torch.no_grad() globally for the inference cycle
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
        
        for sid, hook_point in sorted_sae_items:
            layer_idx = int(sid.split('.')[1])
            sae = sae_objects[sid]
            
            acts = cache[hook_point] 
            feature_acts = sae.encode(acts)[0] # Shape: [seq_len, num_features]
            
            # Vectorized Math Optimization: Compute sum of positive activations across all positions at once
            # Clamp negatives to 0, then sum along the feature dimension (dim=1)
            sum_vals = torch.clamp(feature_acts, min=0.0).sum(dim=1, keepdim=True) # [seq_len, 1]
            
            # Avoid division by zero
            sum_vals[sum_vals == 0] = 1.0
            
            # Compute relative activation matrix entirely on GPU
            rel_activation_matrix = feature_acts / sum_vals # [seq_len, num_features]
            
            # Filter entire matrix via thresholding at once
            passing_indices = (rel_activation_matrix >= LOGGING_THRESHOLD).nonzero(as_tuple=False)
            
            # passing_indices contains pairs of [token_index, feature_index]
            for t, idx in passing_indices:
                t_val = t.item()
                idx_val = idx.item()
                
                prompt_rows.append({
                    "trace_id": trace_id,
                    "timestamp": layer_idx,
                    "activity": f"L{layer_idx}_{idx_val}",
                    "token_idx": t_val,
                    "token": str_tokens[t_val],
                    "relative_activation_value": rel_activation_matrix[t_val, idx_val].item(),
                    "prompt": prompt_str
                })
                        
    if len(prompt_rows) > 0:
        df_prompt = pd.DataFrame(prompt_rows)
        write_header = not os.path.exists(OUTPUT_CSV_PATH)
        df_prompt.to_csv(OUTPUT_CSV_PATH, mode="a", index=False, header=write_header)

print("Finished processing dataset.")