import random
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

PROMPT = "The capital of France is Paris, which is known for"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# All candidate SAE sites across gpt2-small (12 layers, 3 site types).
# Each entry: (release, sae_id, hook_point)
#   hook_z      = individual attention head outputs (post-softmax, post-value projection)
#   hook_attn_out = after concat + output projection of all heads
#   hook_mlp_out  = after the MLP/FFN block (post-GELU activation, post-projection)
ALL_SAE_SITES = []
for layer in range(12):
    ALL_SAE_SITES.append((
        "gpt2-small-hook-z-kk",
        f"blocks.{layer}.hook_z",
        f"blocks.{layer}.attn.hook_z",
    ))
    ALL_SAE_SITES.append((
        "gpt2-small-attn-out-v5-32k",
        f"blocks.{layer}.hook_attn_out",
        f"blocks.{layer}.hook_attn_out",
    ))
    ALL_SAE_SITES.append((
        "gpt2-small-mlp-out-v5-32k",
        f"blocks.{layer}.hook_mlp_out",
        f"blocks.{layer}.hook_mlp_out",
    ))

random.seed(42)
SELECTED_SITES = random.sample(ALL_SAE_SITES, 10)

model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)

saes = {}
for release, sae_id, hook_point in SELECTED_SITES:
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=DEVICE)
    saes[hook_point] = sae

tokens = model.to_tokens(PROMPT)
_, cache = model.run_with_cache(tokens, names_filter=list(saes.keys()))

for hook_point, sae in saes.items():
    acts = cache[hook_point]
    # hook_z activations are [batch, pos, n_heads, d_head] — flatten heads into d_model
    if acts.dim() == 4:
        batch, pos, n_heads, d_head = acts.shape
        acts = acts.reshape(batch, pos, n_heads * d_head)
    feature_acts = sae.encode(acts)
    active_mask = feature_acts > 0
    active_ids = [active_mask[0, t].nonzero(as_tuple=True)[0].tolist() for t in range(active_mask.shape[1])]
    print(f"\n=== {hook_point} ===")
    for t, ids in enumerate(active_ids):
        print(f"  pos {t}: {ids}")
