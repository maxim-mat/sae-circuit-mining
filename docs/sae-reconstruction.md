# Reference numbers

## Q1 — SAE reconstruction, suite `pythia-marks`

64 sequences x 128 tokens from `NeelNanda/pile-10k`, seed 42, BOS position skipped. Acceptance: variance explained >= 0.80.

| ordinal | kind | block | sae_id | variance explained | mean L0 |
|---|---|---|---|---|---|
| 0 | embed | -1 | `blocks.0.hook_resid_pre` | 0.9621 | 2.9 |
| 1 | attn | 0 | `blocks.0.hook_attn_out` | 0.9182 | 128.7 |
| 2 | mlp | 0 | `blocks.0.hook_mlp_out` | 0.9696 | 5.0 |
| 3 | resid | 0 | `blocks.0.hook_resid_post` | 0.9175 | 58.7 |
| 4 | attn | 1 | `blocks.1.hook_attn_out` | 0.8593 | 126.8 |
| 5 | mlp | 1 | `blocks.1.hook_mlp_out` | 0.8435 | 68.9 |
| 6 | resid | 1 | `blocks.1.hook_resid_post` | 0.8500 | 53.8 |
| 7 | attn | 2 | `blocks.2.hook_attn_out` | 0.9006 | 218.6 |
| 8 | mlp | 2 | `blocks.2.hook_mlp_out` | 0.9810 | 89.9 |
| 9 | resid | 2 | `blocks.2.hook_resid_post` | 0.9520 | 109.0 |
| 10 | attn | 3 | `blocks.3.hook_attn_out` | 0.8900 | 171.7 |
| 11 | mlp | 3 | `blocks.3.hook_mlp_out` | 0.8720 | 164.5 |
| 12 | resid | 3 | `blocks.3.hook_resid_post` | 0.9439 | 70.3 |
| 13 | attn | 4 | `blocks.4.hook_attn_out` | 0.8276 | 132.5 |
| 14 | mlp | 4 | `blocks.4.hook_mlp_out` | 0.8996 | 102.0 |
| 15 | resid | 4 | `blocks.4.hook_resid_post` | 0.8787 | 62.3 |
| 16 | attn | 5 | `blocks.5.hook_attn_out` | 0.8595 | 147.2 |
| 17 | mlp | 5 | `blocks.5.hook_mlp_out` | 0.9486 | 103.1 |
| 18 | resid | 5 | `blocks.5.hook_resid_post` | 0.8834 | 72.7 |

**PASS** — every site is above the floor.

