# Reference numbers

## Q1 — SAE reconstruction, suite `pythia-marks`

64 sequences x 128 tokens from `NeelNanda/pile-10k`, seed 42, BOS position skipped. Acceptance: variance explained >= 0.80.

| ordinal | kind | block | sae_id | variance explained | mean L0 |
|---|---|---|---|---|---|
| 0 | embed | -1 | `blocks.0.hook_resid_pre` | 0.9619 | 5.1 |
| 1 | attn | 0 | `blocks.0.hook_attn_out` | 0.9198 | 131.3 |
| 2 | mlp | 0 | `blocks.0.hook_mlp_out` | 0.9658 | 128.6 |
| 3 | resid | 0 | `blocks.0.hook_resid_post` | 0.9107 | 63.5 |
| 4 | attn | 1 | `blocks.1.hook_attn_out` | 0.8596 | 127.5 |
| 5 | mlp | 1 | `blocks.1.hook_mlp_out` | 0.8428 | 68.8 |
| 6 | resid | 1 | `blocks.1.hook_resid_post` | 0.8213 | 62.6 |
| 7 | attn | 2 | `blocks.2.hook_attn_out` | 0.9021 | 221.2 |
| 8 | mlp | 2 | `blocks.2.hook_mlp_out` | 0.9750 | 90.1 |
| 9 | resid | 2 | `blocks.2.hook_resid_post` | 0.9462 | 110.0 |
| 10 | attn | 3 | `blocks.3.hook_attn_out` | 0.8887 | 169.7 |
| 11 | mlp | 3 | `blocks.3.hook_mlp_out` | 0.8242 | 135.1 |
| 12 | resid | 3 | `blocks.3.hook_resid_post` | 0.7979 | 65.5 |
| 13 | attn | 4 | `blocks.4.hook_attn_out` | 0.8265 | 132.5 |
| 14 | mlp | 4 | `blocks.4.hook_mlp_out` | 0.8871 | 91.5 |
| 15 | resid | 4 | `blocks.4.hook_resid_post` | 0.8478 | 59.1 |
| 16 | attn | 5 | `blocks.5.hook_attn_out` | 0.8671 | 164.2 |
| 17 | mlp | 5 | `blocks.5.hook_mlp_out` | 0.9272 | 161.9 |
| 18 | resid | 5 | `blocks.5.hook_resid_post` | 0.8876 | 94.5 |

**FAIL** — sites below the floor: blocks.3.hook_resid_post (0.798)

