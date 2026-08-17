# Reference numbers

## Q1 — SAE reconstruction, suite `pythia-marks`

32 sequences x 128 tokens from `NeelNanda/pile-10k`, seed 42, BOS position skipped. Acceptance: variance explained >= 0.80.

| ordinal | kind | block | sae_id | variance explained | mean L0 |
|---|---|---|---|---|---|
| 0 | embed | -1 | `blocks.0.hook_resid_pre` | 0.9638 | 5.0 |
| 1 | attn | 0 | `blocks.0.hook_attn_out` | 0.9047 | 132.4 |
| 2 | mlp | 0 | `blocks.0.hook_mlp_out` | 0.9682 | 130.3 |
| 3 | resid | 0 | `blocks.0.hook_resid_post` | 0.9075 | 64.0 |
| 4 | attn | 1 | `blocks.1.hook_attn_out` | 0.8192 | 127.4 |
| 5 | mlp | 1 | `blocks.1.hook_mlp_out` | 0.8344 | 68.5 |
| 6 | resid | 1 | `blocks.1.hook_resid_post` | 0.8083 | 62.6 |
| 7 | attn | 2 | `blocks.2.hook_attn_out` | 0.8800 | 220.9 |
| 8 | mlp | 2 | `blocks.2.hook_mlp_out` | 0.9755 | 90.8 |
| 9 | resid | 2 | `blocks.2.hook_resid_post` | 0.9457 | 110.7 |
| 10 | attn | 3 | `blocks.3.hook_attn_out` | 0.8601 | 172.0 |
| 11 | mlp | 3 | `blocks.3.hook_mlp_out` | 0.8137 | 136.0 |
| 12 | resid | 3 | `blocks.3.hook_resid_post` | 0.7900 | 66.8 |
| 13 | attn | 4 | `blocks.4.hook_attn_out` | 0.7329 | 136.5 |
| 14 | mlp | 4 | `blocks.4.hook_mlp_out` | 0.8851 | 93.3 |
| 15 | resid | 4 | `blocks.4.hook_resid_post` | 0.8326 | 60.6 |
| 16 | attn | 5 | `blocks.5.hook_attn_out` | 0.8185 | 167.6 |
| 17 | mlp | 5 | `blocks.5.hook_mlp_out` | 0.9191 | 163.8 |
| 18 | resid | 5 | `blocks.5.hook_resid_post` | 0.8601 | 98.4 |

**FAIL** — sites below the floor: blocks.3.hook_resid_post (0.790), blocks.4.hook_attn_out (0.733)

