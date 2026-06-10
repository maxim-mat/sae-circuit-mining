# Sparse Autoencoder (SAE) Circuit Mining

This repository provides an automated pipeline for **SAE Circuit Mining**, combining **Sparse Autoencoders (SAEs)** and **Process Mining** to discover and visualize latent computational pathways (circuits) within large language models during inference.

By treating the activation of SAE features across successive layers as process events, we can use formal process mining algorithms (like the Inductive Miner in PM4Py) to discover Petri Nets that capture the flow of information through the model.

---

## 🚀 Key Concepts

- **Case Identity (Process Thread):** Each unique token position in each prompt run is treated as an independent process instance (a "case"). This is identified as `{trace_id}_pos_{token_idx}`.
- **Activity:** An activity represents the activation of a specific SAE feature at a particular layer, formatted as `L{layer_idx}_{feature_idx}` (e.g., `L4_1024` for feature 1024 at layer 4).
- **Fake Timestamp:** Since process mining validators require standard date/time structures, model layers are converted to timestamps (Layer 0 corresponds to `1970-01-01 00:00:00`, Layer 1 to `1970-01-01 00:00:01`, and so on). This preserves order relationship while satisfying tool constraints.
- **Trace ID:** A short MD5 hash of the prompt string used to group events belonging to the same prompt sequence.

---

## 🛠️ Setup & Requirements

Ensure you have a Python environment (>= 3.9) with the necessary dependencies:

```bash
pip install torch transformer_lens sae_lens datasets pandas pyarrow pm4py
```

---

## 📖 Running Examples

### 1. Generating Circuit Event Logs (Default Mode)

By default, the pipeline runs GPT-2 Small with pretrained residual stream SAEs (`gpt2-small-res-jb`) over a dataset (default: `NeelNanda/pile-10k`). It applies relative thresholding to filter feature activations and outputs relational process event logs.

```bash
python src/main.py --num_prompts 5
```

**What this does:**
- Loads the `gpt2-small` transformer model.
- Downloads and loads the suite of SAEs for all residual stream layers.
- Processes the first 5 prompts from the dataset.
- Saves two files:
  - `sae_events.parquet`: Contains the filtered activation events.
  - `sae_cases.parquet`: Contains metadata mapping `trace_id` to the raw prompt text.

---

### 2. Discovering Process Architecture (Petri Net Extraction)

Once you have generated `sae_events.parquet`, run the discovery script to execute Inductive Miner analysis and visualize the learned Petri Net:

```bash
python src/discover.py
```

**What this does:**
1. Loads the parquet event log.
2. Converts layer timestamp indices into a datetime representation.
3. Formulates case IDs based on trace and token positions.
4. Executes the **Inductive Miner** algorithm.
5. Displays a graphical representation of the resulting Petri Net showing how features activate sequentially or concurrently.

---

### 3. Custom Event Generation (Thresholding & Top-K)

You can customize feature selection using absolute activation thresholds and restricting activations to the top $K$ features per token.

```bash
python src/main.py \
  --num_prompts 10 \
  --threshold 0.05 \
  --threshold_type absolute \
  --use_top_k \
  --top_k 3 \
  --events_output custom_events.parquet \
  --cases_output custom_cases.parquet
```

**Key Parameters:**
- `--threshold 0.05` & `--threshold_type absolute`: Logs an event only if the absolute feature activation magnitude is at least `0.05`.
- `--use_top_k` & `--top_k 3`: Restricts logged features to the top 3 highest-activating features for each token at each layer.

---

### 4. Partitioned Raw Activations (For Statistical Analysis)

If you are interested in global statistical analysis rather than sequential process traces, you can save all raw SAE activation values above 0.0 partitioned by layer:

```bash
python src/main.py \
  --num_prompts 10 \
  --save_raw_activations \
  --raw_activations_dir sae_activations \
  --chunk_size 5
```

**What this does:**
- Disables thresholding/top-k logic.
- Saves every positive activation ($>0.0$) in a directory structure partitioned by layer (e.g., `sae_activations/layer=0/`, `sae_activations/layer=1/`, etc.).
- Periodically flushes data to disk every `--chunk_size` prompts to optimize memory usage.

---

## 📊 Output Data Schemas

### `sae_events.parquet` (Process Mining Log)
| Column | Type | Description |
| :--- | :--- | :--- |
| `trace_id` | `string` | Unique identifier (hash) of the prompt run. |
| `timestamp` | `int64` | The layer index (used as a sequential event index). |
| `activity` | `string` | The active feature, formatted as `L{layer}_{feature_idx}`. |
| `token_idx` | `int64` | The position index of the token within the prompt. |
| `token` | `string` | The string representation of the token. |
| `relative_activation_value` | `double` | Feature activation divided by sum of activations for that token & layer. |
| `absolute_activation_value` | `double` | Raw output value of the SAE encoder activation. |

### `sae_cases.parquet` (Metadata)
| Column | Type | Description |
| :--- | :--- | :--- |
| `trace_id` | `string` | Matches the `trace_id` in the events table. |
| `prompt` | `string` | The raw text of the processed prompt. |

### `sae_activations` Directory (Raw Partitioned Dataset)
When `--save_raw_activations` is enabled, files are stored as:
```text
sae_activations/
  ├── layer=0/
  │    └── *.parquet
  ├── layer=1/
  │    └── *.parquet
  └── ...
```
Each Parquet partition contains the following columns:
- `prompt_idx` (`int32`): Index of the prompt in the dataset.
- `trace_id` (`string`): Unique hash of the prompt.
- `token_idx` (`int16`): Token position index.
- `token_id` (`int32`): Integer vocabulary ID of the token.
- `token` (`string`): String representation of the token.
- `feature_idx` (`int32`): SAE feature index.
- `activation` (`float32`): Raw activation value ($> 0.0$).
