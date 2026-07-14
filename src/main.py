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
        help="If set, saves counterfactual (activation-patching) attributions per SAE feature and per SAE error node to a partitioned Parquet dataset. Requires a dataset with counterfactual columns (e.g. MIB-bench)."
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
    parser.add_argument(
        "--attribution_method",
        type=str,
        default="attrib",
        choices=["attrib", "ig"],
        help="Node attribution method: 'attrib' = first-order attribution patching (gradient at the clean point); 'ig' = integrated gradients along the clean->counterfactual path (Marks et al. 2024). ig is slower but more accurate for early layers / large activation deltas."
    )
    parser.add_argument(
        "--ig_steps",
        type=int,
        default=10,
        help="Number of interpolation steps for --attribution_method ig. Ignored for 'attrib'. ig_steps=1 reduces exactly to attribution patching."
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


def integrated_gradient_grad_x(model, clean_tokens, hook_point, acts_clean, acts_cf,
                               metric_fn, item, steps):
    """Mean residual-stream gradient of the metric along the clean->counterfactual
    interpolation path at one hook point.

    Integrated-gradients node attribution, following the reference implementation
    _pe_ig in Marks et al. 2024, "Sparse Feature Circuits" (arXiv:2403.19647):
    only this hook's activation is interpolated, so all other layers see the clean
    input. The interpolated activation is (1-a)*acts_clean + a*acts_cf for
    a = step/steps, step in [0, steps); this is exactly decode(interp features) +
    interp residual because decode is affine and coefficients sum to 1, so the
    residual-stream gradient projected by W_dec.T / read directly recovers the
    feature / error gradients. The `steps` points are batched into one pass.

    Returns [seq, d_model]; steps=1 reduces to attribution patching (gradient at
    the clean point).
    """
    alphas = torch.arange(steps, device=clean_tokens.device, dtype=acts_clean.dtype) / steps
    x_interp = acts_clean.unsqueeze(0) + alphas.view(-1, 1, 1) * (acts_cf - acts_clean).unsqueeze(0)
    x_interp = x_interp.detach().requires_grad_()  # [steps, seq, d_model] leaf

    def ig_hook(tensor, hook):
        return x_interp

    logits = model.run_with_hooks(
        clean_tokens.repeat(steps, 1), fwd_hooks=[(hook_point, ig_hook)]
    )
    # Sum per-step metrics: d(sum)/dx_interp[i] isolates the gradient at alpha_i.
    total_metric = 0
    for i in range(steps):
        m_i, _ = metric_fn(logits=logits[i:i+1], tokens=clean_tokens, item=item, model=model)
        total_metric = total_metric + m_i
    model.zero_grad(set_to_none=True)
    total_metric.backward()
    return x_interp.grad.mean(dim=0)  # [seq, d_model]


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
        token_dataset = []
        for row in dataset:
            new_row = row.copy()
            # model.to_tokens honors prepend_bos; the raw tokenizer's
            # add_special_tokens is a no-op for GPT-2, which never inserts a BOS,
            # leaving position 0 off-distribution for the SAEs.
            new_row["tokens"] = model.to_tokens(
                row[text_column], prepend_bos=prepend_bos
            ).squeeze(0)
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
            ('attribution', pa.float32()),
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
            break

        # Counterfactual patching is this script's only output; without patch
        # pairs (or without --save_attributions) the per-prompt forward/backward
        # produces nothing, so skip before doing any work.
        counterfactuals = {k: v for k, v in item.items() if isinstance(v, dict) and 'prompt' in v}
        if not args.save_attributions or not counterfactuals:
            continue

        clean_token_ids = item["tokens"]
        if not isinstance(clean_token_ids, torch.Tensor):
            clean_token_ids = torch.tensor(clean_token_ids)
        clean_tokens = clean_token_ids.unsqueeze(0).to(args.device)
        
        model.zero_grad()
        acts_dict_clean = {}
        
        def make_fwd_hook(hook_name, target_dict):
            def fwd_hook(tensor, hook):
                if target_dict is acts_dict_clean:
                    tensor.retain_grad()
                target_dict[hook_name] = tensor
                return tensor
            return fwd_hook
            
        for sid, hook_point in sorted_sae_items:
            model.add_hook(hook_point, make_fwd_hook(hook_point, acts_dict_clean), dir='fwd')
            
        clean_logits = model(clean_tokens)
        
        metric_kwargs = {
            'logits': clean_logits,
            'tokens': clean_tokens,
            'item': item, 
            'model': model
        }
        metric_tensor, direction = metric_fn(**metric_kwargs)
        # 'ig' recomputes gradients along an interpolation path per layer, so it
        # does not use the clean-point gradient.
        if args.attribution_method == "attrib":
            metric_tensor.backward()

        model.reset_hooks()

        for cf_name, cf_item in counterfactuals.items():
            
            cf_text = cf_item['prompt']
            # to_tokens honors prepend_bos (see the clean-side note above).
            cf_tokens = model.to_tokens(cf_text, prepend_bos=prepend_bos).to(args.device)

            # Counterfactual patching requires clean/CF positions to correspond.
            # MIB pairs are length-matched by design; if BPE makes them differ,
            # positions misalign and the metric-bearing final position would be
            # silently dropped, so skip this pair rather than emit corrupt data.
            if cf_tokens.shape[1] != clean_tokens.shape[1]:
                logger.warning(
                    f"Prompt {p_idx} counterfactual '{cf_name}': token-length mismatch "
                    f"(clean={clean_tokens.shape[1]}, cf={cf_tokens.shape[1]}); skipping pair."
                )
                continue

            acts_dict_cf = {}
            for sid, hook_point in sorted_sae_items:
                model.add_hook(hook_point, make_fwd_hook(hook_point, acts_dict_cf), dir='fwd')
                
            with torch.no_grad():
                model(cf_tokens)
                
            model.reset_hooks()
            
            for sid, hook_point in sorted_sae_items:
                layer_idx = int(sid.split('.')[1])
                sae = sae_objects[sid]
                
                acts_clean = acts_dict_clean[hook_point][0]
                acts_cf = acts_dict_cf[hook_point][0]

                f_clean = sae.encode(acts_clean.unsqueeze(0))[0]
                f_corrupt = sae.encode(acts_cf.unsqueeze(0))[0]

                # Residual-stream gradient of the metric at this hook point.
                if args.attribution_method == "ig":
                    grad_acts = integrated_gradient_grad_x(
                        model, clean_tokens, hook_point, acts_clean, acts_cf,
                        metric_fn, item, args.ig_steps
                    )
                else:
                    grad_acts = acts_dict_clean[hook_point].grad[0]

                W_dec_T = sae.W_dec.T if sae.W_dec.shape[0] == sae.cfg.d_sae else sae.W_dec
                grad_f = torch.matmul(grad_acts, W_dec_T)
                
                attributions = direction * grad_f * (f_corrupt - f_clean)
                
                if args.save_attributions:
                    raw_mask = (f_clean > 0.0) | (f_corrupt > 0.0)
                    raw_passing = raw_mask.nonzero(as_tuple=False)
                    
                    if raw_passing.numel() > 0:
                        token_indices = raw_passing[:, 0]
                        feature_indices = raw_passing[:, 1]
                        vals = attributions[token_indices, feature_indices]
                        
                        token_indices_cpu = token_indices.cpu().numpy()
                        feature_indices_cpu = feature_indices.cpu().numpy()
                        vals_cpu = vals.cpu().detach().to(torch.float16).numpy()
                        
                        num_events = len(token_indices_cpu)
                        total_elements_written += num_events
                        
                        batch_data['counterfactual'].extend([cf_name] * num_events)
                        batch_data['prompt_idx'].extend([p_idx] * num_events)
                        batch_data['token_idx'].extend(token_indices_cpu.tolist())
                        batch_data['token_id'].extend([clean_token_ids[t].item() for t in token_indices_cpu])
                        batch_data['feature_idx'].extend(feature_indices_cpu.tolist())
                        batch_data['attribution'].extend(vals_cpu.tolist())
                        batch_data['layer'].extend([layer_idx] * num_events)

                    # SAE reconstruction-error node (Marks et al. 2024, arXiv:2403.19647):
                    # the residual not captured by any feature. Stored with
                    # feature_idx = -1 so a layer's feature attributions plus the
                    # error attribution sum to the residual-stream IE. Dense by
                    # nature: one value per token.
                    err_clean = acts_clean - sae.decode(f_clean.unsqueeze(0))[0]
                    err_cf = acts_cf - sae.decode(f_corrupt.unsqueeze(0))[0]
                    err_attr = direction * (grad_acts * (err_cf - err_clean)).sum(dim=-1)
                    err_cpu = err_attr.cpu().detach().to(torch.float16).numpy()
                    n_err = len(err_cpu)
                    total_elements_written += n_err

                    batch_data['counterfactual'].extend([cf_name] * n_err)
                    batch_data['prompt_idx'].extend([p_idx] * n_err)
                    batch_data['token_idx'].extend(range(n_err))
                    batch_data['token_id'].extend([clean_token_ids[t].item() for t in range(n_err)])
                    batch_data['feature_idx'].extend([-1] * n_err)
                    batch_data['attribution'].extend(err_cpu.tolist())
                    batch_data['layer'].extend([layer_idx] * n_err)

        for hook_point in acts_dict_clean:
            if acts_dict_clean[hook_point].grad is not None:
                acts_dict_clean[hook_point].grad.zero_()
        acts_dict_clean.clear()

        if args.save_attributions and (p_idx + 1) % args.chunk_size == 0:
            flush_batch()

    # Finalize attributions if enabled
    if args.save_attributions:
        flush_batch()
        if total_elements_written == 0:
            logger.warning(
                "No attributions written: no counterfactuals found. Counterfactual "
                "patching needs items with nested dict columns containing a 'prompt' key "
                "(e.g. MIB-bench with --no_concat); datasets like pile-10k have none."
            )
        logger.info(f"Attribution values saved: {total_elements_written} total")
        logger.info(f"Output directory dataset: {args.attributions_dir}")

    logger.info("Execution complete.")

if __name__ == "__main__":
    main()
