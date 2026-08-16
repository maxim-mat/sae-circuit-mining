"""Residual-stream ordering for SAE ids.

Shared by main.py (node attributions), edge_attribution.py (edges) and
partial_order.py so all three agree on what integer "layer" means. Keep it
dependency-free — partial_order.py imports it without sae_lens/torch loaded.

Self-check: python src/sae_layers.py
"""


def sae_layer_ordinal(sae_id: str) -> int:
    """Position of an SAE's hook in the residual stream, as a strict order index.

    Block number alone is not a valid index. `gpt2-small-res-jb` ships 12
    `blocks.L.hook_resid_pre` SAEs plus one `blocks.11.hook_resid_post`:
    resid_post of block L is the same tensor as resid_pre of block L+1, so only
    the final block's post-hook is a distinct site and only it gets its own SAE.
    Parsing `L` maps resid_pre_11 and resid_post_11 to the same index, which
    collides two disjoint feature spaces in one partition / dict key and loses
    the upstream->downstream ordering between them. resid_post_L sits after
    block L, so its ordinal is L + 1.

    ponytail: only _post is offset. A release with hook_resid_mid SAEs would
    still collide with _pre; add a case here if one is ever used.
    """
    parts = sae_id.split(".")
    if len(parts) < 2 or not parts[1].isdigit():
        return 0
    block = int(parts[1])
    return block + 1 if sae_id.endswith("_post") else block


def sorted_saes(saes_map: dict) -> list[tuple[int, str, str]]:
    """[(ordinal, sae_id, hook_point)] in residual-stream order."""
    return sorted((sae_layer_ordinal(sid), sid, hp) for sid, hp in saes_map.items())


if __name__ == "__main__":
    # The real gpt2-small-res-jb id set: 12 resid_pre + the final resid_post.
    ids = [f"blocks.{i}.hook_resid_pre" for i in range(12)] + ["blocks.11.hook_resid_post"]
    ordinals = [sae_layer_ordinal(s) for s in ids]
    assert ordinals == list(range(12)) + [12], ordinals
    assert len(set(ordinals)) == len(ids), "ordinals must be unique per SAE"

    saes_map = {s: s for s in ids}
    order = [o for o, _, _ in sorted_saes(saes_map)]
    assert order == sorted(order) == list(range(13)), order
    assert sorted_saes(saes_map)[-1][1] == "blocks.11.hook_resid_post"
    assert sae_layer_ordinal("no_block_number") == 0
    print("sae_layers checks passed")
