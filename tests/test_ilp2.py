from itertools import permutations

import pytest
from pm4py.objects.petri_net import semantics

from src.discovery.ilp2 import discover_ilp2

# A partial order is (labels, arcs): event i has label labels[i]; (i, j) means i < j.
SEQ = (["a", "b", "c"], [(0, 1), (1, 2)])
# a < b, a < c, b < d, c < d: b and c are concurrent.
DIAMOND = (["a", "b", "c", "d"], [(0, 1), (0, 2), (1, 3), (2, 3)])


def _replays(net, im, labels):
    """Fire `labels` in order from `im`; return the final marking or None."""
    by_label = {t.label: t for t in net.transitions}
    m = im
    for lab in labels:
        t = by_label[lab]
        if not semantics.is_enabled(t, net, m):
            return None
        m = semantics.execute(t, net, m)
    return m


def _linearisations(po):
    labels, arcs = po
    for order in permutations(range(len(labels))):
        pos = {e: k for k, e in enumerate(order)}
        if all(pos[i] < pos[j] for i, j in arcs):
            yield [labels[e] for e in order]


def _places(net, im):
    return {(im[p], frozenset(a.source.label for a in p.in_arcs),
             frozenset(a.target.label for a in p.out_arcs)) for p in net.places}


def test_a_sequence_replays_and_only_in_order():
    net, im, fm = discover_ilp2([SEQ])
    assert _replays(net, im, ["a", "b", "c"]) == fm
    assert _replays(net, im, ["b"]) is None
    assert _replays(net, im, ["a", "c"]) is None


def test_every_linearisation_of_a_partial_order_replays():
    net, im, fm = discover_ilp2([DIAMOND])
    for trace in _linearisations(DIAMOND):
        assert _replays(net, im, trace) == fm, trace


def test_concurrent_events_need_both_predecessors_to_join():
    net, im, _ = discover_ilp2([DIAMOND])
    assert _replays(net, im, ["a", "b", "d"]) is None
    assert _replays(net, im, ["a", "c", "d"]) is None


def test_every_input_partial_order_replays_with_repeated_labels():
    pos = [(["a", "b", "b", "c"], [(0, 1), (1, 2), (2, 3)]),
           (["a", "b", "c"], [(0, 1), (1, 2)])]
    net, im, fm = discover_ilp2(pos)
    for po in pos:
        for trace in _linearisations(po):
            assert _replays(net, im, trace) is not None, trace


def test_transitive_arcs_do_not_change_the_net():
    closed = (DIAMOND[0], DIAMOND[1] + [(0, 3)])
    assert _places(*discover_ilp2([closed])[:2]) == _places(*discover_ilp2([DIAMOND])[:2])


def test_duplicate_partial_orders_give_the_same_net():
    assert _places(*discover_ilp2([DIAMOND, DIAMOND])[:2]) == _places(*discover_ilp2([DIAMOND])[:2])


def test_every_label_becomes_a_transition_and_no_artificial_ones_remain():
    net, _, _ = discover_ilp2([DIAMOND])
    assert sorted(t.label for t in net.transitions) == ["a", "b", "c", "d"]


def test_rejects_a_cyclic_order():
    with pytest.raises(ValueError, match="cycle"):
        discover_ilp2([(["a", "b"], [(0, 1), (1, 0)])])
