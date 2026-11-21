import pandas as pd
import pytest
from pm4py.objects.petri_net import semantics
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to

from src.discovery.object_centric import merge_nets

# A partial order is (labels, arcs): event i has label labels[i]; (i, j) means i < j.
DIAMOND = (["a", "b", "c", "d"], [(0, 1), (0, 2), (1, 3), (2, 3)])   # b and c concurrent


def _fits(net, im, fm, *traces):
    """Alignment fitness per trace (1.0 = the net replays it exactly)."""
    import pm4py
    rows = [{"case:concept:name": str(k), "concept:name": a,
             "time:timestamp": pd.Timestamp("2000-01-01") + pd.Timedelta(seconds=i)}
            for k, trace in enumerate(traces) for i, a in enumerate(trace)]
    return [r["fitness"] for r in pm4py.conformance_diagnostics_alignments(pd.DataFrame(rows), net, im, fm)]


def _chain(name, labels):
    """Hand-built workflow net: source -> labels[0] -> ... -> sink, optionally with a silent step first."""
    net, im, fm = PetriNet(name), Marking(), Marking()
    places = [PetriNet.Place(f"p{i}") for i in range(len(labels) + 1)]
    net.places.update(places)
    for i, lab in enumerate(labels):
        t = PetriNet.Transition(f"t{i}", lab)
        net.transitions.add(t)
        add_arc_from_to(places[i], t, net)
        add_arc_from_to(t, places[i + 1], net)
    im[places[0]], fm[places[-1]] = 1, 1
    return net, im, fm


def _fire(net, im, labels):
    """Fire labelled transitions in order from im; the final marking, or None if one is not enabled."""
    by_label = {t.label: t for t in net.transitions}
    m = im
    for lab in labels:
        if not semantics.is_enabled(by_label[lab], net, m):
            return None
        m = semantics.execute(by_label[lab], net, m)
    return m


# ---------------------------------------------------------------- discover_powl_po

def test_powl_po_accepts_every_linearisation_of_a_diamond_and_nothing_else():
    from src.discovery.object_centric import discover_powl_po
    net, im, fm = discover_powl_po([DIAMOND])
    assert {t.label for t in net.transitions if t.label} == {"a", "b", "c", "d"}
    assert _fits(net, im, fm, "abcd", "acbd") == [1.0, 1.0]
    assert all(f < 1.0 for f in _fits(net, im, fm, "abdc", "bacd"))


def test_powl_po_closes_the_order_itself():
    # Arcs a<b, b<c only: a<c must still hold.
    from src.discovery.object_centric import discover_powl_po
    net, im, fm = discover_powl_po([(["a", "b", "c"], [(0, 1), (1, 2)])])
    assert _fits(net, im, fm, "abc") == [1.0]
    assert _fits(net, im, fm, "acb")[0] < 1.0


def test_powl_po_learns_an_optional_activity_across_orders():
    from src.discovery.object_centric import discover_powl_po
    net, im, fm = discover_powl_po([(["a", "b"], [(0, 1)]), (["a"], [])])
    assert _fits(net, im, fm, "ab", "a") == [1.0, 1.0]


def test_powl_po_skips_empty_orders_and_rejects_all_empty():
    from src.discovery.object_centric import discover_powl_po
    net, im, fm = discover_powl_po([([], []), (["a"], [])])
    assert _fits(net, im, fm, "a") == [1.0]
    with pytest.raises(ValueError, match="no events"):
        discover_powl_po([([], [])])


# ---------------------------------------------------------------- merge_nets

def test_merge_fuses_shared_labels_and_keeps_one_source_and_sink_per_type():
    net, im, fm = merge_nets({"x": _chain("x", ["a", "b"]), "y": _chain("y", ["c", "b"])})
    assert sorted(t.label for t in net.transitions) == ["a", "b", "c"]
    assert len(net.places) == 6
    assert sorted(p.name for p in im) == ["x:p0", "y:p0"]
    assert sorted(p.name for p in fm) == ["x:p2", "y:p2"]
    b = next(t for t in net.transitions if t.label == "b")
    assert sorted(a.source.name for a in b.in_arcs) == ["x:p1", "y:p1"]


def test_merged_shared_transition_synchronises_the_types():
    net, im, fm = merge_nets({"x": _chain("x", ["a", "b"]), "y": _chain("y", ["c", "b"])})
    assert _fire(net, im, ["a", "b"]) is None                # b waits for c on y
    assert _fire(net, im, ["a", "c", "b"]) == fm
    assert _fire(net, im, ["c", "a", "b"]) == fm


def test_merge_keeps_silent_transitions_apart():
    nets = {}
    for k in ("x", "y"):
        net, im, fm = _chain(k, ["a"] if k == "x" else ["c"])
        tau = PetriNet.Transition("tau", None)
        net.transitions.add(tau)
        add_arc_from_to(next(iter(fm)), tau, net)
        nets[k] = (net, im, fm)
    net, _, _ = merge_nets(nets)
    assert sum(t.label is None for t in net.transitions) == 2


def test_merge_rejects_a_label_repeated_within_one_type():
    with pytest.raises(ValueError, match="'a'.*'x'"):
        merge_nets({"x": _chain("x", ["a", "a"]), "y": _chain("y", ["a"])})
