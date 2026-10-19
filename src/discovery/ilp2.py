"""ILP² Miner: Petri net discovery from partially ordered event logs.

Port of Folz-Weinstein, Bergenthum, Desel & Kovář, "ILP² Miner – Process
Discovery for Partially Ordered Event Logs Using Integer Linear Programming",
PETRI NETS 2023 (LNCS 13929), following the reference TypeScript
implementation (ILPN/ILPN-Components @ c542298,
src/lib/algorithms/pn/synthesis/ilp2-miner/), with scipy's HiGHS MILP in
place of glpk.js.

One ILP per place. Variables: the place's initial marking and its binary arc
weights to and from every activity, plus a token flow per partial order (into
each event from the initial marking, along each arc of the order, and from
each maximal event to the final marking). Every partial order must be
executable: tokens reaching an event cover what it consumes (firing rule),
and what flows out of it equals what flows in, minus consumed, plus produced.
The objective, the total token flow, favours compact places. One ILP per
one-way directly-follows pair (a, b) forces an arc a -> p -> b.

Deviations from the reference implementation:
- identical partial orders are solved once, their flows weighted by count
  (same optimum as one block per copy); the reference merges isomorphic
  orders without weighting.
- pairs whose ILP is infeasible are skipped with a warning, not turned into
  a place.
- a transition without input places is left always enabled; the reference
  adds a place with one token, which lets it fire only once.
- the artificial stop transition is removed too, and the final marking is
  the one the most frequent partial order reaches (marking equation).
"""
import warnings
from collections import Counter

import networkx as nx
import numpy as np
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

START, STOP = "▶", "■"   # the reference's artificial start/stop labels


def _variants(partial_orders, add_start_stop):
    """Transitively reduce each order, add start/stop, and count duplicates."""
    counts = Counter()
    for labels, arcs in partial_orders:
        labels = list(labels)
        if START in labels or STOP in labels:
            raise ValueError(f"labels {START!r} and {STOP!r} are reserved")
        g = nx.DiGraph()
        g.add_nodes_from(range(len(labels)))
        g.add_edges_from(arcs)
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError("partial order has a cycle")
        g = nx.transitive_reduction(g)
        edges = list(g.edges)
        if add_start_stop:
            n = len(labels)
            edges += [(n, v) for v in g if g.in_degree(v) == 0]
            edges += [(v, n + 1) for v in g if g.out_degree(v) == 0]
            labels += [START, STOP]
        # ponytail: exact match on (labels, arcs) in input order, not
        # isomorphism; enough when orders come from one deterministic builder.
        counts[(tuple(labels), tuple(sorted(edges)))] += 1
    return counts


def discover_ilp2(partial_orders, add_start_stop=True):
    """Mine a Petri net from partial orders with the ILP² Miner.

    Args:
        partial_orders: iterable of (labels, arcs). Event i has activity
            labels[i]; an arc (i, j) means event i precedes event j. Arcs may
            be the full order or its skeleton (it is reduced first).
        add_start_stop: frame every order with artificial start and stop
            events, as the reference does, so that places marking the first
            activities and collecting the last ones are found. They are
            removed from the returned net.

    Returns:
        (net, initial_marking, final_marking), as pm4py's miners do.

    Raises:
        ValueError: if an order has a cycle or uses a reserved label.
    """
    variants = _variants(partial_orders, add_start_stop)
    acts = sorted({lab for labels, _ in variants for lab in labels})
    idx = {a: k for k, a in enumerate(acts)}
    n_place = 1 + 2 * len(acts)                        # m0, in_t..., out_t...

    def w_in(a):                                        # weight p -> a
        return 1 + idx[a]

    def w_out(a):                                       # weight a -> p
        return 1 + len(acts) + idx[a]

    cost = [0.0] * n_place
    rows, cols, vals, lo, hi = [], [], [], [], []

    def flow(weight):
        cost.append(float(weight))
        return len(cost) - 1

    def constrain(terms, lb, ub):
        for c, v in terms:
            rows.append(len(lo))
            cols.append(c)
            vals.append(v)
        lo.append(lb)
        hi.append(ub)

    follows = set()
    for (labels, arcs), count in variants.items():
        init = [flow(count) for _ in labels]
        into = [[] for _ in labels]
        out_of = [[] for _ in labels]
        for i, j in arcs:
            v = flow(count)
            out_of[i].append(v)
            into[j].append(v)
            follows.add((labels[i], labels[j]))
        for e, lab in enumerate(labels):
            arriving = [(init[e], 1)] + [(v, 1) for v in into[e]]
            leaving = [(v, -1) for v in out_of[e]] or [(flow(count), -1)]
            constrain(arriving + [(w_in(lab), -1)], 0, np.inf)               # firing rule
            constrain(arriving + leaving + [(w_in(lab), -1), (w_out(lab), 1)], 0, 0)  # token flow
        constrain([(v, 1) for v in init] + [(0, -1)], 0, 0)                 # initial marking

    a_mat = coo_matrix((vals, (rows, cols)), shape=(len(lo), len(cost))).tocsr()
    constraints = LinearConstraint(a_mat, lo, hi)
    ub = np.full(len(cost), np.inf)
    ub[:n_place] = 1                                    # binary marking and arc weights
    integrality = np.ones(len(cost))

    places, infeasible = set(), []
    for a, b in sorted(p for p in follows if p[::-1] not in follows):
        lb = np.zeros(len(cost))
        lb[w_out(a)] = lb[w_in(b)] = 1
        res = milp(cost, integrality=integrality, bounds=Bounds(lb, ub), constraints=constraints)
        if not res.success:
            infeasible.append((a, b))
            continue
        x = np.round(res.x[:n_place]).astype(int)
        places.add((int(x[0]), frozenset(t for t in acts if x[w_out(t)]),
                    frozenset(t for t in acts if x[w_in(t)])))
    if infeasible:
        warnings.warn(f"ILP² found no place for {len(infeasible)} pairs: {infeasible}",
                      stacklevel=2)
    return _build_net(places, acts, max(variants, key=variants.get)[0])


def _build_net(places, acts, reference_labels):
    """pm4py net from (m0, preset, postset) places, minus start and stop."""
    kept = set()
    for m0, pre, post in places:
        if START in post or STOP in pre:               # only start's input / stop's output
            continue
        pre_, post_ = pre - {START}, post - {STOP}
        if pre_ or post_:
            kept.add((m0 + (START in pre), pre_, post_))

    net, im, fm = PetriNet("ilp2"), Marking(), Marking()
    trans = {a: PetriNet.Transition(a, a) for a in acts if a not in (START, STOP)}
    net.transitions.update(trans.values())
    fired = Counter(lab for lab in reference_labels if lab in trans)
    for k, (tokens, pre, post) in enumerate(sorted(kept, key=lambda p: (sorted(p[1]), sorted(p[2]), p[0]))):
        p = PetriNet.Place(f"p{k}")
        net.places.add(p)
        for t in pre:
            add_arc_from_to(trans[t], p, net)
        for t in post:
            add_arc_from_to(p, trans[t], net)
        if tokens:
            im[p] = tokens
        end = tokens + sum(fired[t] for t in pre) - sum(fired[t] for t in post)
        if end:
            fm[p] = end
    return net, im, fm
