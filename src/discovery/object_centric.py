"""Object-centric Petri nets from partially ordered event data.

The three steps of van der Aalst & Berti, "Discovering Object-centric Petri
Nets", Fundamenta Informaticae 175 (2020): project the events onto each object
type, discover one net per type, and merge the nets on their shared activities.
There the per-type miner is the inductive miner on a flattened, totally ordered
log. Here the per-type input stays a partial order. discover_powl_po mines it
with the partial-order POWL miner (Kourani, Park & van der Aalst, "Revealing
Inherent Concurrency in Event Data: A Partial Order Approach to Process
Discovery", arXiv:2509.15346), and any miner with the same (labels, arcs) ->
(net, im, fm) contract, such as src.discovery.ilp2.discover_ilp2, can replace
it. The projection depends on what the caller's objects are, so it stays with
the caller.
"""
from collections import Counter

import networkx as nx
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to


def discover_powl_po(partial_orders):
    """Mine a workflow net from partial orders with the partial-order POWL miner.

    The orders go straight to powl's miner. powl's public
    discover_from_partially_ordered_log rebuilds orders from start/end
    timestamps, which can only express interval orders.

    Args:
        partial_orders: iterable of (labels, arcs), as for discover_ilp2. Event
            i has activity labels[i]; an arc (i, j) means event i precedes
            event j. The arcs may be any generating set of the order: it is
            transitively closed here. The k-th event of a label, in list
            order, is that activity's instance k. Orders with no events are
            skipped.

    Returns:
        (net, initial_marking, final_marking): one source, one sink.

    Raises:
        ValueError: if an order has a cycle, or no order has an event.
    """
    import powl     # optional dependency (requirements.txt); powl 2.4.1 module paths below
    from powl.discovery.partial_order_based.utils.constants import VARIANT_FREQUENCY_KEY
    from powl.discovery.partial_order_based.utils.simplified_objects import ActivityInstance, Graph
    from powl.discovery.partial_order_based.variants.base import miner

    variants = Counter()
    for labels, arcs in partial_orders:
        if not labels:
            continue
        g = nx.DiGraph()
        g.add_nodes_from(range(len(labels)))
        g.add_edges_from(arcs)
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError("partial order has a cycle")
        seen, inst = Counter(), []
        for lab in labels:
            seen[lab] += 1
            inst.append(ActivityInstance(lab, seen[lab]))
        closed = nx.transitive_closure_dag(g).edges
        variants[frozenset(inst), frozenset((inst[i], inst[j]) for i, j in closed)] += 1
    if not variants:
        raise ValueError("the partial orders have no events")
    graphs = [Graph(nodes, edges, {VARIANT_FREQUENCY_KEY: n}) for (nodes, edges), n in variants.most_common()]
    return powl.convert_to_petri_net(miner.apply(graphs))


def merge_nets(nets):
    """Merge per-object-type nets into one object-centric net.

    Transitions with the same label are fused into one transition, which
    synchronises the types that share it. Places and silent transitions are
    renamed "<type>:<name>" and kept apart, so every type keeps its own source
    and sink.

    Args:
        nets: {object_type: (net, initial_marking, final_marking)}.

    Returns:
        (net, initial_marking, final_marking). Each marking is the union of the
        types' markings.

    Raises:
        ValueError: if a label sits on two transitions of one type's net,
            because then which copy to fuse is undefined.
    """
    out, im, fm = PetriNet("object-centric"), Marking(), Marking()
    fused = {}
    for typ, (net, net_im, net_fm) in nets.items():
        repeated = [lab for lab, n in Counter(t.label for t in net.transitions if t.label).items() if n > 1]
        if repeated:
            raise ValueError(f"labels {repeated} repeat within the net of object type {typ!r}")
        node = {p: PetriNet.Place(f"{typ}:{p.name}") for p in net.places}
        out.places.update(node.values())
        for t in net.transitions:
            if t.label is None:
                node[t] = PetriNet.Transition(f"{typ}:{t.name}", None)
            else:
                node[t] = fused.setdefault(t.label, PetriNet.Transition(t.label, t.label))
            out.transitions.add(node[t])
        for a in net.arcs:
            add_arc_from_to(node[a.source], node[a.target], out, weight=a.weight)
        for p, n in net_im.items():
            im[node[p]] = n
        for p, n in net_fm.items():
            fm[node[p]] = n
    return out, im, fm
