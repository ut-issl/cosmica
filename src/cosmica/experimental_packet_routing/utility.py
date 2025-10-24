"""Utility functions for experimental packet routing."""

__all__ = [
    "get_edge_data",
    "has_edge_bidirectional",
    "remove_edge_safe",
]

from networkx import Graph

from cosmica.models.node import Node


def has_edge_bidirectional(graph: Graph, u: Node, v: Node) -> bool:
    """Check if edge exists in either direction (u,v) or (v,u)."""
    return graph.has_edge(u, v) or graph.has_edge(v, u)


def remove_edge_safe(graph: Graph, u: Node, v: Node) -> bool:
    """Safely remove edge in either direction. Returns True if edge was removed."""
    if graph.has_edge(u, v):
        graph.remove_edge(u, v)
        return True
    elif graph.has_edge(v, u):
        graph.remove_edge(v, u)
        return True
    return False


def get_edge_data(graph: Graph, u: Node, v: Node) -> dict | None:
    """Get edge data for either direction (u,v) or (v,u)."""
    if graph.has_edge(u, v):
        return graph.edges[u, v]
    elif graph.has_edge(v, u):
        return graph.edges[v, u]
    return None
