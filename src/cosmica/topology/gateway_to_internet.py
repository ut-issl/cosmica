__all__ = [
    "GatewayToInternetTopologyBuilder",
]

from collections.abc import Collection, Hashable
from typing import Any

import networkx as nx

from cosmica.models import Gateway, Internet, Node


class GatewayToInternetTopologyBuilder:
    def build[GatewayId: Hashable, InternetId: Hashable](
        self,
        *,
        gateways: Collection[Gateway[GatewayId]],
        internet: Internet[InternetId],
    ) -> nx.DiGraph:
        graph: nx.Graph[Node[Any]] = nx.Graph()
        graph.add_nodes_from(gateways)
        graph.add_node(internet)
        graph.add_edges_from((gateway, internet) for gateway in gateways)

        # Each physical link is bidirectional: represent it as two directed edges
        return graph.to_directed()
