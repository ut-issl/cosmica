__all__ = [
    "GatewayToInternetTopologyBuilder",
]

from collections.abc import Collection

import networkx as nx

from cosmica.models import Gateway, Internet


class GatewayToInternetTopologyBuilder:
    def build(
        self,
        *,
        gateways: Collection[Gateway],
        internet: Internet,
    ) -> nx.DiGraph:
        graph = nx.Graph()
        graph.add_nodes_from(gateways)
        graph.add_node(internet)
        graph.add_edges_from((gateway, internet) for gateway in gateways)

        # Each physical link is bidirectional: represent it as two directed edges
        return graph.to_directed()
