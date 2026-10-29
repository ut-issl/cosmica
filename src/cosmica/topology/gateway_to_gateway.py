__all__ = [
    "GatewayToGatewayTopologyBuilder",
]

from collections.abc import Collection, Hashable

import networkx as nx

from cosmica.models import Gateway


class GatewayToGatewayTopologyBuilder:
    def build(
        self,
        *,
        gateways: Collection[Gateway[Hashable]],
    ) -> nx.DiGraph:
        graph = nx.Graph()
        gateway_list = list(gateways)
        graph.add_nodes_from(gateway_list)

        # Connect every pair of gateways (a complete graph).
        for i, gateway1 in enumerate(gateway_list):
            for gateway2 in gateway_list[i + 1 :]:
                graph.add_edge(gateway1, gateway2)

        # Each physical link is bidirectional: represent it as two directed edges
        return graph.to_directed()
