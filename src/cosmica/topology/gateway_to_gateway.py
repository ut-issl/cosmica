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
    ) -> nx.Graph:
        graph = nx.Graph()
        gateway_list = list(gateways)
        graph.add_nodes_from(gateway_list)

        # すべてのgateway同士を接続する(完全グラフ)
        for i, gateway1 in enumerate(gateway_list):
            for gateway2 in gateway_list[i + 1 :]:
                graph.add_edge(gateway1, gateway2)

        return graph
