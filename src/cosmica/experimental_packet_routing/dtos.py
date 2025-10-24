from __future__ import annotations

from networkx import Graph

__all__ = [
    "PacketRoutingResult",
    "PacketRoutingSetting",
]

import logging
import pickle
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from cosmica.experimental_packet_routing.case_definitions import BackupCaseType
    from cosmica.models.demand import Demand
    from cosmica.models.node import NodeGID

from typing import TYPE_CHECKING

from cosmica.experimental_packet_routing.comm_data import CommDataDemand, CommDataLSA
from cosmica.experimental_packet_routing.node_knowledge import NodeKnowledge
from cosmica.models.node import Node

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True, slots=True)
class PacketRoutingResult:
    all_graphs_after_simulation: list[Graph] = field(default_factory=list[Graph])
    node_knowledge_known_by_each_node: dict[Node, NodeKnowledge] = field(
        default_factory=dict[Node, NodeKnowledge],
    )
    comm_data_demand_list: list[CommDataDemand] = field(default_factory=list[CommDataDemand])
    comm_data_lsa_list: list[CommDataLSA] = field(default_factory=list[CommDataLSA])

    # Save methods
    def save_all_graphs_after_simulation(self, save_path: Path) -> None:
        if self.all_graphs_after_simulation is not None:
            with save_path.open("wb") as f:
                pickle.dump(self.all_graphs_after_simulation, f)
            logger.info(f"Saved the simulation results of graph to {save_path}")
        else:
            logger.info("No graph data to save.")

    def save_node_knowledge_known_by_each_node(self, save_path: Path) -> None:
        if self.node_knowledge_known_by_each_node is not None:
            with save_path.open("wb") as f:
                pickle.dump(self.node_knowledge_known_by_each_node, f)
            logger.info(f"Saved the simulation results of node knowledge to {save_path}")
        else:
            logger.info("No node knowledge data to save.")

    def save_comm_data_demand_list(self, save_path: Path) -> None:
        if self.comm_data_demand_list is not None:
            with save_path.open("wb") as f:
                pickle.dump(self.comm_data_demand_list, f)
            logger.info(f"Saved the simulation results of comm data demand to {save_path}")
        else:
            logger.info("No comm data demand to save.")

    def save_comm_data_lsa_list(self, save_path: Path) -> None:
        if self.comm_data_lsa_list is not None:
            with save_path.open("wb") as f:
                pickle.dump(self.comm_data_lsa_list, f)
            logger.info(f"Saved the simulation results of comm data lsa to {save_path}")
        else:
            logger.info("No comm data lsa to save.")

    # Modified load method
    @classmethod
    def load(
        cls,
        graphs_path: Path | None = None,
        node_knowledge_path: Path | None = None,
        comm_data_demand_path: Path | None = None,
        comm_data_lsa_path: Path | None = None,
    ) -> Self:
        """指定されたパスからpickleファイルを読み込み."""
        all_graphs_after_simulation = []
        if graphs_path is not None and graphs_path.exists():
            with graphs_path.open("rb") as f:
                all_graphs_after_simulation = pickle.load(f)  # noqa: S301
            logger.info(f"Loaded all graphs after simulation: {len(all_graphs_after_simulation)}")
        else:
            logger.info("No graph data to load.")

        node_knowledge_known_by_each_node = {}
        if node_knowledge_path is not None and node_knowledge_path.exists():
            with node_knowledge_path.open("rb") as f:
                node_knowledge_known_by_each_node = pickle.load(f)  # noqa: S301
            logger.info(f"Loaded network information: {len(node_knowledge_known_by_each_node)}")
        else:
            logger.info("No node knowledge data to load.")

        comm_data_demand_list = []
        if comm_data_demand_path is not None and comm_data_demand_path.exists():
            with comm_data_demand_path.open("rb") as f:
                comm_data_demand_list = pickle.load(f)  # noqa: S301
            logger.info(f"Loaded comm data demand: {len(comm_data_demand_list)}")
        else:
            logger.info("No comm data demand to load.")

        comm_data_lsa_list = []
        if comm_data_lsa_path is not None and comm_data_lsa_path.exists():
            with comm_data_lsa_path.open("rb") as f:
                comm_data_lsa_list = pickle.load(f)  # noqa: S301
            logger.info(f"Loaded comm data lsa: {len(comm_data_lsa_list)}")
        else:
            logger.info("No comm data lsa to load.")

        return cls(
            all_graphs_after_simulation=all_graphs_after_simulation,
            node_knowledge_known_by_each_node=node_knowledge_known_by_each_node,
            comm_data_demand_list=comm_data_demand_list,
            comm_data_lsa_list=comm_data_lsa_list,
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class PacketRoutingSetting:
    time: npt.NDArray[np.datetime64]
    nodes_dict: dict[NodeGID, Node]
    demands: list[Demand]
    backup_case: BackupCaseType
    hop_limit: int
    packet_size: int

    def save(self, save_path: Path) -> None:
        with save_path.open("wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved the packet routing setting to {save_path}")

    @classmethod
    def load(cls, load_path: Path) -> PacketRoutingSetting:
        with load_path.open("rb") as f:
            return pickle.load(f)  # noqa: S301
        logger.info(f"Loaded the packet routing setting from {load_path}")
        return None
