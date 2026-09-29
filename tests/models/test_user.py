from collections.abc import Callable, Hashable
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pytest

from cosmica.models import CommunicationTerminal, RFCommunicationTerminal, StationaryOnGroundUser, User


@dataclass(frozen=True, kw_only=True, slots=True)
class _MinimalUser[T: Hashable](User[T]):
    """A user subclass that does not redeclare `terminals`."""


def _terminal(terminal_id: Hashable) -> CommunicationTerminal[Hashable]:
    return RFCommunicationTerminal(id=terminal_id)


def _minimal_user(user_id: str, terminals: list[CommunicationTerminal[Hashable]]) -> User[str]:
    return _MinimalUser(id=user_id, terminals=terminals)


def _stationary_user(user_id: str, terminals: list[CommunicationTerminal[Hashable]]) -> User[str]:
    return StationaryOnGroundUser(
        id=user_id,
        latitude=np.deg2rad(35.0),
        longitude=np.deg2rad(139.0),
        minimum_elevation=np.deg2rad(10.0),
        terminals=terminals,
    )


type _UserFactory = Callable[[str, list[CommunicationTerminal[Hashable]]], User[str]]

_USER_FACTORIES = pytest.mark.parametrize(
    "make_user",
    [
        pytest.param(_minimal_user, id="minimal-subclass"),
        pytest.param(_stationary_user, id="stationary-on-ground"),
    ],
)


@_USER_FACTORIES
def test_user_with_terminals_is_usable_as_graph_node_and_dict_key(make_user: _UserFactory) -> None:
    user = make_user("user", [_terminal(0)])

    graph = nx.Graph()
    graph.add_node(user)

    assert user in graph
    assert {user: "value"}[user] == "value"


@_USER_FACTORIES
def test_user_identity_ignores_terminals(make_user: _UserFactory) -> None:
    user = make_user("user", [_terminal(0)])
    same_id = make_user("user", [_terminal(1), _terminal(2)])

    assert user == same_id
    assert hash(user) == hash(same_id)
    assert user != make_user("other", [_terminal(0)])


@_USER_FACTORIES
def test_user_terminal_mutation_does_not_change_identity(make_user: _UserFactory) -> None:
    user = make_user("user", [])
    original_hash = hash(user)
    graph = nx.Graph()
    graph.add_node(user)

    user.terminals.append(_terminal(0))

    assert hash(user) == original_hash
    assert user == make_user("user", [])
    assert user in graph
