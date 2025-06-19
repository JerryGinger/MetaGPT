from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Set


@dataclass
class GraphNode:
    """Node in a graph of thoughts."""

    id: str
    name: str
    value: Any
    parents: Set[GraphNode] = field(default_factory=set)
    children: Set[GraphNode] = field(default_factory=set)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, GraphNode):
            return self.id == other.id
        return False

    def add_child(self, node: GraphNode):
        self.children.add(node)
        node.parents.add(self)

    def remove_child(self, node: GraphNode):
        self.children.discard(node)
        node.parents.discard(self)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "value": self.value,
            "parents": [parent.id for parent in self.parents],
            "children": [child.id for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict, nodes: dict[str, GraphNode]) -> GraphNode:
        node = cls(id=data["id"], name=data["name"], value=data["value"])
        # Children and parents will be populated when the full graph is built
        return node


@dataclass
class GraphOfThoughts:
    """Graph of thoughts, managing nodes and their relationships."""

    nodes: dict[str, GraphNode] = field(default_factory=dict)

    def add_node(self, node: GraphNode) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Node with id {node.id} already exists.")
        self.nodes[node.id] = node

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self.nodes.get(node_id)

    def remove_node(self, node_id: str) -> bool:
        node = self.get_node(node_id)
        if not node:
            return False

        for parent in list(node.parents):  # Iterate over a copy
            parent.remove_child(node)
        for child in list(node.children):  # Iterate over a copy
            node.remove_child(child)

        del self.nodes[node_id]
        return True

    def add_edge(self, parent_id: str, child_id: str) -> bool:
        parent_node = self.get_node(parent_id)
        child_node = self.get_node(child_id)

        if not parent_node or not child_node:
            return False

        parent_node.add_child(child_node)
        return True

    def remove_edge(self, parent_id: str, child_id: str) -> bool:
        parent_node = self.get_node(parent_id)
        child_node = self.get_node(child_id)

        if not parent_node or not child_node:
            return False

        parent_node.remove_child(child_node)
        return True

    def get_roots(self) -> List[GraphNode]:
        return [node for node in self.nodes.values() if not node.parents]

    def get_leaves(self) -> List[GraphNode]:
        return [node for node in self.nodes.values() if not node.children]

    def to_dict(self) -> dict:
        return {"nodes": {id: node.to_dict() for id, node in self.nodes.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> GraphOfThoughts:
        graph = cls()
        raw_nodes = data.get("nodes", {})

        # First pass: create all nodes
        for node_id, node_data in raw_nodes.items():
            node = GraphNode.from_dict(node_data, graph.nodes)
            graph.add_node(node)

        # Second pass: connect nodes
        for node_id, node_data in raw_nodes.items():
            node = graph.get_node(node_id)
            if node:
                for parent_id in node_data.get("parents", []):
                    parent_node = graph.get_node(parent_id)
                    if parent_node:
                        parent_node.add_child(node)
                # Children are implicitly handled by parents adding them
        return graph


from enum import Enum

class GoTStrategy(Enum):
    """Enumeration of Graph of Thoughts (GoT) strategies."""

    IO = "IO"  # Input-Output prompting (simple prompting)
    COT = "COT"  # Chain-of-Thought prompting
    TOT = "TOT"  # Tree-of-Thoughts prompting (typically a tree, special case of GoT)
    DFS = "DFS"  # Depth-First Search traversal for GoT
    BFS = "BFS"  # Breadth-First Search traversal for GoT
    BEST_FIRST = "BEST_FIRST"  # Best-First Search traversal
    A_STAR = "A_STAR"  # A* Search algorithm
    BEAM_SEARCH = "BEAM_SEARCH" # Beam Search
    # Add other specific GoT strategies as needed
    AGGREGATE = "AGGREGATE" # Aggregate thoughts from different paths
    REFINE = "REFINE" # Iteratively refine thoughts
    SELF_CRITIQUE = "SELF_CRITIQUE" # Self-critique and improve thoughts
    ROLLOUT = "ROLLOUT" # Rollout-based simulation or lookahead
