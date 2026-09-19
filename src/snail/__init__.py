"""SNAIL — Single Node Activated Inference Layer.

A Python DSL for composing frozen, single-pass neural primitives
into statically-typed dataflow programs.

Public API (import from `snail`):
    node           — decorator that wraps a function as a frozen SNAIL node
    NodeContext    — per-call context passed to node bodies
    Program        — typed DAG container for nodes
    edge           — builder for typed connections between nodes
    NodeResult     — base class for all node output schemas
    OODSignal      — out-of-distribution signal type
    Manifest       — per-run audit log

Subpackages:
    snail.wrappers — ExternalLocalNode, HostedNode, DeterministicNode
    snail.lint     — discipline check that catches direct model SDK imports
    snail.types    — internal type definitions

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from snail.node import node, NodeContext, FrozenWeights
from snail.program import Program, Edge, edge, NodeProxy, ProgramRunResult
from snail.types.result import NodeResult
from snail.types.ood import OODSignal
from snail.manifest import Manifest, ManifestBuilder

__version__ = "0.2.1"

__all__ = [
    # core
    "node",
    "NodeContext",
    "FrozenWeights",
    "Program",
    "Edge",
    "edge",
    "NodeProxy",
    "ProgramRunResult",
    # types
    "NodeResult",
    "OODSignal",
    # observability
    "Manifest",
    "ManifestBuilder",
]
