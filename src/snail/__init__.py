"""SNAIL — Single Node Activated Inference Layer.

A Python DSL for composing frozen, single-pass neural primitives
into statically-typed dataflow programs.

Public API (import from `snail`):
    node           — decorator that wraps a function as a frozen SNAIL node
    NodeContext    — per-call context passed to node bodies
    Program        — typed DAG container for nodes
    edge           — builder for typed connections between nodes
    escalate       — builder for OOD → heavier-node routing (v0.3.0)
    parallel_edges — builder for fan-out routing (v0.3.0)
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
from snail.escalation import escalate, EscalationSpec, CostTier
from snail.parallel import parallel_edges, ParallelGroup
from snail.types.result import NodeResult
from snail.types.ood import OODSignal
from snail.manifest import Manifest, ManifestBuilder

__version__ = "0.3.0"

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
    # v0.3.0 workflow primitives
    "escalate",
    "EscalationSpec",
    "CostTier",
    "parallel_edges",
    "ParallelGroup",
    # types
    "NodeResult",
    "OODSignal",
    # observability
    "Manifest",
    "ManifestBuilder",
]
