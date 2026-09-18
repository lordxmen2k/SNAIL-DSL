"""Example 4: A ML training pipeline expressed as a SNAIL program.

This shows the unusual case: SNAIL can also describe how a model gets
TRAINED, not just how it runs at inference. Each "node" is one step
in the training loop: generate synthetic data, train, evaluate, freeze,
emit manifest.
"""

from __future__ import annotations
from pydantic import BaseModel

from snail import node, Program, edge, NodeContext, NodeResult, OODSignal
from snail.wrappers import DeterministicNode


class DataOk(BaseModel):
    n_samples: int
    distribution: str


class Data(NodeResult):
    ok: DataOk | None = None
    ood: OODSignal | None = None


class TrainedOk(BaseModel):
    weights_path: str
    epochs: int
    final_loss: float


class Trained(NodeResult):
    ok: TrainedOk | None = None
    ood: OODSignal | None = None


class EvalOk(BaseModel):
    accuracy: float
    ood_rate: float


class Eval(NodeResult):
    ok: EvalOk | None = None
    ood: OODSignal | None = None


class FrozenOk(BaseModel):
    frozen_path: str
    sha256: str


class Frozen(NodeResult):
    ok: FrozenOk | None = None
    ood: OODSignal | None = None


generate_data = DeterministicNode(
    name="generate_synthetic_data",
    input_schema=dict,
    output_schema=Data,
    fn=lambda cfg: Data(ok=DataOk(
        n_samples=10000,
        distribution=cfg.get("distribution", "synthetic_v1"),
    )),
    ood_on_none=False,
    distribution="deterministic",
)


train_model = DeterministicNode(
    name="train_model",
    input_schema=dict,
    output_schema=Trained,
    fn=lambda data: Trained(ok=TrainedOk(
        weights_path=f"/tmp/weights_{getattr(data, 'distribution', 'x')}.safetensors",
        epochs=10,
        final_loss=0.034,
    )),
    ood_on_none=False,
    distribution="deterministic",
)


evaluate_model = DeterministicNode(
    name="evaluate_model",
    input_schema=dict,
    output_schema=Eval,
    fn=lambda trained: Eval(ok=EvalOk(
        accuracy=0.94,
        ood_rate=0.02,
    )),
    ood_on_none=False,
    distribution="deterministic",
)


freeze_weights = DeterministicNode(
    name="freeze_and_pin_weights",
    input_schema=dict,
    output_schema=Frozen,
    fn=lambda eval_result: Frozen(ok=FrozenOk(
        frozen_path="/tmp/frozen.snail.json",
        sha256="abc123def456" * 4,
    )),
    ood_on_none=False,
    distribution="deterministic",
)


human_review = DeterministicNode(
    name="human_review",
    input_schema=dict,
    output_schema=Frozen,
    fn=lambda _: Frozen(ok=FrozenOk(
        frozen_path="/tmp/REJECTED.txt",
        sha256="0" * 64,
    )),
    ood_on_none=False,
    distribution="deterministic",
)


training_pipeline = Program(
    name="training_pipeline_v1",
    nodes=[generate_data, train_model, evaluate_model, freeze_weights, human_review],
    edges=[
        edge(generate_data.ok),
        edge(train_model.ok),
        edge(evaluate_model.ok),
        edge(generate_data.ood),
        edge(train_model.ood),
        edge(evaluate_model.ood),
    ],
)


if __name__ == "__main__":
    result = training_pipeline.run({"distribution": "synthetic_invoices_v2"})
    print("Manifest:", result.manifest.to_json(indent=None))
    for n, out in result.outputs.items():
        v = "OOD" if out.is_ood else "OK"
        print(f"  {n:30s} → {v}")
