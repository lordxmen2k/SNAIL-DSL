"""Example 2: Image classifier — ExternalLocalNode wrapping a pretrained model.

This shows how a real pretrained model (e.g. a ResNet) becomes a SNAIL node
through ExternalLocalNode. The weight_pin guarantees silent upgrades are
impossible — if you change the model without updating the pin, the program
refuses to load.
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel

from snail.wrappers import ExternalLocalNode
from snail import Program, edge, NodeContext, NodeResult, OODSignal


class ClassOk(BaseModel):
    label: str
    confidence: float


class ClassResult(NodeResult):
    ok: ClassOk | None = None
    ood: OODSignal | None = None


class MultiClassOk(BaseModel):
    top_k: list[tuple[str, float]]


class MultiClassResult(NodeResult):
    ok: MultiClassOk | None = None
    ood: OODSignal | None = None


# Pretrained ResNet — pinned to a specific version + hash.
# In production: the call would invoke torch.load + the actual ResNet.
def resnet_forward(payload) -> ClassOk:
    """Stub: pretend we ran ResNet on the image.

    `payload` is whatever the upstream node passed — for top-level programs,
    that's the program_input dict.
    """
    # Real impl:
    #   image_path = payload["image_path"]
    #   model = torch.load(weights.raw["path"])
    #   output = model(preprocess(image_path))
    #   return ClassOk(label=decode(output), confidence=softmax(output).max())
    image_path = payload.get("image_path", "") if isinstance(payload, dict) else str(payload)
    if "cat" in image_path:
        return ClassOk(label="tabby_cat", confidence=0.94)
    if "dog" in image_path:
        return ClassOk(label="golden_retriever", confidence=0.91)
    return ClassOk(label="unknown", confidence=0.4)


resnet_classifier = ExternalLocalNode(
    name="resnet50_classifier",
    input_schema=dict,
    output_schema=ClassResult,
    distribution="imagenet_2012",
    call=resnet_forward,
    weight_pin="resnet50@sha256:abc123def456abc123def456abc123def456abc123def456abc123def456",
    confidence_threshold=0.7,
)


# Second pass: top-k via the same model (same weights pin).
topk_classifier = ExternalLocalNode(
    name="resnet50_topk",
    input_schema=dict,
    output_schema=MultiClassResult,
    distribution="imagenet_2012",
    call=lambda path: MultiClassOk(top_k=[("tabby_cat", 0.94), ("tiger_cat", 0.04), ("carton", 0.01)]),
    weight_pin="resnet50@sha256:abc123def456abc123def456abc123def456abc123def456abc123def456",
    confidence_threshold=0.6,
)


resnet_pipeline = Program(
    name="resnet_image_v1",
    nodes=[resnet_classifier, topk_classifier],
    edges=[edge(resnet_classifier.ok)],
)


if __name__ == "__main__":
    result = resnet_pipeline.run({"image_path": "/tmp/cat.jpg"})
    print("Manifest:", result.manifest.to_json(indent=None))
    print("Classification:", result.outputs["resnet50_classifier"].ok)
    print("Top-K:", result.outputs["resnet50_topk"].ok)
