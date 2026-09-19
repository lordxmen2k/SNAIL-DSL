"""Render a sample SNAIL program to SVG.

Demonstrates the DAG renderer. Writes SVG to dist/render_demo.svg by
default or to the path given as argv[1].

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import sys
from pathlib import Path

from pydantic import BaseModel

from snail import NodeContext, NodeResult, OODSignal, Program, edge, node
from snail.render import render_program_svg


class In(BaseModel):
    text: str


class IntentOk(BaseModel):
    intent: str
    confidence: float


class Intent(NodeResult):
    ok: IntentOk | None = None
    ood: OODSignal | None = None


class ScoreOk(BaseModel):
    score: float


class Score(NodeResult):
    ok: ScoreOk | None = None
    ood: OODSignal | None = None


@node(name="classify_intent", input_schema=In, output_schema=Intent, distribution="customer_intents_v3")
def classify_intent(ctx, w, payload):
    text = payload.text if hasattr(payload, "text") else payload.get("text", "")
    return Intent(ok=IntentOk(intent="billing", confidence=0.9))


@node(name="extract_score", input_schema=In, output_schema=Score, distribution="score_v1")
def extract_score(ctx, w, payload):
    return Score(ok=ScoreOk(score=0.85))


@node(name="decide", input_schema=In, output_schema=Score, distribution="deterministic")
def decide(ctx, w, payload):
    return Score(ok=ScoreOk(score=1.0))


demo_program = Program(
    name="render_demo",
    nodes=[classify_intent, extract_score, decide],
    edges=[edge(classify_intent.ok), edge(extract_score.ok)],
)


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dist/render_demo.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    svg = render_program_svg(demo_program, title=demo_program.name)
    out.write_text(svg)
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
