"""Example 3: Email triage pipeline — DeterministicNode + HostedNode + frozen classifier.

Demonstrates mixing the three node kinds in one program.
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel

from snail import node, Program, edge, NodeContext, NodeResult, OODSignal
from snail.wrappers import DeterministicNode, HostedNode


class EmailOk(BaseModel):
    sender: str
    subject: str
    body: str


class Email(NodeResult):
    ok: EmailOk | None = None
    ood: OODSignal | None = None


class SpamOk(BaseModel):
    is_spam: bool
    spam_score: float
    confidence: float


class Spam(NodeResult):
    ok: SpamOk | None = None
    ood: OODSignal | None = None


class SummaryOk(BaseModel):
    one_liner: str
    action_items: list[str]
    confidence: float


class Summary(NodeResult):
    ok: SummaryOk | None = None
    ood: OODSignal | None = None


class FolderOk(BaseModel):
    folder: Literal["inbox", "spam", "urgent", "archive"]
    reasoning: str


class Folder(NodeResult):
    ok: FolderOk | None = None
    ood: OODSignal | None = None


@node(
    name="parse_email",
    input_schema=dict,
    output_schema=Email,
    distribution="deterministic",
    frozen_weights="",
)
def parse_email(ctx, weights, raw_email):
    # Stub: real impl uses email.parser.
    return Email(ok=EmailOk(
        sender=raw_email.get("from", "unknown"),
        subject=raw_email.get("subject", ""),
        body=raw_email.get("body", ""),
    ))


@node(
    name="spam_classifier",
    input_schema=dict,
    output_schema=Spam,
    distribution="spam_v3",
    frozen_weights="",
    confidence_threshold=0.7,
)
def spam_classifier(ctx, weights, email):
    body = email.body.lower() if hasattr(email, "body") else ""
    is_spam = "viagra" in body or "lottery" in body or "free money" in body
    score = 0.95 if is_spam else 0.05
    return Spam(ok=SpamOk(is_spam=is_spam, spam_score=score, confidence=0.93))


summarize = HostedNode(
    name="summarize_email",
    input_schema=dict,
    output_schema=Summary,
    distribution="email_summary_v1",
    endpoint="anthropic://claude-sonnet-4-5",
    prompt_template="Summarize this email in one line and list 2-3 action items. "
                    "Subject: {subject}\nBody: {body}\n"
                    "Output JSON: {one_liner: str, action_items: list[str]}",
    api_key_env="ANTHROPIC_API_KEY",
    confidence_threshold=0.6,
)


route = DeterministicNode(
    name="route_email",
    input_schema=dict,
    output_schema=Folder,
    fn=lambda spam: Folder(ok=FolderOk(
        folder="spam" if getattr(spam, "is_spam", False) else "inbox",
        reasoning="spam_score_threshold",
    )),
    ood_on_none=False,
    distribution="deterministic",
)


human_review = DeterministicNode(
    name="human_review",
    input_schema=dict,
    output_schema=Folder,
    fn=lambda _: Folder(ok=FolderOk(folder="urgent", reasoning="ood_routed")),
    ood_on_none=False,
    distribution="deterministic",
)


email_triage = Program(
    name="email_triage_v1",
    nodes=[parse_email, spam_classifier, summarize, route, human_review],
    edges=[
        edge(parse_email.ok),
        edge(spam_classifier.ok),
        edge(parse_email.ood),
        edge(spam_classifier.ood),
        edge(summarize.ood),
    ],
)


if __name__ == "__main__":
    sample = {
        "from": "boss@example.com",
        "subject": "Q4 planning",
        "body": "Please review the Q4 plan before Friday's meeting.",
    }
    result = email_triage.run(sample)
    print("Manifest:", result.manifest.to_json(indent=None))
    for n, out in result.outputs.items():
        v = "OOD" if out.is_ood else "OK"
        print(f"  {n:20s} → {v}")
    print("Final:", result.terminal.ok if result.terminal.is_ok else result.terminal.ood)
