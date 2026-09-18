"""Example 5: Data validation pipeline — pure deterministic SNAIL.

The simplest case: validation is a DAG of pure functions, each one
checking one property. OOD means "we don't know if this is valid" —
the program routes uncertain cases to a human queue.
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel

from snail import node, Program, edge, NodeContext, NodeResult, OODSignal
from snail.wrappers import DeterministicNode


class FieldOk(BaseModel):
    name: str
    value: str


class Field(NodeResult):
    ok: FieldOk | None = None
    ood: OODSignal | None = None


class ValidationOk(BaseModel):
    field_name: str
    valid: bool
    reason: str


class Validation(NodeResult):
    ok: ValidationOk | None = None
    ood: OODSignal | None = None


class ReportOk(BaseModel):
    all_valid: bool
    failures: list[str]


class Report(NodeResult):
    ok: ReportOk | None = None
    ood: OODSignal | None = None


parse_field = DeterministicNode(
    name="parse_field",
    input_schema=dict,
    output_schema=Field,
    fn=lambda raw: Field(ok=FieldOk(name=raw["name"], value=raw["value"])) if "name" in raw and "value" in raw else None,
    ood_on_none=True,
    distribution="deterministic",
)


validate_email = DeterministicNode(
    name="validate_email_format",
    input_schema=dict,
    output_schema=Validation,
    fn=lambda field: Validation(ok=ValidationOk(
        field_name=getattr(field, "name", ""),
        valid="@" in getattr(field, "value", ""),
        reason="email_format",
    )) if getattr(field, "name", "") == "email" else Validation(ok=ValidationOk(
        field_name=getattr(field, "name", ""), valid=True, reason="skipped",
    )),
    ood_on_none=True,
    distribution="deterministic",
)


validate_age = DeterministicNode(
    name="validate_age_range",
    input_schema=dict,
    output_schema=Validation,
    fn=lambda field: Validation(ok=ValidationOk(
        field_name=getattr(field, "name", ""),
        valid=0 <= int(getattr(field, "value", "-1")) <= 150,
        reason="age_range",
    )) if getattr(field, "name", "") == "age" else Validation(ok=ValidationOk(
        field_name=getattr(field, "name", ""), valid=True, reason="skipped",
    )),
    ood_on_none=True,
    distribution="deterministic",
)


validate_phone = DeterministicNode(
    name="validate_phone_format",
    input_schema=dict,
    output_schema=Validation,
    fn=lambda field: Validation(ok=ValidationOk(
        field_name=getattr(field, "name", ""),
        valid=sum(1 for c in getattr(field, "value", "") if c.isdigit()) >= 7,
        reason="phone_format",
    )) if getattr(field, "name", "") == "phone" else Validation(ok=ValidationOk(
        field_name=getattr(field, "name", ""), valid=True, reason="skipped",
    )),
    ood_on_none=True,
    distribution="deterministic",
)


build_report = DeterministicNode(
    name="build_report",
    input_schema=dict,
    output_schema=Report,
    fn=lambda _: Report(ok=ReportOk(all_valid=True, failures=[])),
    ood_on_none=False,
    distribution="deterministic",
)


human_review = DeterministicNode(
    name="human_review",
    input_schema=dict,
    output_schema=Report,
    fn=lambda _: Report(ok=ReportOk(all_valid=False, failures=["ood_routed"])),
    ood_on_none=False,
    distribution="deterministic",
)


validation_pipeline = Program(
    name="data_validation_v1",
    nodes=[
        parse_field,
        validate_email,
        validate_age,
        validate_phone,
        build_report,
        human_review,
    ],
    edges=[
        edge(parse_field.ok),
        edge(validate_email.ok),
        edge(validate_age.ok),
        edge(parse_field.ood),
        edge(validate_email.ood),
        edge(validate_age.ood),
    ],
)


if __name__ == "__main__":
    samples = [
        {"name": "email", "value": "user@example.com"},
        {"name": "age", "value": "30"},
        {"name": "phone", "value": "+1-555-1234"},
    ]
    for sample in samples:
        result = validation_pipeline.run(sample)
        print(f"\nInput: {sample}")
        for n, out in result.outputs.items():
            v = "OOD" if out.is_ood else "OK"
            print(f"  {n:25s} → {v}")
