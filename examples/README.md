# SNAIL Examples

Five end-to-end programs showing the architecture in action. Each one
runs with `python examples/<name>.py`.

## 1. invoice_pipeline.py

Pure SNAIL with frozen classifier nodes. Extract invoice total → classify
risk → route for manual review or approval. Tests the `edge(node.ok)` /
`edge(node.ood)` pattern with auto-routing of OOD to terminal sink.

## 2. resnet_classifier.py

ExternalLocalNode wrapping a pretrained model. Shows how a real model
becomes a SNAIL node with weight pinning — silent upgrades are impossible.

## 3. email_triage.py

Mixed program: DeterministicNode (parsing), frozen classifier (spam), and
HostedNode (LLM summarization). Real-world agent recipe — three kinds of
nodes in one DAG.

## 4. customer_support_v3.py

Full customer-support pipeline. The LLM is one node among many; safety
check is mandatory; OOD routes to human review. Demonstrates the
"recipe that's more resilient than a markdown spec" framing.

## 5. ml_training_pipeline.py

SNAIL describes not just inference but the training loop: synthetic
data generation → training → evaluation → freezing → shipping. Every
stage is a typed, deterministic step with audit trail.

## 6. data_validation.py

Pure deterministic SNAIL. Validation as a DAG of pure functions, each
checking one property. OOD means "we don't know if this is valid" →
routes to human queue.

## Running the examples

```bash
pip install -e ".[dev]"
python examples/invoice_pipeline.py
python examples/resnet_classifier.py
python examples/email_triage.py        # uses ANTHROPIC_API_KEY env var
python examples/customer_support_v3.py
python examples/ml_training_pipeline.py
python examples/data_validation.py
```

Example 3 (email_triage) and example 4 (customer_support_v3) reference
hosted LLMs. Set `ANTHROPIC_API_KEY` in your environment to wire a real
provider; without it, the HostedNode returns OOD and the program
gracefully routes.
