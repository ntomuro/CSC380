"""Quick end-to-end test: smoke test + mini eval on 6 conversations."""
import pprint
from dotenv import load_dotenv
load_dotenv(".env")

from data.synthetic_dataset import get_dataset
from graph.detection_graph import detection_graph, make_initial_state
from evaluation.metrics import compute_metrics, print_metrics

dataset = get_dataset()

# ── Smoke test ────────────────────────────────────────────────────────────────
test_conv = next(
    c for c in dataset
    if c["label"] == "Injection" and c["category"] == "direct_ignore"
)
print("=" * 60)
print(f"SMOKE TEST — {test_conv['id']} ({test_conv['category']})")
print("=" * 60)
for t in test_conv["turns"]:
    print(f"  [{t['role'].upper()}]: {t['content']}")
print()

initial = make_initial_state(test_conv)
final = detection_graph.invoke(initial)

print("Agent 1 — Intent Analysis:")
pprint.pprint(final["intent_result"])
print()
print("Agent 2 — Instruction Hierarchy:")
pprint.pprint(final["hierarchy_result"])
print()
print("Agent 3 — Risk Classification:")
pprint.pprint(final["risk_result"])
print()
print("Errors:", final["errors"])

verdict = final["risk_result"]["verdict"]
print(f"\nVERDICT: {verdict}  (ground truth: {test_conv['label']})")
assert verdict in ("Injection", "Suspicious"), f"Expected Injection/Suspicious, got {verdict}"
print("Smoke test PASSED.\n")

# ── Mini evaluation: 2 from each label ───────────────────────────────────────
print("=" * 60)
print("MINI EVAL — 6 conversations (2 per label)")
print("=" * 60)

sample = []
for label in ("Benign", "Suspicious", "Injection"):
    sample.extend([c for c in dataset if c["label"] == label][:2])

ground_truth, predictions = [], []
for conv in sample:
    print(f"  {conv['id']} [{conv['label']}] ...", end=" ", flush=True)
    state = detection_graph.invoke(make_initial_state(conv))
    pred = state["risk_result"]["verdict"] if state["risk_result"] else "Suspicious"
    ground_truth.append(conv["label"])
    predictions.append(pred)
    match = "OK" if pred == conv["label"] else f"WRONG (got {pred})"
    print(match)

print()
metrics = compute_metrics(ground_truth, predictions)
print_metrics(metrics)
print("\nAll tests complete.")
