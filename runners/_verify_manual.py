import sys
import pathlib

ROOT = pathlib.Path(r"c:\Users\DIPESH\PycharmProjects\genai-qa-eval")
sys.path.insert(0, str(ROOT))

from runners.build_dashboard import load_all, group_runs, compute_diff, _state
import eval_config as cfg

reports_dir = ROOT / "reports"
all_rows = load_all(reports_dir)
run_groups = group_runs(all_rows, bucket_seconds=cfg.RUN_BUCKET_SECONDS)

print(f"Total runs: {len(run_groups)}")
print()

latest_label, latest_rows = run_groups[-1]
prev_label, prev_rows = run_groups[-2]

print(f"Latest run: {latest_label}")
print(f"Prev run:   {prev_label}")
print()

latest_rag = [r for r in latest_rows if r["metric"] == "Answer Relevancy"]
print("Latest Answer Relevancy rows:")
for r in latest_rag:
    case = r["case"]
    score = r["score"]
    passed = r["passed"]
    state = _state(r)
    print(f"  {case} | score={score} | passed={passed} | state={state}")

print()
nf, np_, drops = compute_diff(prev_rows, latest_rows, cfg.REGRESSION_DELTA)
print(f"Newly failing  ({len(nf)}):")
for p, r in nf:
    print(f"  {r['case']} {r['metric']}: {p['score']} -> {r['score']}")

print(f"Newly passing  ({len(np_)}):")
for p, r in np_:
    print(f"  {r['case']} {r['metric']}: {p['score']} -> {r['score']}")

print(f"Score drops    ({len(drops)}):")
for p, r in drops:
    print(f"  {r['case']} {r['metric']}: {p['score']} -> {r['score']}")

print()
errors = [r for r in latest_rows if r["passed"] is None]
print(f"Error rows in latest: {len(errors)}")
for e in errors:
    print(f"  {e['case']} {e['metric']} score={e['score']} state={_state(e)}")

print()
assert len(nf) >= 1, "Expected at least 1 newly-failing (ECOM_RAG_001)"
assert any(r["case"] == "ECOM_RAG_001" for _, r in nf), "ECOM_RAG_001 not in newly-failing"
assert len(errors) >= 1, "Expected at least 1 Error row (ECOM_RAG_002 blank score)"
assert any(e["case"] == "ECOM_RAG_002" for e in errors), "ECOM_RAG_002 not in errors"

print("=== MANUAL TEST section9: ALL CHECKS PASS ===")
