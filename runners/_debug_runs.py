import sys
import pathlib

ROOT = pathlib.Path(r"c:\Users\DIPESH\PycharmProjects\genai-qa-eval")
sys.path.insert(0, str(ROOT))
from runners.build_dashboard import load_all, group_runs
import eval_config as cfg

all_rows = load_all(ROOT / "reports")
run_groups = group_runs(all_rows, bucket_seconds=cfg.RUN_BUCKET_SECONDS)

# Show last 4 runs
for label, rows in run_groups[-4:]:
    print(f"Run: {label}")
    for r in rows:
        case = r["case"]
        metric = r["metric"]
        passed = r["passed"]
        score = r["score"]
        print(f"  {case} | {metric} | passed={passed} | score={score}")
    print()
