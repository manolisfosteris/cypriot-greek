"""Replay the step-by-step history of finished training runs into Trackio, so old runs appear on the
same dashboard as new ones. Run it once per run: running it again creates duplicate runs. Usage:
  python scripts/backfill_trackio.py runs/finetune_whisper-small_lora/trainer_state.json [...]"""
import json
import sys
from pathlib import Path

import trackio

for path in map(Path, sys.argv[1:]):
    state = json.loads(path.read_text())
    name = path.parent.name.removeprefix("finetune_")            # e.g. whisper-small_lora
    trackio.init(project="cyasr", name=name, config={"backfilled_from": "trainer_state.json",
                 "best_step": int(state["best_model_checkpoint"].rsplit("-", 1)[1])})
    by_step = {}                                                 # merge train and eval lines of the same step
    for entry in state["log_history"]:
        for k, v in entry.items():
            if k in ("step", "epoch") or not isinstance(v, (int, float)):
                continue
            key = "eval/" + k[5:] if k.startswith("eval_") else "train/" + k   # same names as live runs
            by_step.setdefault(entry["step"], {})[key] = v
    for step in sorted(by_step):
        trackio.log(by_step[step], step=step)
    trackio.finish()
    print(f"{name}: {len(by_step)} steps logged")
