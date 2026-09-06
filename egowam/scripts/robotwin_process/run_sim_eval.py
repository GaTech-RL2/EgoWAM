#!/usr/bin/env python3
"""RoboTwin closed-loop sim eval sweep for EgoWAM policies (plan §5 / §6.4).

Drives the grid  tasks x {single, cross} x variants x difficulty x seeds  through
``external/RoboTwin/policy/EgoWAM/eval.sh`` (which calls RoboTwin's ``script/eval_policy.py``),
parses each run's success rate and IK-failure markers, and emits:
  * a raw results JSON (per cell: success/total, success%, ik_fail count, returncode),
  * Table 1 (per task: rows {single, cross} x cols {hpt, vae, rae}, success% Easy/Hard),
  * Table 2 (transfer: success(cross) - success(single) per (task, variant)).

The rollout embodiment is always arx-x5 (single & cross alike), so single-vs-cross is a clean
comparison in the same sim (plan §5).

Checkpoint resolution (first hit wins):
  1. --manifest manifest.json : {"<setting>_<task>_<variant>": "/abs/last.ckpt", ...}
  2. --ckpt-root DIR          : DIR/<setting>_<task>_<variant>/last.ckpt

ENV: this must run in an interpreter that has BOTH RoboTwin's SAPIEN stack (sapien, mplib) AND
EgoVerse's torch/egowam. emimic2 lacks sapien/mplib today (see plan §5) — use --dry-run to print
the command matrix without launching, or run under a combined env once available.

Examples:
  # print the full matrix without running anything
  python -m egowam.scripts.robotwin_process.run_sim_eval --dry-run
  # one smoke cell with an explicit ckpt
  python -m egowam.scripts.robotwin_process.run_sim_eval \
      --tasks pick --settings cross --variants vae --difficulties Easy --seeds 0 \
      --ckpt $EGOWAM_ROBOTWIN_ROOT/_smoke/cross_pick_vae/checkpoints/last.ckpt
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EGOWAM_DIR = REPO_ROOT / "external" / "RoboTwin" / "policy" / "EgoWAM"

TASKS = {
    "pick": "pick_diverse_bottles",
    "stack": "stack_bowls_three",
    "hang": "hanging_mug",
}
DIFFICULTY = {"Easy": "demo_clean", "Hard": "demo_randomized"}
SETTINGS = ["single", "cross"]
VARIANTS = ["hpt", "vae", "rae"]
EVAL_ROBOT = (
    "aloha-agilex"  # native-bimanual rollout robot, common to all 3 task robot-sets
)

_SUCCESS_RE = re.compile(r"Success rate:\D*(\d+)\s*/\s*(\d+)")


def _cell_key(setting: str, task: str, variant: str) -> str:
    return f"{setting}_{task}_{variant}"


def resolve_ckpt(setting, task, variant, args) -> str | None:
    if args.ckpt:  # single-cell override
        return args.ckpt
    key = _cell_key(setting, task, variant)
    if args.manifest:
        m = json.loads(Path(args.manifest).read_text())
        return m.get(key)
    if args.ckpt_root:
        cand = Path(args.ckpt_root) / key / "last.ckpt"
        return str(cand) if cand.exists() else None
    return None


def parse_output(text: str) -> dict:
    succ = total = None
    for m in _SUCCESS_RE.finditer(text):  # take the final tally
        succ, total = int(m.group(1)), int(m.group(2))
    ik_fail = text.count("EGOWAM_IK_FAIL:")
    return {
        "success": succ,
        "total": total,
        "success_pct": (100.0 * succ / total) if (succ is not None and total) else None,
        "ik_fail": ik_fail,
    }


def run_cell(setting, task, variant, difficulty, seed, ckpt, args) -> dict:
    task_name = TASKS[task]
    task_config = DIFFICULTY[difficulty]
    ckpt_setting = f"rt_{setting}_{task}_{variant}"
    ckpt_arg = (
        ckpt if ckpt else "<CKPT>"
    )  # placeholder so dry-run/no-ckpt cells still render
    cmd = [
        "bash",
        "eval.sh",
        task_name,
        task_config,
        ckpt_setting,
        ckpt_arg,
        EVAL_ROBOT,
        str(seed),
        str(args.gpu),
    ]
    cell = {
        "setting": setting,
        "task": task,
        "variant": variant,
        "difficulty": difficulty,
        "seed": seed,
        "ckpt": ckpt,
        "ckpt_setting": ckpt_setting,
        "cmd": " ".join(cmd),
    }
    if args.dry_run:
        cell["status"] = "dry-run"
        return cell
    if not ckpt or not Path(ckpt).exists():
        cell["status"] = "skip-no-ckpt"
        return cell

    proc = subprocess.run(
        cmd, cwd=str(EGOWAM_DIR), capture_output=True, text=True, timeout=args.timeout
    )
    tail = (proc.stdout or "")[-4000:]
    cell.update(parse_output(proc.stdout or ""))
    cell["returncode"] = proc.returncode
    cell["status"] = "ok" if proc.returncode == 0 else "error"
    if proc.returncode != 0:
        cell["stderr_tail"] = (proc.stderr or "")[-2000:]
    if args.verbose:
        print(tail)
    return cell


def _fmt_pct(v):
    return f"{v:5.1f}" if isinstance(v, (int, float)) else "  -  "


def print_tables(results: list[dict], tasks, difficulties):
    from collections import defaultdict

    idx = defaultdict(list)
    for r in results:
        idx[(r["task"], r["setting"], r["variant"], r["difficulty"])].append(r)

    def best_pct(task, setting, variant, diff):
        cells = idx.get((task, setting, variant, diff), [])
        pcts = [c["success_pct"] for c in cells if c.get("success_pct") is not None]
        return sum(pcts) / len(pcts) if pcts else None  # mean over seeds

    print(
        "\n================ Table 1 — arx-x5 sim success % (mean over seeds) ================"
    )
    for task in tasks:
        print(f"\n[{task}]")
        for diff in difficulties:
            print(f"  {diff:5s} | " + " | ".join(f"{v:>5s}" for v in VARIANTS))
            for setting in SETTINGS:
                row = " | ".join(
                    _fmt_pct(best_pct(task, setting, v, diff)) for v in VARIANTS
                )
                print(f"  {setting:>6s}{'':0s}  {row}")

    print(
        "\n================ Table 2 — transfer: success(cross) - success(single) ============"
    )
    for task in tasks:
        print(f"\n[{task}]")
        for diff in difficulties:
            deltas = []
            for v in VARIANTS:
                c = best_pct(task, "cross", v, diff)
                s = best_pct(task, "single", v, diff)
                deltas.append(
                    f"{c - s:+5.1f}" if (c is not None and s is not None) else "  -  "
                )
            print(f"  {diff:5s} | " + " | ".join(f"{v:>5s}" for v in VARIANTS))
            print(f"        {' | '.join(deltas)}")


def preflight(dry_run: bool):
    missing = []
    for mod in ("sapien", "mplib"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing and not dry_run:
        raise SystemExit(
            f"[run_sim_eval] Missing sim deps {missing}. Closed-loop eval needs RoboTwin's SAPIEN "
            "stack in the same env as egowam/torch. Use --dry-run to print the matrix, or run "
            "under a combined env (plan §5 env note)."
        )
    if missing:
        print(f"[run_sim_eval] (dry-run) note: sim deps not importable here: {missing}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--settings", nargs="+", default=SETTINGS, choices=SETTINGS)
    p.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    p.add_argument(
        "--difficulties", nargs="+", default=list(DIFFICULTY), choices=list(DIFFICULTY)
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument(
        "--ckpt",
        default=None,
        help="single ckpt for a one-cell smoke (overrides resolution)",
    )
    p.add_argument(
        "--ckpt-root", default=None, help="DIR/<setting>_<task>_<variant>/last.ckpt"
    )
    p.add_argument(
        "--manifest", default=None, help="JSON: {<setting>_<task>_<variant>: ckpt_path}"
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--timeout", type=int, default=60 * 60 * 6, help="per-cell seconds")
    p.add_argument("--out", default=None, help="write raw results JSON here")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    preflight(args.dry_run)
    if not EGOWAM_DIR.exists():
        raise SystemExit(f"[run_sim_eval] missing EgoWAM policy dir: {EGOWAM_DIR}")

    results = []
    for task in args.tasks:
        for setting in args.settings:
            for variant in args.variants:
                ckpt = resolve_ckpt(setting, task, variant, args)
                for difficulty in args.difficulties:
                    for seed in args.seeds:
                        cell = run_cell(
                            setting, task, variant, difficulty, seed, ckpt, args
                        )
                        results.append(cell)
                        tag = f"{setting}/{task}/{variant}/{difficulty}/seed{seed}"
                        print(
                            f"[{cell['status']:>12s}] {tag}"
                            + (
                                f"  {cell.get('success')}/{cell.get('total')}"
                                f"  ik_fail={cell.get('ik_fail')}"
                                if cell.get("total")
                                else ""
                            )
                        )

    out = args.out or str(REPO_ROOT / "logs" / "robotwin_sim_eval_results.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2))
    print(f"\n[run_sim_eval] raw results -> {out}")
    print_tables(results, args.tasks, args.difficulties)


if __name__ == "__main__":
    main()
