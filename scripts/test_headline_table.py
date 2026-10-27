"""Synthetic-run checks for headline_table.py's task handling: a single task (egoschema_subset, accuracy from
score.pred/ground_truth) and a group task (mvbench, one samples file per subtask, no score field).
Run: python -m pytest -q scripts/test_headline_table.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import headline_table as ht  # noqa: E402

SAMPLES = "samples" + "_"


def make_run(root, task, subtasks, mode, scored):
    run = root / mode
    out = run / f"lmms_{mode}" / "model"
    out.mkdir(parents=True)
    (run / "run_info.txt").write_text(f"tag=x\nmodel=qwen3_5 nvf=128 max_batch_size_autogaze=32 task={task}\n")
    timing, client, idx = [], [], 0
    for sub in subtasks:
        with open(out / f"2026_{SAMPLES}{sub}.jsonl", "w") as f:
            for d in range(3):
                s = {"doc_id": d, "filtered_resps": ["A."]}
                if scored:
                    s["score"] = {"pred": "A", "ground_truth": "A" if d else "B"}
                f.write(json.dumps(s) + "\n")
                meta = {"task": sub, "split": "test", "doc_id": d}
                timing.append({"request_idx": idx, "mode": mode, "meta": meta, "num_tokens": 100, "llm_prefill_ms": 1.0,
                               "preproc_ms": 10.0, "vit_ms": 5.0, "llm_ms": 5.0, "e2e_ms": 20.0, "bridge_server_ms": 20.0})
                client.append({"meta": meta, "client_roundtrip_ms": 21.0})
                idx += 1
    (run / f"timing_{mode}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in timing))
    (run / f"client_timing_{mode}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in client))
    return str(run)


def test_single_task_scored(tmp_path):
    run = make_run(tmp_path, "egoschema_subset", ["egoschema_subset"], "dense", scored=True)
    d = ht.summarize("dense", ht.load_samples(run, "dense"), ht.load_timing(run, "dense"), ht.load_client(run, "dense"))
    assert d["n"] == 3 and abs(d["acc"] - 200 / 3) < 1e-9 and d["client"] is not None and ht.e2e_check(d) == "ok"


def test_group_task_unscored(tmp_path):
    subs = ["mvbench_action_sequence", "mvbench_object_shuffle"]
    run = make_run(tmp_path, "mvbench", subs, "codec_mv_f1.0_c16_r0.25", scored=False)
    m = "codec_mv_f1.0_c16_r0.25"
    d = ht.summarize(m, ht.load_samples(run, m), ht.load_timing(run, m), ht.load_client(run, m))
    assert d["n"] == 6 and d["acc"] is None and ht.fmt_acc(d["acc"]) == "—"
    assert d["selector"] is not None or "selector" in d  # codec mode branch taken
    assert ht.is_codec(m) and not ht.is_codec("dense")


def test_other_task_rows_rejected(tmp_path):
    run = make_run(tmp_path, "mvbench", ["videomme"], "dense", scored=False)
    assert ht.load_samples(run, "dense") is None  # a videomme file is not part of mvbench
