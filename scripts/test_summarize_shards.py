"""Synthetic-shard checks for summarize_shards.py: shards of a scored single task and of a group task merge into one
summary, and wrong shard sets (missing shard, overlap, mismatched settings, gap) are refused.
Run: python -m pytest -q scripts/test_summarize_shards.py"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize_shards as sh  # noqa: E402
from test_headline_table import make_run  # noqa: E402

MODES = ["dense", "codec_mv_f1.0_c16_r0.25"]


def make_shard(root, i, n, task, subtasks, docs, arms="main", scored=True):
    suite = root / f"t_s{i}of{n}"
    (suite / "full").mkdir(parents=True)
    (suite / "preflight.txt").write_text(
        f"tag={suite.name} model=nvila task={task} arms={arms} nvf=16 modes={' '.join(MODES)} smoke_n=20 smoke_only=0\n"
        f"shard={i}/{n} offset={docs[0]} full_limit=none\nok   aarch64 node\n")
    for m in MODES:
        make_run(suite / "full", task, subtasks, m, scored=scored, docs=docs)
    return str(suite)


def test_merge_single_task(tmp_path, capsys):
    suites = [make_shard(tmp_path, i, 2, "egoschema_subset", ["egoschema_subset"], range(3 * i, 3 * i + 3)) for i in (1, 0)]
    info, modes = sh.check_shards(suites)
    assert modes == MODES and info["task"] == "egoschema_subset"
    samples, timing, client = sh.merge_arm(suites, "dense")
    assert len(samples) == len(timing) == len(client) == 6
    sh.sys.argv = ["summarize_shards.py", *suites]
    sh.main()
    out = capsys.readouterr()
    assert "| dense | 6 |" in out.out and "[status] full dense: timing ok" in out.err


def test_merge_group_task(tmp_path):
    subs = ["mvbench_action_sequence", "mvbench_object_shuffle"]
    suites = [make_shard(tmp_path, i, 2, "mvbench", subs, range(3 * i, 3 * i + 3), scored=False) for i in (0, 1)]
    samples, _, _ = sh.merge_arm(suites, MODES[1])
    assert sorted(samples) == sorted((s, d) for s in subs for d in range(6))


def test_missing_shard(tmp_path):
    suites = [make_shard(tmp_path, 0, 2, "egoschema_subset", ["egoschema_subset"], range(3))]
    with pytest.raises(SystemExit, match="exactly once"):
        sh.check_shards(suites)


def test_overlap(tmp_path):
    suites = [make_shard(tmp_path, i, 2, "egoschema_subset", ["egoschema_subset"], range(i, i + 3)) for i in (0, 1)]
    with pytest.raises(SystemExit, match="more than one shard"):
        sh.merge_arm(suites, "dense")


def test_gap(tmp_path):
    suites = [make_shard(tmp_path, i, 2, "egoschema_subset", ["egoschema_subset"], range(4 * i, 4 * i + 3)) for i in (0, 1)]
    with pytest.raises(SystemExit, match="missing 1 doc ids"):
        sh.merge_arm(suites, "dense")


def test_mismatched_settings(tmp_path):
    a = make_shard(tmp_path, 0, 2, "egoschema_subset", ["egoschema_subset"], range(3))
    b = make_shard(tmp_path, 1, 2, "egoschema_subset", ["egoschema_subset"], range(3, 6), arms="grid")
    with pytest.raises(SystemExit, match="arms="):
        sh.check_shards([a, b])
