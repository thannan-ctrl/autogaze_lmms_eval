"""Merge the full stages of the shards of one run_ext.sbatch run (--shard I/N, one tag per shard) into one summary.

Usage: python summarize_shards.py <runs/<tag_s0>> <runs/<tag_s1>> ... > <merged>.md

Checks before merging: every shard was run with the same model, task, arm set, frames and arms; the shards are
0..N-1 of the same N, each exactly once; no question appears in two shards; and per (sub)task the merged doc ids are
0..max with no gap (the shards together cover the task). Then the arms are summarized as summarize_suite.py does, over
the union of the shards. Accuracy is left out ("—") for tasks whose samples carry no score.pred (as headline_table.py).
Status lines go to stderr. Stdlib only, so it runs in either env.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import headline_table as ht  # noqa: E402
import summarize_suite as ss  # noqa: E402

SAME = ("model", "task", "arms", "nvf", "modes")


def read_preflight(suite):
    """The key=value pairs of preflight.txt's first two lines (run settings and shard)."""
    path = os.path.join(suite, "preflight.txt")
    if not os.path.exists(path):
        sys.exit(f"{suite}: no preflight.txt")
    head = open(path).read().splitlines()[:2]
    kv = dict(re.findall(r"\b([a-z_]+)=(\S*)", head[0]))
    m = re.search(r"modes=(.*?) smoke_n=", head[0])  # modes is a space-separated list
    kv["modes"] = m.group(1) if m else ""
    kv.update(re.findall(r"\b([a-z_]+)=(\S*)", head[1]) if len(head) > 1 and head[1].startswith("shard=") else [])
    return kv


def check_shards(suites):
    """Return (settings, modes) after checking the suites are shards 0..N-1 of one run."""
    infos = [read_preflight(s) for s in suites]
    for s, kv in zip(suites, infos):
        if kv.get("shard", "none") == "none":
            sys.exit(f"{s}: not a shard (preflight has no shard=I/N)")
        for k in SAME:
            if kv.get(k) != infos[0].get(k):
                sys.exit(f"{s}: {k}={kv.get(k)!r} differs from {suites[0]}: {infos[0].get(k)!r}")
    shards = [tuple(int(x) for x in kv["shard"].split("/")) for kv in infos]
    n = shards[0][1]
    if any(sn != n for _, sn in shards):
        sys.exit(f"shards of different N: {[kv['shard'] for kv in infos]}")
    got = sorted(si for si, _ in shards)
    if got != list(range(n)):
        sys.exit(f"need shards 0..{n - 1} exactly once, got {got}")
    return infos[0], infos[0]["modes"].split()


def merge_arm(suites, mode):
    """Union of one arm's samples, timing and client rows over the shards' full stages (None parts stay None)."""
    samples, timing, client = {}, {}, {}
    no_timing = no_client = False
    for i, suite in enumerate(suites):
        arm_dir = os.path.join(suite, "full", mode)
        s = ht.load_samples(arm_dir, mode)
        if s is None:
            sys.exit(f"{mode}: no samples in {arm_dir}")
        t, c = ht.load_timing(arm_dir, mode), ht.load_client(arm_dir, mode)
        dup = set(s) & set(samples)
        if dup:
            sys.exit(f"{mode}: {len(dup)} questions in more than one shard, e.g. {sorted(dup)[:5]} (again in {suite})")
        samples.update(s)
        if t is None:
            no_timing = True
        else:  # keep each shard's request order, shards in suite order
            timing.update({k: dict(r, request_idx=(i, r["request_idx"])) for k, r in t.items()})
        if c is None:
            no_client = True
        else:
            client.update(c)
    by_task = {}
    for name, doc_id in samples:
        by_task.setdefault(name, set()).add(doc_id)
    for name, ids in sorted(by_task.items()):
        gaps = sorted(set(range(max(ids) + 1)) - ids)
        if gaps:
            sys.exit(f"{mode}: {name} is missing {len(gaps)} doc ids, e.g. {gaps[:5]}; the shards don't cover the task")
    return samples, None if no_timing else timing, None if no_client else client


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("suites", nargs="+", help="runs/<tag> of every shard")
    args = ap.parse_args()

    info, modes = check_shards(args.suites)
    tags = [os.path.basename(os.path.normpath(s)) for s in args.suites]
    print(f"# {info['task']}, {info['model']}, arms {info['arms']}: {len(tags)} shards merged\n")
    print("Shards: " + ", ".join(f"`{t}`" for t in tags) + "\n")
    for suite in args.suites:
        print(f"<details><summary>preflight.txt ({os.path.basename(os.path.normpath(suite))})</summary>\n\n```\n"
              + open(os.path.join(suite, "preflight.txt")).read().rstrip() + "\n```\n</details>\n")
    res = {}
    for mode in modes:
        samples, timing, client = merge_arm(args.suites, mode)
        res[mode] = ht.summarize(mode, samples, timing, client)
    print(f"## full, merged: {info['model']}, {info['nvf']} frames, lmms-eval `{info['task']}`\n")
    ss.print_table(res)
    print()
    for mode, d in res.items():
        print(f"[status] full {mode}: {ss.status_line(d)}", file=sys.stderr)


if __name__ == "__main__":
    main()
