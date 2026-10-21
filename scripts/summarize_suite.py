"""Summarize one run_eval.sbatch suite (runs/<tag>/{smoke,full}) into SUMMARY.md.

Usage: python summarize_suite.py <runs/<tag>> > <runs/<tag>>/SUMMARY.md

Reads only what the launcher wrote: each stage's run_info.txt, lmms-eval samples,
timing_<mode>.jsonl and client_timing_<mode>.jsonl, through the same loaders as
headline_table.py (one parsing path for both files). No pass/fail on the numbers:
the reference values below are H100 model_compat results with model_compat's own
prompt and parser, shown for information only.
Stdlib only, so it runs in either env.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import headline_table as ht  # noqa: E402

# AutoGaze/RESULTS_LATENCY_ACCURACY.md (branch pareto/main, 9899fd6), EgoSchema
# detailed tables: H100 SXM, model_compat prompt/parser, max_new_tokens 8.
# (seconds per question, accuracy %, visual tokens)
REFERENCE = {
    ("qwen3_5", 128): {"dense": (1.14, 72.8, 11225), "codec_mv_hybrid": (0.81, 69.4, 3847),
                       "codec_mv": (0.70, 54.6, 872), "codec_mv_keep": (0.81, 68.6, 3285)},
    ("nemotron_omni", 64): {"dense": (0.66, 64.0, 5216), "codec_mv_hybrid": (0.52, 59.8, 1786),
                            "codec_mv": (0.51, 47.4, 405), "codec_mv_keep": (0.53, 59.0, 1525)},
}
STAGES = ("smoke", "full")


def read_info(stage_dir):
    path = os.path.join(stage_dir, "run_info.txt")
    if not os.path.exists(path):
        return None, {}
    text = open(path).read()
    kv = {}
    for line in text.splitlines():
        for part in line.split(" "):
            k, sep, v = part.partition("=")
            if sep and re.fullmatch(r"[a-z_]+", k):
                kv.setdefault(k, v)
    return text, kv


def stage_results(stage_dir, modes):
    out = {}
    for mode in modes:
        samples = ht.load_samples(stage_dir, mode)
        if samples is None:
            out[mode] = None
            continue
        out[mode] = ht.summarize(mode, samples, ht.load_timing(stage_dir, mode), ht.load_client(stage_dir, mode))
    return out


def status_line(d):
    if d is None:
        return "MISSING (no samples)"
    return ", ".join([
        "timing ok" if "e2e" in d else "NO TIMING LOG",
        "client log ok" if d.get("client") is not None else "NO CLIENT TIMING LOG",
        "e2e cross-check ok" if ht.e2e_check(d) == "ok" else "E2E CROSS-CHECK MISMATCH",
        "replies ok" if d["no_letter"] <= d["n"] / 2 else "MOST REPLIES HAVE NO LETTER",
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("suite_dir")
    args = ap.parse_args()

    tag = os.path.basename(os.path.normpath(args.suite_dir))
    print(f"# {tag}\n")
    pre = os.path.join(args.suite_dir, "preflight.txt")
    if os.path.exists(pre):
        print("## Preflight\n\n```\n" + open(pre).read().rstrip() + "\n```\n")

    for stage in STAGES:
        stage_dir = os.path.join(args.suite_dir, stage)
        text, info = read_info(stage_dir)
        if text is None:
            print(f"## {stage}\n\nnot run.\n")
            continue
        m = re.search(r"^modes=(.*?) gpus=", text, re.M)
        modes = m.group(1).split() if m else []
        res = stage_results(stage_dir, modes)
        model, nvf = info.get("model", "?"), int(info.get("nvf", "0") or 0)
        print(f"## {stage}: {model}, {nvf} frames, lmms-eval `{ht.TASK}`\n")
        print("```\n" + text.rstrip() + "\n```\n")
        print("| Mode | N | Accuracy | Tokens | Preproc (incl. selector) | of which selector | ViT | LLM | **E2E** "
              "| Bridge server | Client | E2E check | No letter |")
        print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|")
        for mode, d in res.items():
            if d is None:
                print(f"| {mode} | missing | | | | | | | | | | | |")
            elif "e2e" not in d:
                print(f"| {mode} | {d['n']} | {d['acc']:.1f}% | (no timing log) | | | | | | | | | {d['no_letter']} |")
            else:
                print(f"| {mode} | {d['n']} | {d['acc']:.1f}% | {d['tokens']:,.0f} | {d['preproc']:.2f} s "
                      f"| {ht.fmt_selector(d)} | {d['vit']:.2f} s | {d['llm']:.2f} s | **{d['e2e']:.2f} s** "
                      f"| {ht.fmt_s(d.get('server'))} | {ht.fmt_s(d.get('client'))} | {ht.e2e_check(d)} "
                      f"| {d['no_letter']}/{d['n']} |")
        ref = REFERENCE.get((model, nvf))
        if stage == "full" and ref:
            print("\n**Reference, for information only** (H100 SXM, model_compat's own prompt, parser and "
                  "max_new_tokens 8; `AutoGaze/RESULTS_LATENCY_ACCURACY.md`). Different GPU, stack and protocol, "
                  "so differences here are expected and are not a pass/fail.\n")
            print("| Mode | Reference E2E | Reference accuracy | Reference tokens | This run − reference (pts) |")
            print("|---|--:|--:|--:|--:|")
            for mode, d in res.items():
                if mode in ref:
                    s, acc, tok = ref[mode]
                    diff = f"{d['acc'] - acc:+.1f}" if d else "—"
                    print(f"| {mode} | {s:.2f} s | {acc:.1f}% | {tok:,} | {diff} |")
        print()
        for mode, d in res.items():
            print(f"[status] {stage} {mode}: {status_line(d)}", file=sys.stderr)


if __name__ == "__main__":
    main()
