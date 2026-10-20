"""Summarize an EgoSchema run of run_egoschema_headline.sh in the layout of the
"Headline result" table of AutoGaze/Codec_Selector_Feasibility.md.

Usage: python headline_table.py <runs/<tag>> [mode ...]

Accuracy comes from lmms-eval's logged samples (score.pred == score.ground_truth,
as egoschema_aggregate_score computes it). Timing comes from the server's
per-request log (timing_<mode>.jsonl), joined to the samples on meta.doc_id.
Stdlib only, so it runs in either env.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

TASK = "egoschema_subset"
MODES = ["codec_nvdec", "codec", "dense", "autogaze"]
CODEC_MODES = {"codec", "codec_nvdec", "codec_geo", "codec_ord"}

# Codec_Selector_Feasibility.md, Headline result (AutoGaze harness, commit cfa11e0,
# job 2383304): accuracy %, tokens, then seconds.
HEADLINE = {
    "codec_nvdec": dict(acc=67.0, tokens=2783, preproc=0.86, selector=0.09, selector_total=0.84, vit=0.11, llm=0.35, e2e=1.37),
    "codec": dict(acc=66.0, tokens=2783, preproc=1.01, selector=0.26, selector_total=1.00, vit=0.11, llm=0.36, e2e=1.54),
    "dense": dict(acc=64.6, tokens=24690, preproc=0.47, selector=None, selector_total=None, vit=2.91, llm=0.69, e2e=4.11),
    "autogaze": dict(acc=66.8, tokens=1837, preproc=4.93, selector=4.07, selector_total=None, vit=0.41, llm=0.42, e2e=5.81),
}


def load_samples(run_dir, mode):
    paths = glob.glob(os.path.join(run_dir, f"lmms_{mode}", "**", f"*samples_{TASK}.jsonl"), recursive=True)
    if not paths:
        return None
    if len(paths) > 1:
        sys.exit(f"{mode}: more than one samples file, expected one per run: {paths}")
    samples = {}
    for line in open(paths[0]):
        s = json.loads(line)
        if s["doc_id"] in samples:
            sys.exit(f"{mode}: duplicate doc_id {s['doc_id']} in {paths[0]}")
        samples[s["doc_id"]] = s
    return samples


def load_timing(run_dir, mode):
    path = os.path.join(run_dir, f"timing_{mode}.jsonl")
    if not os.path.exists(path):
        return None
    rows = {}
    for line in open(path):
        r = json.loads(line)
        meta = r.get("meta") or {}
        if meta.get("task") != TASK:
            sys.exit(f"{mode}: timing row {r['request_idx']} is for task {meta.get('task')!r}, not {TASK}")
        if r["mode"] != mode:
            sys.exit(f"{mode}: timing row {r['request_idx']} was served in mode {r['mode']!r}")
        if meta["doc_id"] in rows:
            sys.exit(f"{mode}: duplicate doc_id {meta['doc_id']} in {path} (server restarted into the same log?)")
        rows[meta["doc_id"]] = r
    return rows


def reply_of(sample):
    resp = sample.get("filtered_resps")
    while isinstance(resp, list):
        resp = resp[0] if resp else ""
    return resp or ""


def has_letter(reply):
    """True if lmms-eval's parse_multi_choice_response finds an index-style
    candidate ("(A)", "A ", "A."). Without one it matches option text (replies
    over 5 words) or picks a random letter."""
    r = " " + reply.strip(",.!?;:'") + " "
    return bool(re.search(r"\([A-E]\)|[A-E] |[A-E]\.", r))


def summarize(mode, samples, timing):
    n = len(samples)
    correct = sum(str(s["score"]["pred"]) == str(s["score"]["ground_truth"]) for s in samples.values())
    replies = [reply_of(s) for s in samples.values()]
    out = {
        "n": n,
        "acc": 100.0 * correct / n,
        "bare_letter": sum(bool(re.fullmatch(r"\s*\(?[A-E]\)?[.:]?\s*", r)) for r in replies),
        "no_letter": sum(not has_letter(r) for r in replies),
    }
    if timing is None:
        return out
    missing = sorted(set(samples) - set(timing))
    extra = sorted(set(timing) - set(samples))
    if missing or extra:
        sys.exit(f"{mode}: samples and timing rows don't match (missing timing for {missing[:5]}…, timing without sample {extra[:5]}…)")
    rows = [timing[d] for d in sorted(timing, key=lambda d: timing[d]["request_idx"])]
    mean = lambda k: statistics.fmean(r.get(k) or 0.0 for r in rows)
    out.update(
        tokens=mean("num_tokens"),
        preproc=mean("preproc_ms") / 1000,
        vit=mean("vit_ms") / 1000,
        llm=mean("llm_ms") / 1000,
        e2e=mean("e2e_ms") / 1000,
    )
    if mode in CODEC_MODES:
        out.update(selector=mean("selector_wait_ms") / 1000, selector_total=mean("selector_wall_ms") / 1000)
    elif mode.startswith("autogaze"):
        out.update(selector=mean("autogaze_model_ms") / 1000, selector_total=None)
    else:
        out.update(selector=None, selector_total=None)
    # Prefill split by whether this exact prompt length was already seen earlier
    # in the run (AutoGaze/scripts/print_table.py): first-seen lengths pay cuDNN
    # plan building inside the timed prefill.
    seen, steady, first = set(), [], []
    for r in rows:
        (steady if r["num_tokens"] in seen else first).append(r["llm_prefill_ms"])
        seen.add(r["num_tokens"])
    out["prefill_steady"] = (statistics.median(steady) / 1000, len(steady)) if steady else None
    out["prefill_first"] = (statistics.median(first) / 1000, len(first)) if first else None
    return out


def fmt_selector(d):
    if d.get("selector") is None:
        return "—"
    if d.get("selector_total") is None:
        return f"{d['selector']:.2f} s"
    return f"{d['selector']:.2f} s ({d['selector_total']:.2f} s total)"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("modes", nargs="*", default=MODES)
    args = ap.parse_args()

    results = {}
    for mode in args.modes:
        samples = load_samples(args.run_dir, mode)
        if samples is None:
            print(f"[{mode}] no {TASK} samples file under {args.run_dir}/lmms_{mode}, skipped", file=sys.stderr)
            continue
        results[mode] = summarize(mode, samples, load_timing(args.run_dir, mode))
    if not results:
        sys.exit("nothing to summarize")

    info = os.path.join(args.run_dir, "run_info.txt")
    if os.path.exists(info):
        print("```\n" + open(info).read().rstrip() + "\n```\n")

    print("**This run** (lmms-eval `egoschema_subset` through the bridge). Mean seconds per question.\n")
    print("| Mode | N | Accuracy | Tokens | Preproc (incl. selector) | of which selector | ViT | LLM | **E2E** |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for mode, d in results.items():
        if "e2e" in d:
            print(f"| {mode} | {d['n']} | {d['acc']:.1f}% | {d['tokens']:,.0f} | {d['preproc']:.2f} s | {fmt_selector(d)} "
                  f"| {d['vit']:.2f} s | {d['llm']:.2f} s | **{d['e2e']:.2f} s** |")
        else:
            print(f"| {mode} | {d['n']} | {d['acc']:.1f}% | (no timing log) | | | | | |")

    print("\n**Headline** (AutoGaze harness, `Codec_Selector_Feasibility.md`), and the difference (this run − headline).\n")
    print("| Mode | Accuracy | Tokens | Preproc | of which selector | ViT | LLM | **E2E** | ΔAcc (pts) | ΔE2E |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for mode, d in results.items():
        h = HEADLINE.get(mode)
        if h is None:
            continue
        d_e2e = f"{d['e2e'] - h['e2e']:+.2f} s" if "e2e" in d else ""
        print(f"| {mode} | {h['acc']:.1f}% | {h['tokens']:,} | {h['preproc']:.2f} s | {fmt_selector(h)} | {h['vit']:.2f} s "
              f"| {h['llm']:.2f} s | **{h['e2e']:.2f} s** | {d['acc'] - h['acc']:+.1f} | {d_e2e} |")

    print("\n**Replies and prefill.** *No letter*: replies where lmms-eval's parser finds no \"(A)\", \"A \" or \"A.\" "
          "and falls back to option-text matching or a random letter.\n")
    print("| Mode | Bare letter | No letter | Prefill median, repeated length | Prefill median, first-seen length |")
    print("|---|--:|--:|--:|--:|")
    for mode, d in results.items():
        cell = lambda p: f"{p[0]:.3f} s (n={p[1]})" if p else "—"
        print(f"| {mode} | {d['bare_letter']}/{d['n']} | {d['no_letter']}/{d['n']} "
              f"| {cell(d.get('prefill_steady'))} | {cell(d.get('prefill_first'))} |")


if __name__ == "__main__":
    main()
