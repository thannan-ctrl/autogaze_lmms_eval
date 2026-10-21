"""Summarize an EgoSchema run of run_egoschema_headline.sh in the layout of the
"Headline result" table of AutoGaze/Codec_Selector_Feasibility.md.

Usage: python headline_table.py <runs/<tag>> [mode ...]

Accuracy comes from lmms-eval's logged samples (score.pred == score.ground_truth,
as egoschema_aggregate_score computes it). Timing comes from the server's
per-request log (timing_<mode>.jsonl), joined to the samples on meta.doc_id.

End-to-end cross-check: the reported e2e_ms comes from AutoGaze's (or
model_compat's) own timers. The bridge measures the same questions twice with
its own clocks: bridge_server_ms (server.py, around each answer) and
client_roundtrip_ms (client.py inside lmms-eval, around each HTTP request,
client_timing_<mode>.jsonl). A reported E2E well below the bridge's numbers
means the repo's timer leaves work out.
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
CODEC_MODES = {"codec", "codec_nvdec", "codec_geo", "codec_ord", "codec_mv", "codec_mv_hybrid", "codec_mv_keep"}

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


def load_client(run_dir, mode):
    path = os.path.join(run_dir, f"client_timing_{mode}.jsonl")
    if not os.path.exists(path):
        return None
    rows = {}
    for line in open(path):
        r = json.loads(line)
        if r["meta"].get("task") != TASK:
            continue
        if r["meta"]["doc_id"] in rows:
            sys.exit(f"{mode}: duplicate doc_id {r['meta']['doc_id']} in {path}")
        rows[r["meta"]["doc_id"]] = r["client_roundtrip_ms"]
    return rows


def pctl(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


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


def summarize(mode, samples, timing, client=None):
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

    # End-to-end cross-check against the bridge's own clocks.
    out["parts"] = out["preproc"] + out["vit"] + out["llm"]
    if all(r.get("bridge_server_ms") is not None for r in rows):
        # Steps the bridge runs outside the repo's timer on purpose (model_compat's
        # untimed video pre-read and per-question cache cleanup) are timed by the
        # bridge and taken out of the gap; anything else still counts.
        untimed = [(r.get("preread_ms") or 0.0) + (r.get("cleanup_ms") or 0.0) for r in rows]
        gap = [r["bridge_server_ms"] - r["e2e_ms"] - u for r, u in zip(rows, untimed)]
        out.update(server=mean("bridge_server_ms") / 1000, untimed=statistics.fmean(untimed) / 1000,
                   gap_mean=statistics.fmean(gap) / 1000, gap_p95=pctl(gap, 0.95) / 1000)
    if client is not None:
        missing = sorted(set(timing) - set(client))
        if missing:
            sys.exit(f"{mode}: client timing log lacks {len(missing)} questions, e.g. doc_id {missing[:5]}")
        out["client"] = statistics.fmean(client[d] for d in timing) / 1000
    return out


def fmt_s(x):
    return "—" if x is None else f"{x:.2f} s"


def e2e_check(d):
    """ok when the bridge's server clock agrees with the reported E2E within
    max(5%, 50 ms) on average; otherwise how far apart they are."""
    if d.get("server") is None:
        return "no bridge clock in this run"
    tol = max(0.05 * d["e2e"], 0.05)
    if abs(d["gap_mean"]) <= tol:
        return "ok"
    return f"**bridge clock {100 * d['gap_mean'] / d['e2e']:+.0f}% vs reported E2E**"


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
        results[mode] = summarize(mode, samples, load_timing(args.run_dir, mode), load_client(args.run_dir, mode))
    if not results:
        sys.exit("nothing to summarize")

    info = os.path.join(args.run_dir, "run_info.txt")
    model = "nvila"
    if os.path.exists(info):
        text = open(info).read()
        print("```\n" + text.rstrip() + "\n```\n")
        m = re.search(r"^model=(\S+)", text, re.M)
        model = m.group(1) if m else model

    print("**This run** (lmms-eval `egoschema_subset` through the bridge). Mean seconds per question.\n")
    print("| Mode | N | Accuracy | Tokens | Preproc (incl. selector) | of which selector | ViT | LLM | **E2E** |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for mode, d in results.items():
        if "e2e" in d:
            print(f"| {mode} | {d['n']} | {d['acc']:.1f}% | {d['tokens']:,.0f} | {d['preproc']:.2f} s | {fmt_selector(d)} "
                  f"| {d['vit']:.2f} s | {d['llm']:.2f} s | **{d['e2e']:.2f} s** |")
        else:
            print(f"| {mode} | {d['n']} | {d['acc']:.1f}% | (no timing log) | | | | | |")

    print("\n**End-to-end latency cross-check.** Mean seconds per question. *Reported*: `e2e_ms` from the "
          "repo's own timers (the E2E column above). *Sum of parts*: preproc + ViT + LLM as reported. "
          "*Bridge server*: the bridge server's clock around each answer (includes untimed file reads). "
          "*Client*: lmms-eval's side of each HTTP request, so also JSON and HTTP overhead. "
          "*Untimed by design*: the video pre-read and cache cleanup that model_compat leaves out of its E2E "
          "(as its own runner does), timed by the bridge. "
          "*Gap*: bridge server − reported − untimed by design, per question.\n")
    print("| Mode | Reported E2E | Sum of parts | Bridge server | Client round trip | Untimed by design | Gap mean | Gap p95 | Check |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|---|")
    for mode, d in results.items():
        if "e2e" not in d:
            continue
        print(f"| {mode} | {d['e2e']:.2f} s | {d['parts']:.2f} s | {fmt_s(d.get('server'))} | {fmt_s(d.get('client'))} "
              f"| {fmt_s(d.get('untimed'))} | {fmt_s(d.get('gap_mean'))} | {fmt_s(d.get('gap_p95'))} | {e2e_check(d)} |")

    if model == "nvila":
        print("\n**Headline** (AutoGaze harness, `Codec_Selector_Feasibility.md`), and the difference (this run − headline).\n")
        print("| Mode | Accuracy | Tokens | Preproc | of which selector | ViT | LLM | **E2E** | ΔAcc (pts) | ΔE2E |")
        print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    else:
        print(f"\n(model={model}: the NVILA headline comparison does not apply; see AutoGaze/RESULTS_LATENCY_ACCURACY.md "
              "on branch pareto/main for the H100 model_compat numbers.)")
    for mode, d in results.items():
        h = HEADLINE.get(mode)
        if h is None or model != "nvila":
            continue
        d_e2e = f"{d['e2e'] - h['e2e']:+.2f} s" if "e2e" in d else ""
        print(f"| {mode} | {h['acc']:.1f}% | {h['tokens']:,} | {h['preproc']:.2f} s | {fmt_selector(h)} | {h['vit']:.2f} s "
              f"| {h['llm']:.2f} s | **{h['e2e']:.2f} s** | {d['acc'] - h['acc']:+.1f} | {d_e2e} |")

    # Pass/fail per mode on stderr, without numbers, for whoever runs the
    # launcher (the numbers stay in the results file).
    for mode in args.modes:
        d = results.get(mode)
        if d is None:
            print(f"[status] {mode}: MISSING (no samples)", file=sys.stderr)
            continue
        checks = ["timing ok" if "e2e" in d else "NO TIMING LOG",
                  "client log ok" if d.get("client") is not None else "NO CLIENT TIMING LOG",
                  "e2e cross-check ok" if e2e_check(d) == "ok" else "E2E CROSS-CHECK MISMATCH",
                  "replies ok" if d["no_letter"] <= d["n"] / 2 else "MOST REPLIES HAVE NO LETTER"]
        print(f"[status] {mode}: " + ", ".join(checks), file=sys.stderr)

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
