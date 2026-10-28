"""Gate 1a of EXTENSION_SPEC.md: the codec selections of the extension branch (pool=max default) are bit-identical to
the reproduction's pinned sha. Two steps:

  dump     run one model_compat tree's selector (the bridge's v3mv path, no model weights, no GPU) on videos and write
           one JSON line per (video, mode): sampled frame indices, unit grid, kept units per temporal group, kept
           fraction, and a hash of the score grids.
  compare  check two dumps item by item; exit 1 on any difference.

Usage (server env, e.g. nemo-omni on x86):
  git -C AutoGaze archive repro-pin model_compat | tar -x -C <scratch>/repro_pin   # the pinned tree, no git state change
  python selection_identity.py dump --mc-root <scratch>/repro_pin --video-dir <dir> --n 20 --out pin.jsonl
  python selection_identity.py dump --mc-root <codec_reproduce>/AutoGaze --video-dir <dir> --n 20 --out ext.jsonl
  python selection_identity.py compare pin.jsonl ext.jsonl

The env is the bridge's (mc_backend.COMMON_ENV and MODEL_ENV, explicit MC_* exports win) except MC_QWEN_MAX_PIXELS,
which is --max-pixels (default: the extension's stock 25165824). Selection runs through selector_v2.score_grids_mp, the
same scoring as the server's overlapped path (MC_OVERLAP only reorders the work). Modes are bridge mode names.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the autogaze_lmms_eval package
from autogaze_lmms_eval.mc_backend import COMMON_ENV, MODEL_ENV, PATH_ENV, arm_for_mode  # noqa: E402

DEFAULT_MODES = ["codec_mv", "codec_mv_f1.0_c16_r0.25"]
QWEN_PATH = ("/home/scratch.thannan_wwfo/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/"
             "c202236235762e1c871ad0ccb60c8ee5ba337b9a")


def dump(args):
    os.environ["MC_QWEN_MAX_PIXELS"] = str(args.max_pixels)
    for k, v in {**COMMON_ENV, **MODEL_ENV[args.model]}.items():
        os.environ.setdefault(k, v)
    os.environ[PATH_ENV[args.model]] = args.model_path
    if os.environ["MC_SELECTOR"] != "v3mv":
        sys.exit(f"MC_SELECTOR={os.environ['MC_SELECTOR']}: this gate is for the v3mv selector")
    sys.path.insert(0, os.path.abspath(args.mc_root))
    import importlib

    import numpy as np
    from model_compat import codec_scores, runner, selection, selector_v2
    if not os.path.realpath(selection.__file__).startswith(os.path.realpath(args.mc_root)):
        sys.exit(f"model_compat imported from {selection.__file__}, not from --mc-root {args.mc_root}")

    adapter = importlib.import_module(f"model_compat.adapters.{args.model}").Adapter()
    adapter.num_frames = args.nvf
    if args.model == "qwen3_5":  # unit_grid needs only the video processor (Adapter.load's pixel-cap override)
        from transformers import AutoProcessor
        adapter.vp = AutoProcessor.from_pretrained(args.model_path).video_processor
        adapter.vp.size["longest_edge"] = args.max_pixels

    videos = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))[:args.n] if args.video_dir else []
    videos += args.videos
    if not videos:
        sys.exit("no videos (--video-dir and/or --videos)")
    env = " ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("MC_"))
    print(f"[dump] mc_root={args.mc_root} model={args.model} nvf={args.nvf} videos={len(videos)} modes={args.modes}\n"
          f"[dump] {env}", file=sys.stderr)
    with open(args.out, "w") as f:
        for path in videos:
            idxs = codec_scores.sample_frame_indices(path, args.nvf)
            w0, h0 = codec_scores.probe_wh(path)
            gh, gw = adapter.unit_grid((h0, w0))
            grids, _, _ = selector_v2.score_grids_mp(path, idxs, (gh, gw), args.cache_dir, method="mv")
            grids = np.ascontiguousarray(grids)
            for mode in args.modes:
                _, kw, chunk, _ = runner.parse_arm(arm_for_mode(mode))
                kept, frac = selection.select_from_pooled(grids, adapter.temporal_patch, chunk or adapter.chunk_frames, **kw)
                f.write(json.dumps({
                    "video": os.path.basename(path), "mode": mode, "frame_indices": [int(i) for i in idxs],
                    "grid": [gh, gw], "grids_sha256": hashlib.sha256(grids.tobytes()).hexdigest(),
                    "grids_dtype": str(grids.dtype), "kept": [sorted(int(u) for u in k) for k in kept],
                    "kept_fraction": float(frac)}) + "\n")
            print(f"[dump] {os.path.basename(path)}: grid {gh}x{gw}, {len(idxs)} frames", file=sys.stderr)
    if selector_v2._POOL is not None:  # its spawn workers would otherwise outlive us and hold stdout open
        selector_v2._POOL.shutdown(wait=True)


def compare(args):
    def load(p):
        rows = {}
        for line in open(p):
            r = json.loads(line)
            rows[(r["video"], r["mode"])] = r
        return rows
    a, b = load(args.a), load(args.b)
    if set(a) != set(b):
        print(f"FAIL item sets differ: only in {args.a}: {sorted(set(a) - set(b))[:5]}, "
              f"only in {args.b}: {sorted(set(b) - set(a))[:5]}")
        sys.exit(1)
    bad = 0
    for key in sorted(a):
        diff = [k for k in ("frame_indices", "grid", "kept", "grids_sha256", "kept_fraction") if a[key][k] != b[key][k]]
        if diff:
            bad += 1
            print(f"DIFF {key[0]} {key[1]}: {', '.join(diff)}")
    videos = {v for v, _ in a}
    modes = sorted({m for _, m in a})
    print(f"{'FAIL' if bad else 'PASS'}: {len(a) - bad}/{len(a)} (video, mode) selections identical "
          f"({len(videos)} videos, modes {' '.join(modes)})")
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--mc-root", required=True, help="directory holding the model_compat package to test")
    d.add_argument("--model", default="qwen3_5", choices=sorted(MODEL_ENV))
    d.add_argument("--model-path", default=QWEN_PATH, help="weights dir (only the processor config is read)")
    d.add_argument("--nvf", type=int, default=128)
    d.add_argument("--max-pixels", type=int, default=25165824, help="MC_QWEN_MAX_PIXELS (repro: 4294967296)")
    d.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    d.add_argument("--video-dir", help="take the first --n *.mp4 in sorted order")
    d.add_argument("--n", type=int, default=20)
    d.add_argument("--videos", nargs="*", default=[])
    d.add_argument("--cache-dir", default=None)
    d.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    for m in getattr(args, "modes", []):
        if arm_for_mode(m) is None:
            sys.exit(f"unknown mode {m}")
    dump(args) if args.cmd == "dump" else compare(args)


if __name__ == "__main__":
    main()
