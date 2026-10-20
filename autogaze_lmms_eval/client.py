"""lmms-eval model plugin that forwards each request to the AutoGaze server.

Registered as `autogaze` through the `lmms_eval.models` entry point (see plugin.py):

    python -m lmms_eval --model autogaze \
        --model_args server_url=http://127.0.0.1:8765,mode=autogaze \
        --tasks videomme --batch_size 1

The server must run on the same filesystem, since requests carry the video's
path, not its bytes (codec modes need the original file anyway).
"""
import json
import logging
import urllib.error
import urllib.request
from typing import List, Tuple

from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms

eval_logger = logging.getLogger("lmms-eval")


class AutoGaze(lmms):
    is_simple = True

    def __init__(self, server_url: str = "http://127.0.0.1:8765", mode: str = None, timeout: float = 1800, **kwargs) -> None:
        super().__init__()
        if kwargs:
            eval_logger.warning(f"[autogaze] ignoring unknown model_args: {sorted(kwargs)}")
        self.server_url = server_url.rstrip("/")
        self.timeout = float(timeout)
        self.server_info = self._request("GET", "/health")
        eval_logger.info(f"[autogaze] connected to {self.server_url}: {self.server_info}")
        # Optional guard so results are never logged under the wrong mode.
        if mode is not None and mode != self.server_info["mode"]:
            raise ValueError(f"model_args mode={mode!r} but the server at {self.server_url} runs mode={self.server_info['mode']!r}")

    def _request(self, method: str, path: str, body: dict = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.server_url + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"AutoGaze server {path} failed ({e.code}): {e.read().decode(errors='replace')}") from None

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            if len(visuals) != 1 or not isinstance(visuals[0], str):
                raise ValueError(f"autogaze expects exactly one video path per question, got {visuals!r} ({task} doc {doc_id})")
            if gen_kwargs.get("temperature", 0) not in (0, 0.0, None):
                eval_logger.warning(f"[autogaze] temperature={gen_kwargs['temperature']} ignored; decoding is greedy unless do_sample is set")
            out = self._request("POST", "/generate", {
                "video_path": visuals[0],
                "prompt": contexts,
                "max_new_tokens": gen_kwargs.get("max_new_tokens", 16),
                "do_sample": bool(gen_kwargs.get("do_sample", False)),
                # Logged next to the server's per-request timing, so it can be
                # joined with lmms-eval's --log_samples output by (task, doc_id).
                "meta": {"task": task, "split": split, "doc_id": doc_id},
            })
            text = out["text"]
            for stop in gen_kwargs.get("until", None) or []:
                if stop:
                    text = text.split(stop)[0]
            res.append(text)
            self.cache_hook.add_partial("generate_until", (contexts, gen_kwargs), text)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("autogaze bridge supports generate_until tasks only")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("autogaze bridge supports generate_until tasks only")

