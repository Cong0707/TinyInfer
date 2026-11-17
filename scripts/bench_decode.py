"""单进程连续解码基准：真实 prompt、递增位置、逐步 finite 校验。

仅初始化一次引擎。预填充与预热单独记录，不混入 decode 吞吐。
测量期不 reset，不反复输入同一个 token，不修改任何引擎调优参数。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def measure_decode(eng, ids, *, tokens, warmup, synchronize, emit):
    """可用小型假引擎测试的位置/采样契约；模型只由外层创建一次。"""
    import torch
    from tinyinfer.loading import _async_h2d_stats

    if tokens <= 0 or warmup < 0:
        raise ValueError("tokens must be positive and warmup must be nonnegative")
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError("benchmark needs one nonempty prompt")
    rows = []

    def step(inputs, position, phase):
        synchronize()
        before = _async_h2d_stats()
        copies = getattr(eng, "_flat_applies", 0)
        started = time.perf_counter()
        logits = eng.forward(inputs, position)
        next_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
        synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        after = _async_h2d_stats()
        finite = bool(torch.isfinite(logits).all())
        row = {
            "phase": phase, "position": position, "input_tokens": inputs.shape[1],
            "ms": elapsed, "finite": finite, "next_token": int(next_ids.item()),
            "flat_copies": getattr(eng, "_flat_applies", 0) - copies,
            "h2d_bytes": after["ok_bytes"] - before["ok_bytes"],
            "h2d_calls": after["ok_calls"] - before["ok_calls"],
            "h2d_failed": after["rc_fail"] - before["rc_fail"],
        }
        rows.append(row)
        emit(row)
        if not finite:
            raise FloatingPointError(f"nonfinite logits: {phase} at position {position}")
        return next_ids

    with torch.inference_mode():
        next_ids = step(ids, 0, "prefill")
        position = ids.shape[1]
        for phase, count in (("warmup", warmup), ("decode", tokens)):
            for _ in range(count):
                next_ids = step(next_ids, position, phase)
                position += 1
    times = [row["ms"] for row in rows if row["phase"] == "decode"]
    median = statistics.median(times)
    return {
        "finite": True, "n": len(times), "warmup_steps": warmup,
        "median_ms": median, "min_ms": min(times), "max_ms": max(times),
        "spread": (max(times) - min(times)) / median,
        "tok_per_s": len(times) * 1000 / sum(times),
        "median_tok_per_s": 1000 / median, "steps": rows,
    }


def bench(model_dir, *, tokens=16, warmup=6):
    import torch
    from transformers import AutoTokenizer
    from tinyinfer.engine import Engine

    # 正文续写负载，不假设 checkpoint 提供聊天模板。
    prompt = "推理引擎加载模型权重的过程如下：\n"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"].to("cuda")
    eng = Engine.from_hf(str(model_dir), device="cuda", tokenizer=tokenizer)
    try:
        result = measure_decode(
            eng, ids, tokens=tokens, warmup=warmup,
            synchronize=torch.cuda.synchronize,
            emit=lambda row: print("[step] " + json.dumps(row, ensure_ascii=False), flush=True),
        )
        layout = eng.hw_plan
        state = getattr(eng, "_flat_state", None)
        result.update({
            "model": str(model_dir), "prompt": prompt,
            "prompt_tokens": ids.shape[1], "resident": layout.resident,
            "slots": layout.slots, "stream_layers": len(eng._stream_layers()),
            "ring_buffers": len(state[1]) if isinstance(state, tuple) else None,
            "slab_layers": len(state[2]) if isinstance(state, tuple) and state[2] is not None else None,
            "overrides": {k: v for k, v in os.environ.items() if k.startswith("TI_")},
            "generated_text": tokenizer.decode([row["next_token"] for row in result["steps"]]),
        })
        return result
    finally:
        eng.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    result = bench(args.model, tokens=args.tokens)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("decode %.1f ms 中位数，%.1f~%.1f ms" % (
            result["median_ms"], result["min_ms"], result["max_ms"]))
        print("吞吐 %.4f tok/s，finite=%s" % (result["tok_per_s"], result["finite"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
