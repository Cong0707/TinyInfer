"""Drive Qwen layer by layer through the engine, measuring real decode speed.

The engine owns residency: layers live on the card up to the plan's budget and stream in
for their turn beyond it. This loop is the minimal caller of that contract -- embed, run
each layer, head, sample -- so the number it reports is the engine's number.
"""
import sys, time, json
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import gdn_patch, qwen_norm
gdn_patch.install(); qwen_norm.install()
from tinyinfer.engine import Engine
from transformers import AutoTokenizer


def build(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    eng = Engine.from_hf(model_dir, device="cuda", allow_missing_layers=True)
    return eng, tok


def forward_one(eng, kv, tok, ids, start_pos, cap):
    """One step, delegating to the model's capturable step function."""
    # the model owns how a step runs; the engine owns where its layers live
    return eng.adapter.model.step(eng, kv, ids, torch.tensor([start_pos], device="cuda"))


def main(model_dir, n_new=16):
    eng, tok = build(model_dir)
    print(f"layers {eng.cfg.n_layers} | resident {eng.gpu_layers} | "
          f"stream {eng.cfg.n_layers - eng.gpu_layers}", flush=True)
    ids = tok("hi", return_tensors="pt").input_ids.to("cuda")
    kv = eng.adapter.make_kv(ids.shape[1] + n_new + 8)
    with torch.inference_mode():
        t0 = time.perf_counter()
        logits = forward_one(eng, kv, tok, ids, 0, 0)
        torch.cuda.synchronize()
        prefill = time.perf_counter() - t0
        print(f"prefill {ids.shape[1]} tok: {prefill:.2f}s | "
              f"first {int(logits[0, -1].argmax())}", flush=True)
        pos = ids.shape[1]
        cur = int(logits[0, -1].argmax())
        t0 = time.perf_counter()
        n = 0
        for _ in range(n_new):
            nxt = torch.tensor([[cur]], device="cuda")
            lg = forward_one(eng, kv, tok, nxt, pos, 0)
            cur = int(lg[0, -1].argmax())
            pos += 1
            n += 1
        torch.cuda.synchronize()
        dec = time.perf_counter() - t0
    print(f"decode {n} tok in {dec:.2f}s -> {n/dec:.3f} tok/s", flush=True)
    return n / dec


if __name__ == "__main__":
    md = sys.argv[1] if len(sys.argv) > 1 else r"E:\Models\Qwen3.5-0.8B"
    main(md, int(sys.argv[2]) if len(sys.argv) > 2 else 16)