import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.nn import rms_norm, apply_rope, window_idxs_general
from tinyinfer.kernels import sparse_attn_dispatch
from tinyinfer.quant import dequant_fp8_block

torch.manual_seed(0)
eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=list(range(19)), gpu_layers=19, head_on_cpu=True)
eng.enable_prefetch = False
prompt = list(range(8))
with torch.inference_mode():
    logits = eng.forward(torch.tensor([prompt]), 0)
    nid = int(logits[0, -1].argmax())
    for p in range(8, 12):
        eng.forward(torch.tensor([[nid]]), p)
    torch.cuda.synchronize()

    def ev(fn, n=20):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(n): fn()
        e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1) / n

    for layer in (0, 2, 4):
        a = eng.block(layer).attn
        x = torch.randn(1, 1, 5120, dtype=torch.bfloat16, device="cuda") * 0.02
        print(f"--- L{layer} ({a.__class__.__name__}, ratio={getattr(a,'ratio',0)}, source={getattr(a,'is_kv_source',None)}) ---")
        print("  full attn      ", round(ev(lambda: a.forward(x, 6)), 2), "ms")
        print("  wq_a           ", round(ev(lambda: a.wq_a.forward(x)), 3), "ms")
        qr = a.wq_a.forward(x)
        print("  wq_b           ", round(ev(lambda: a.wq_b.forward(qr)), 3), "ms")
        print("  wkv            ", round(ev(lambda: a.wkv.forward(x)), 3), "ms")
        if getattr(a, "compressor", None) is not None:
            print("  compressor     ", round(ev(lambda: a.compressor.forward(x, 6)), 2), "ms")
        if getattr(a, "indexer", None) is not None:
            print("  indexer(full)  ", round(ev(lambda: a.indexer.forward(x, qr, None, 6, 2)), 2), "ms")
        # sparse attn with the shared compress set
        from tinyinfer.compressor import shared_attn
        print("  sparse_attn    ", round(ev(lambda: sparse_attn_dispatch(
            torch.randn(1,1,64,512,dtype=torch.bfloat16,device='cuda'),
            torch.randn(1,130,512,dtype=torch.bfloat16,device='cuda'),
            a.sink, torch.randint(0,129,(1,1,128),device='cuda',dtype=torch.int32), a.scale)), 3), "ms")
