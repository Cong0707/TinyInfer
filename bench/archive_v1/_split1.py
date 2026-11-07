
import os, re, shutil, pathlib

root = pathlib.Path(r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer")
DEEPSEEK = ["model.py","moe.py","moe_fused.py","moe_gemv.py","mtp.py","engram.py",
            "compressor.py","hc.py","hc_fused.py","attention.py","compressed_attention.py",
            "attn_fused.py","deepseek_kernels.py","gemv_fp4.py","gemv_fp4_marlin.py",
            "gemv_fp4_native.py","vllm_ops.py"]
QWEN = ["qwen_model.py","qwen_layer.py","qwen_norm.py","qwen_kernels.py","qwen_elt.py",
        "gdn.py","gdn_fused.py","gdn_patch.py"]

for d in ("deepseek","qwen"):
    (root/d).mkdir(exist_ok=True)

moved = {}
for f in DEEPSEEK:
    src = root/f
    if src.exists():
        dst = root/"deepseek"/f
        shutil.move(str(src), str(dst))
        moved[f[:-3]] = "deepseek"
for f in QWEN:
    src = root/f
    if src.exists():
        dst = root/"qwen"/f
        shutil.move(str(src), str(dst))
        moved[f[:-3]] = "qwen"

(root/"deepseek"/"__init__.py").write_text(
    '"""DeepSeek-V4.1 specific layers: MLA + compressor + engram + hyper-connections + MoE."""\n',
    encoding="utf-8")
(root/"qwen"/"__init__.py").write_text(
    '"""Qwen3.5 specific layers: GatedDeltaNet hybrid attention."""\n', encoding="utf-8")

# 记录新位置供 import 重写用
import json
(root/"_moved.json").write_text(json.dumps(moved, indent=1), encoding="utf-8")
print("moved:", len(moved))
print("deepseek:", sum(1 for v in moved.values() if v=="deepseek"))
print("qwen:", sum(1 for v in moved.values() if v=="qwen"))
