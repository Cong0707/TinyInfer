from pathlib import Path
p = Path(r"E:\Develop\Deepseek\tinyinfer\CONTRIBUTING.md")
src = p.read_text(encoding="utf-8")

section = """
## Kernels: copy a proven implementation before writing one

Every kernel starts from an upstream implementation whenever one exists. vLLM's csrc/ and
llama.cpp's ggml/ are checked out under repositories/ for exactly this; read them before
opening an editor.

This is not a style preference. Measured on this machine (RTX 5060 Laptop, sm_120, fp4
experts N=16384 K=5120, 40 MB of weights, three warm-up rounds):

| implementation | time | bandwidth |
| --- | --- | --- |
| vLLM Marlin (thread_n=16 thread_k=4, 26 blocks) | 232-260 us | **151-169 GB/s** |
| tinyinfer, hand-written scalar decode | 657 us | 59 GB/s |
| tinyinfer, tilelang fallback | 1044 us | 40 GB/s |

Writing it from scratch cost five iterations (table lookup, ALU bit-assembly, SWAR, tensor
core with manual fragments, 16-byte vector loads) and the best of them reached 2.2x SLOWER
than the kernel that was already sitting in repositories/vllm/csrc/quantization/marlin/.
The 2.2x is not in the arithmetic; it is in the four-stage cp.async pipeline, the ldmatrix
fragment loads, and the k-major interleaved weight repack -- three things Marlin already
does.

So: vendor the upstream kernel, adapt its interface, keep its name in a comment with the
commit or version it came from, and only then consider a replacement. A from-scratch kernel
has to beat the vendored one on a measurement before it replaces it.
"""

if "Kernels: copy a proven implementation" not in src:
    p.write_text(src.rstrip() + chr(10) + section, encoding="utf-8")
    print("appended")
else:
    print("already present")