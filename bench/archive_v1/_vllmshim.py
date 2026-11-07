"""vLLM 0.33 wants torch APIs that 2.9 removed. Shim them, then use vLLM's own Marlin."""
import sys, types

# shim the missing symbols torch._functorch no longer exports
import torch._functorch as _f
if not hasattr(_f, "draw_graph"):
    _f.draw_graph = lambda *a, **k: None
_f.get_aot_graph_name = lambda *a, **k: "graph"
_f.get_graph_being_compiled = lambda *a, **k: "graph"

import torch._inductor.codegen.wrapper as _w
_orig = _w.PythonWrapperCodegen
class _W(_orig):
    pass
sys.modules["torch._inductor.codegen.wrapper"].PythonWrapperCodegen = _W

print("shims installed")
try:
    import vllm._custom_ops as co
    print("custom_ops ok")
    names = [x for x in dir(co) if "marlin" in x.lower()]
    print("marlin ops:", names[:8])
except Exception as e:
    print("failed:", type(e).__name__, str(e)[:300])
