
import torch
import tinyinfer._C_stable_libtorch  # noqa
import tinyinfer._moe_C_stable_libtorch  # noqa

names = torch._C._dispatch_get_all_op_names()
sel = sorted(n for n in names if n.startswith(("_C::", "_moe_C::")))
print("total dispatched ops:", len(names))
print("our two namespaces:", len(sel))
c = [n for n in sel if n.startswith("_C::")]
m = [n for n in sel if n.startswith("_moe_C::")]
print(f"  _C::{len(c)}   _moe_C::{len(m)}")
print("  _C:", [n.split("::")[1] for n in c])
print("  _moe_C:", [n.split("::")[1] for n in m])
