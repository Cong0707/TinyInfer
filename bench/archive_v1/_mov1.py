import pathlib
p = pathlib.Path(r'E:\\Develop\\Deepseek\\tinyinfer\\src\\tinyinfer\\gemm.py')
t = p.read_text(encoding='utf-8')
old_head = 'def fp8_linear(x: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor, block: int = 32) -> torch.Tensor:\n    if (x.is_cuda'
new_head = ('def fp8_linear(x: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor, block: int = 32) -> torch.Tensor:\n'
            '    # Cast BEFORE the fast path decides anything. act_quant only has a native' + chr(10) +
            '    # kernel for bf16 input; handing it fp32 falls back to the tilelang one, which' + chr(10) +
            '    # raises ImportError on a deployment without tilelang and looks exactly like' + chr(10) +
            '    # "the fast path is broken". The cast used to sit below the branch.' + chr(10) +
            '    if x.dtype != torch.bfloat16:' + chr(10) +
            '        x = x.to(torch.bfloat16)' + chr(10) +
            '    if (x.is_cuda')
if old_head not in t:
    print('MISS')
else:
    t = t.replace(old_head, new_head, 1)
    # 删掉原来那处（现在在函数开头了）
    t = t.replace('    if x.dtype != torch.bfloat16:' + chr(10) + '        x = x.to(torch.bfloat16)' + chr(10) + '    # decode (M==1): the fused GEMV kernel',
                  '    # decode (M==1): the fused GEMV kernel', 1)
    p.write_text(t, encoding='utf-8', newline=chr(10))
    print('ok, count=', t.count('if x.dtype != torch.bfloat16'))