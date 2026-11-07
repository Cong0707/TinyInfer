
import pathlib
lines = open(r"E:\Develop\Deepseek\deepseek\_cu.txt", encoding="utf-8").read()
pathlib.Path(r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\kernels\cuda\sparse_attn_sink.cu").write_text(lines, encoding="utf-8", newline="\n")
print("written", len(lines))
