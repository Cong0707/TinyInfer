from pathlib import Path
p = Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/csrc/vl_msvc_compat.h")
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
out = []
i = 0
while i < len(lines):
    s = lines[i].strip()
    # drop the NAN/INFINITY blocks (and the comment line directly above them)
    if s in ("#ifndef NAN", "#ifndef INFINITY"):
        depth = 0
        j = i
        while j < len(lines):
            st = lines[j].strip()
            if st.startswith("#ifndef"):
                depth += 1
            elif st.startswith("#endif"):
                depth -= 1
                if depth == 0:
                    break
            j += 1
        # drop a preceding comment line that mentions them
        if out and "C99 NAN" in out[-1]:
            out.pop()
        i = j + 1
        continue
    out.append(lines[i]); i += 1
p.write_text("".join(out), encoding="utf-8")
t = "".join(out)
print("NAN present:", "#define NAN" in t, "INFINITY present:", "#define INFINITY" in t)