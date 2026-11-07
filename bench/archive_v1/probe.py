import httpx, os
B = "https://hf-mirror.com/antirez/deepseek-v4.1-flash-gguf/resolve/main/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf"
os.makedirs(r"E:\Develop\Deepseek\deepseek\dl_test", exist_ok=True)
c1 = httpx.Client(follow_redirects=False, timeout=60.0)
r1 = c1.get(B)
loc = r1.headers.get("location")
print("302:", r1.status_code, "loc:", bool(loc))
c2 = httpx.Client(follow_redirects=False, timeout=180.0)
with c2.stream("GET", loc, headers={"Range": "bytes=0-2097151"}) as r2:
    print("range status:", r2.status_code)
    n = 0
    p = r"E:\Develop\Deepseek\deepseek\dl_test\probe.bin"
    with open(p, "wb") as f:
        for ch in r2.iter_bytes(1024*1024):
            f.write(ch); n += len(ch)
    print("bytes:", n, "filesize:", os.path.getsize(p))
