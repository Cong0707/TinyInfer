import io
# 1) the decoder needs a step that stops before argmax, so the backend can sample
p = r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\hf_graph.py"
s = io.open(p, encoding="utf-8").read()
marker = "    def drain(self) -> list[int]:"
add = '''    def replay_step(self) -> None:
        """Advance one step and leave the logits in logits_buf, without choosing a token.

        next_token() argmaxes; a sampling backend has to see the distribution, so this
        splits the step in two.
        """
        with torch.inference_mode():
            if self.graph is None:
                self._fwd()
            else:
                self.graph.replay()

    def set_token(self, tok: int) -> None:
        """Feed the token the caller sampled back in, and advance the position."""
        self.ids_buf.fill_(int(tok))
        self.pos_buf += 1

'''
assert "def replay_step" not in s
i = s.index(marker)
s = s[:i] + add + s[i:]
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("hf_graph patched")

# 2) the sampler itself
p2 = r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\server.py"
t = io.open(p2, encoding="utf-8").read()
anchor = "class QwenBackend(BaseBackend):"
sampler = '''class _Sampler:
    """Greedy / temperature / top-p / top-k sampling over one logits row.

    Kept device-side apart from the final choice: the row is 248320 wide for Qwen, so the
    sort and the cumulative sum stay on the card and only the chosen id round-trips.
    """

    def __init__(self, params: SamplingParams, torch_module) -> None:
        self.torch = torch_module
        self.temperature = float(params.temperature or 0.0)
        self.top_p = float(params.top_p or 1.0)
        self.top_k = int(params.top_k or 0)
        self.repetition_penalty = float(params.repetition_penalty or 1.0)
        self.generator = None
        if params.seed is not None:
            self.generator = torch_module.Generator(device="cuda")
            self.generator.manual_seed(int(params.seed))
        self._seen: list[int] = []

    def pick(self, logits: Any, generated: int) -> int:
        torch = self.torch
        row = logits[0]
        if self.repetition_penalty != 1.0 and self._seen:
            ids = torch.tensor(self._seen, dtype=torch.long, device=row.device).unique()
            picked = row[ids]
            row = row.clone()
            row[ids] = torch.where(picked > 0, picked / self.repetition_penalty,
                                   picked * self.repetition_penalty)
        if self.temperature <= 0.0:
            tok = int(row.argmax())
            self._seen.append(tok)
            return tok
        scores = row.float() / self.temperature
        if self.top_k > 0:
            k = min(self.top_k, scores.numel())
            thresh = torch.topk(scores, k).values[-1]
            scores = torch.where(scores < thresh, float("-inf"), scores)
        if 0.0 < self.top_p < 1.0:
            ordered, order = torch.sort(scores, descending=True)
            probs = torch.softmax(ordered, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            cut = cumulative - probs > self.top_p
            ordered = torch.where(cut, float("-inf"), ordered)
            scores = torch.full_like(scores, float("-inf")).scatter(0, order, ordered)
        probs = torch.softmax(scores, dim=-1)
        tok = int(torch.multinomial(probs, 1, generator=self.generator))
        self._seen.append(tok)
        return tok


'''
assert "class _Sampler" not in t
i2 = t.index(anchor)
t = t[:i2] + sampler + t[i2:]
io.open(p2, "w", encoding="utf-8", newline="").write(t)
print("sampler added")
