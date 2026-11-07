// GPU 侧的缺专家请求：把「本层缺哪几个专家」写进一个 device 队列。
//
// CPU 不再主动来问（那是一次同步 D2H）；反过来，是 GPU 把请求写出去，CPU 用一次
// 异步 D2H 把它取回来。取回来的时机是**下一次 forward 的开头** —— 那时上一次的
// GPU 工作早已完成，读那块 pinned 内存是免费的，不构成停顿。
//
// 队列布局（device int32）：
//   [0]            count（本层 miss 的专家数）
//   [1..CAP]       mgr 用的 layer
//   [1+CAP..2CAP]  expert
// 由调用方一次分配，两层共用同一块。
extern "C" __global__ void moe_req_probe(
    const int* __restrict__ slot_ids,
    int n_exp,
    int layer,
    int* __restrict__ q,
    int cap) {
    if (threadIdx.x != 0) return;
    int miss = 0;
    int* ql = q + 1;
    int* qe = q + 1 + cap;
    for (int e = 0; e < n_exp; ++e) {
        if (slot_ids[e] >= 0) continue;
        if (miss < cap) {
            ql[miss] = layer;
            qe[miss] = e;
        }
        ++miss;
    }
    q[0] = miss;
}
