// GPU 侧的缺专家请求 + 就绪标志。
//
// 分工：GPU 是买家，只负责说「我这一层缺哪几个」（写进 host-mapped 队列，
// 零拷贝，CPU 立刻能看见）；CPU 是卖家，负责备货，备好一批就把 dev_ready
// 加一（cuStreamWriteValue32，一次流内写，没有 kernel）；GPU 在真正读专家
// 权重之前用 cuStreamWaitValue32 等这个计数到位。
//
// 为什么这样比「CPU 每层等一次计数」好：原来的做法是每层一次 D2H 同步，
// 40 层就是 40 次全设备停顿（实测 130ms/token）。把等待排进流里之后，
// CPU 布置完就继续发下一层的 kernel，等待在 GPU 上完成，主机侧零停顿。
// GPU 侧的就绪屏障：等 CPU 把 dev_ready 写到 target 再继续。
//
// 用 kernel 的形式（而不是纯 cuStreamWaitValue32）是为了能同时兼容
// 「驱动不支持 32 位等待值」的老卡：退化成自旋等待同一块 host-mapped 内存，
// 语义一样，只是占一个 SM。正常路径下 CPU 侧直接用 stream_wait_value32。
extern "C" __global__ void moe_ready_spin(
    const int* __restrict__ dev_ready,
    int target) {
    if (threadIdx.x != 0) return;
    while (*(volatile int*)dev_ready < target) {
        __nanosleep(64);
    }
}

// 把 miss 计数从 slot_ids 里数出来（已有 moe_miss_scan 走 device buffer；
// 这个版本直接写 host-mapped 计数，省掉一次 D2H）。
extern "C" __global__ void moe_miss_count(
    const int* __restrict__ slot_ids,
    int n_exp,
    int* __restrict__ out) {
    if (threadIdx.x != 0) return;
    int n = 0;
    for (int e = 0; e < n_exp; ++e) if (slot_ids[e] < 0) ++n;
    *out = n;
}

// GPU 自己检查「本层缺不缺专家」，缺就写进 host-mapped 环，然后等 CPU 备好。
//
// 这样一处改动去掉的是每层一次 D2H 同步（40 层 x 3.3ms = 130ms/token）：
// 以前 CPU 每层把 miss 计数拉回主机才知道要不要搬，现在是 GPU 在流里等
// 一个 host-mapped 计数器 —— 主机侧完全不需要知道「缺没缺」。
//
// q 的布局（int32，host-mapped）：
//   [0]  发布计数（atomicAdd）
//   [1]  最新序号
//   [2]  就绪计数（CPU 写，GPU 等）
//   [3..] 请求记录，每条 34 个 int: [layer, n, e0..e31, epoch]
extern "C" __global__ void moe_wait_ready(
    const int* __restrict__ slot_ids,
    int n_exp,
    int layer,
    int epoch,
    volatile int* __restrict__ q,
    int cap) {
    if (threadIdx.x != 0) return;
    // 1) 扫出本层缺的专家（slot_ids 本来就是 device 上的）
    int n = 0;
    int miss[32];
    for (int e = 0; e < n_exp && n < 32; ++e) {
        if (slot_ids[e] < 0) miss[n++] = e;
    }
    if (n == 0) return;              // 全在显存里：直接走，不等任何人
    // 2) 发布请求给 CPU（host-mapped，写完 CPU 立刻看得到）
    int idx = atomicAdd((int*)&q[0], 1);
    __threadfence();
    int* rec = (int*)q + 3 + (idx % cap) * 34;
    rec[0] = layer;
    rec[1] = n;
    for (int i = 0; i < n; ++i) rec[2 + i] = miss[i];
    rec[33] = epoch;
    __threadfence();
    q[1] = idx + 1;
    // 3) 等 CPU 把这一批备好（就绪计数 >= 请求序号 + 1）
    int target = idx + 1;
    while (*(volatile int*)&q[2] < target) {
        __nanosleep(128);
    }
    __threadfence();
}
extern "C" __global__ void moe_req_post(
    const int* __restrict__ slot_ids,
    int n_exp,
    int layer,
    int epoch,
    int* __restrict__ q) {
    // q 的布局（int32，host-mapped）:
    //   [0]            发布计数（atomicAdd）
    //   [1]            最新一条的序号（CPU 只读）
    //   [2]            就绪计数（CPU 写，GPU 等）
    //   [3..]          每条 34 个 int: [layer, n, e0..e31, epoch]
    __shared__ int s_miss[32];
    if (threadIdx.x == 0) {
        int n = 0;
        for (int e = 0; e < n_exp; ++e) {
            if (slot_ids[e] < 0 && n < 32) s_miss[n++] = e;
        }
        if (n > 0) {
            int idx = atomicAdd(q, 1);          // 领一条槽
            __threadfence();
            // 环头 3 个 int（发布计数 / 最新序号 / 就绪计数），记录从第 4 个开始。
            // 这个 3 必须和 expert_queue.ExpertRequestRing.HEAD 一致。
            int* rec = q + 3 + (idx % 64) * 34;
            rec[0] = layer;
            rec[1] = n;
            for (int i = 0; i < n; ++i) rec[2 + i] = s_miss[i];
            rec[33] = epoch;                     // 发布序号的副本，CPU 用来判断新旧
            __threadfence();
            q[1] = idx + 1;                      // 让 CPU 知道有新东西
        }
    }
}